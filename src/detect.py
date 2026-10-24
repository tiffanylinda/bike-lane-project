"""Headless batch detector: turn recorded bike-lane video into per-video detections CSVs.

WHY THIS SCRIPT EXISTS
----------------------
The rest of the team works in pandas, not in video. Decoding video and running a
YOLO model is slow and heavy, so we do it exactly once per clip (in Google Colab,
on free GPUs) and write the result to a plain CSV. Everyone else reads those CSVs
and never has to touch the raw footage again.

INPUTS
------
- ``--input``: a single ``.mp4`` file OR a folder of ``.mp4`` files to process.
  Paths are explicit (typically Google Drive paths mounted in Colab); this script
  never assumes a local ``data/`` directory exists.
- A per-camera config at ``<config-dir>/<camID>.yaml`` (default config dir: ``config/``).
  The camera id is parsed from the filename pattern ``commave_<camID>_YYYY-MM-DD_HHMM.mp4``.
  Model weights, confidence threshold, and detection classes all come from this file,
  so nothing is hard-coded and the two cameras are never mixed together.

OUTPUTS
-------
- One detections CSV per video at ``<output>/<video_name>_detections.csv`` with a
  fixed column contract (see ``CSV_COLUMNS`` below) that downstream pandas code relies on.
- Optionally, with ``--save-annotated``, a blurred annotated ``.mp4`` at
  ``<output>/<video_name>_annotated.mp4`` with boxes and track IDs drawn, and faces and
  license plates blurred so a demo clip can be shared without exposing identities.

This module also holds small shared helpers (filename parsing, config loading) that the
point-picker tool imports, so that logic lives in exactly one place.
"""

# --- Standard library -------------------------------------------------------
# We lean on the standard library for everything that does not need heavy CV code,
# so the pure helpers stay fast and easy to unit-test without a real video or model.
import argparse  # the CLI: --input, --output, --config-dir, --stride, etc.
import re  # parse the camera id and start time out of the filename
import sys  # write errors to stderr and return a process exit code
import time  # time.monotonic() to throttle progress lines to once every 5 seconds
from datetime import datetime, timedelta  # build each detection's timestamp and offset it by frame/fps seconds
from pathlib import Path  # path handling that works the same on every OS
from zoneinfo import ZoneInfo  # make start times timezone-aware in America/New_York

# --- Third-party ------------------------------------------------------------
# pyyaml is tiny and we need it for the pure config helpers, so we import it at the
# top level. The HEAVY third-party libraries — ultralytics (YOLO), cv2 (OpenCV), and
# pandas — are slow to import and only needed once we actually process a video, so we
# import THOSE INSIDE the functions that use them (added in later tasks). That keeps
# importing this module for its pure helpers (filename parsing, config loading) cheap.
import yaml  # read the per-camera <camID>.yaml settings file (yaml.safe_load)


# The detections CSV column contract: the single source of truth for column order.
# Downstream pandas code reads every camera the same way, so this list must never be
# reordered or changed without telling the whole team. Every CSV row is built against it.
CSV_COLUMNS = ["video", "camera", "time", "frame", "track_id", "uid", "cls", "conf", "cx", "cy", "w", "h"]

# The one true filename pattern: ``commave_<camID>_YYYY-MM-DD_HHMM`` (e.g.
# ``commave_camA_2024-05-17_0830``). We compile it once at import time so every call to
# ``parse_video_name`` reuses the same regex instead of recompiling it. Named groups
# (``cam``, ``date``, ``time``) make the code below read like the pattern it matches.
# WHY a strict pattern: the camera id and the recording start time both come straight
# from the filename, so a name that does not fit this shape has no start time we can
# trust — better to fail loudly than to guess.
FILENAME_RE = re.compile(r"^commave_(?P<cam>[A-Za-z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{4})$")


def parse_video_name(video_path):
    """Pull the camera id and recording start time out of a video's filename.

    Takes a path (a ``str`` or a ``pathlib.Path``) and returns a
    ``(camera_id, start_time)`` tuple, where ``start_time`` is a timezone-aware
    ``datetime`` in America/New_York.

    WHY the filename is the source of truth: these clips are named at recording time,
    so the name is the only place the real wall-clock start lives. Every detection's
    timestamp is later built from this start time, so getting it right here matters.

    WHY the timezone matters: the cameras film a street in the Eastern US. If we left
    the start time "naive" (no timezone), later analysis — joining against Telraam
    counts, grouping by hour, plotting across a daylight-saving change — would silently
    be off. Anchoring to ``ZoneInfo("America/New_York")`` keeps every downstream
    timestamp honest.

    Raises ``ValueError`` (naming the offending filename) when the name does not match
    the ``commave_<camID>_YYYY-MM-DD_HHMM`` pattern, so a bad name stops this one video
    instead of producing a wrong start time.
    """
    # Accept either a str or a Path, and match against the *stem* only: no directory
    # and no extension. ``Path("/drive/commave_camA_2024-05-17_0830.mp4").stem`` is
    # exactly ``commave_camA_2024-05-17_0830``, which is what the pattern expects.
    stem = Path(video_path).stem
    match = FILENAME_RE.match(stem)
    if match is None:
        # Name the offending filename so a batch run tells you *which* clip was wrong.
        raise ValueError(f"Video filename does not match the expected pattern 'commave_<camID>_YYYY-MM-DD_HHMM': {stem!r}")

    camera_id = match.group("cam")
    date_part = match.group("date")  # "YYYY-MM-DD"
    time_part = match.group("time")  # "HHMM"

    # Build the start time from the date and HHMM parts. We parse the 4-digit time as
    # hours (first two) and minutes (last two), then attach the America/New_York zone so
    # the datetime is timezone-aware from the moment it is created.
    start_time = datetime(
        year=int(date_part[0:4]),
        month=int(date_part[5:7]),
        day=int(date_part[8:10]),
        hour=int(time_part[0:2]),
        minute=int(time_part[2:4]),
        tzinfo=ZoneInfo("America/New_York"),
    )

    return camera_id, start_time


def load_camera_config(config_dir, camera_id):
    """Load one camera's settings from ``<config_dir>/<camID>.yaml``.

    Takes the config directory (a ``str`` or a ``pathlib.Path``) and a camera id
    (for example ``"camA"``), and returns the loaded config as a ``dict``.

    WHY per-camera files: two cameras film the same street from different angles, and
    their counts must NEVER be combined. Giving each camera its own file —
    ``config/camA.yaml``, ``config/camB.yaml`` — makes that separation physical: there
    is no single shared config to accidentally mix them in. The id parsed from the
    video filename picks exactly one file, so every video is scored with its own camera's
    settings and nothing else.

    WHY load from config at all: model weights, the confidence threshold, and which
    object classes to keep are DECISIONS, not constants. They live in the YAML file so
    we can tune them per camera without editing code, and so nothing is hard-coded. The
    detector later reads ``model.weights`` (a str), ``model.conf`` (a float), and
    ``model.classes`` (a list of COCO class ids) straight out of what this returns.

    Raises ``FileNotFoundError`` (naming the expected path) when the config file does
    not exist (Req 2.4), so a typo'd camera id or a missing file stops this one video
    with a clear message instead of a confusing crash later. Raises ``KeyError`` (naming
    both the missing field and the config path) when any of ``model.weights``,
    ``model.conf``, or ``model.classes`` is absent (Req 2.5), so an incomplete config is
    caught up front rather than when the detector tries to use the missing value.
    """
    # Build the path the same way whether config_dir arrived as a str or a Path: wrap
    # it in Path() first so "/" path-joining always works. The filename is the camera id
    # plus ".yaml" — the one naming convention that ties a video's camID to its config.
    config_path = Path(config_dir) / f"{camera_id}.yaml"

    # Fail loudly and early if the file is not there. Naming the full expected path tells
    # the caller exactly where we looked, which is the fix (create that file or correct
    # the --config-dir) rather than a vague "not found".
    if not config_path.exists():
        raise FileNotFoundError(f"Camera config not found for camera {camera_id!r}: {config_path}")

    # safe_load (never plain load) builds only basic Python types from the YAML and will
    # not execute arbitrary tags — the safe default for reading a settings file.
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    # The three settings the detector cannot run without. We pull the nested model.<field>
    # values and, the moment one is missing, raise a KeyError that names BOTH the field and
    # the file — so you know what to add and exactly which config to add it to.
    model_section = config.get("model") if isinstance(config, dict) else None
    if not isinstance(model_section, dict):
        raise KeyError(f"Missing required config field 'model' in {config_path}")
    for field in ("weights", "conf", "classes"):
        if field not in model_section:
            raise KeyError(f"Missing required config field 'model.{field}' in {config_path}")

    # Return the whole loaded dict: the detector reads model.weights / model.conf /
    # model.classes from it, and other sections (counting_line, lane_polygon, ...) stay
    # available for the counting and blockage code added in later tasks.
    return config


def frame_time(start, frame_index, fps):
    """Build the timestamp for one frame as an ISO 8601 string with its UTC offset.

    Takes the recording ``start`` (a timezone-aware ``datetime`` in America/New_York,
    exactly as ``parse_video_name`` returns), the frame's ``frame_index`` (the TRUE
    source index, an int >= 0), and the video's ``fps`` (frames per second, a positive
    float), and returns an ISO 8601 string with the UTC offset at millisecond precision.

    WHY this math: a frame's real wall-clock time is simply when recording began plus
    how many seconds into the clip that frame sits. Frame ``frame_index`` arrives
    ``frame_index / fps`` seconds after the start, so the timestamp is
    ``start + frame_index/fps`` seconds. We MUST use the TRUE frame index here (not a
    compacted counter), so that a strided or time-limited run still reports when each
    frame actually happened.

    WHY keep it timezone-aware: ``start`` already carries the America/New_York zone, and
    adding a ``timedelta`` preserves that zone, so ``.isoformat()`` emits the UTC offset
    (for example ``-04:00``). Downstream analysis joins these timestamps against Telraam
    counts and groups by hour across daylight-saving changes, so dropping the offset
    would silently shift every time. ``timespec="milliseconds"`` keeps the string
    uniform and precise enough to order frames without noisy microsecond digits.
    """
    # start + (frame_index / fps) seconds. timedelta does the second-to-duration
    # conversion, and adding it to a tz-aware datetime returns another tz-aware datetime.
    moment = start + timedelta(seconds=frame_index / fps)
    # isoformat with the offset still attached; milliseconds is plenty of precision here.
    return moment.isoformat(timespec="milliseconds")


def build_uid(video_name, camera_id, track_id):
    """Join Video_Name, Camera_ID, and track_id into one globally unique id (Req 4.3).

    WHY a composite uid: YOLO's ``track_id`` only counts up within a single video, so the
    same integer id appears in every clip and on both cameras. Prefixing it with the
    video name and camera id makes the uid unique across the whole dataset, so two
    cameras filming the same street at the same moment never collide and downstream code
    can treat each uid as one real object.
    """
    return f"{video_name}_{camera_id}_{track_id}"


def should_process_frame(frame_index, fps, stride, max_seconds):
    """Decide whether one frame (by its TRUE source index) should be processed.

    Takes the frame's ``frame_index`` (the true index into the video, an int >= 0),
    the video's ``fps`` (frames per second, a positive float), the sampling ``stride``
    (an int >= 1, validated in ``parse_args``), and ``max_seconds`` (a positive float,
    or ``None`` for "no time limit"). Returns ``True`` iff this frame is one we keep.

    A frame is kept when BOTH of these hold:
    - ``frame_index % stride == 0`` -- it lands on the stride grid, and
    - ``max_seconds is None or frame_index / fps < max_seconds`` -- it falls inside
      the time window.

    WHY stride: decoding video and running the model is the slow part. ``--stride N``
    lets us sample every Nth frame (``N=1`` keeps every frame) so a long clip can be
    scored several times faster when we do not need every single frame.

    WHY max-seconds: ``--max-seconds N`` limits a run to roughly the first N seconds of
    footage. That makes a quick test pass over a clip cheap -- you see real output in
    seconds instead of waiting for the whole video. ``None`` means "process the whole
    clip". A frame sits ``frame_index / fps`` seconds into the video, so the cutoff is a
    strict ``< max_seconds`` comparison (frame 0 is at second 0.0 and is always in range).

    WHY the TRUE index: we pass the real source index here, never a compacted
    0,1,2,... counter. Keeping the true index is what lets ``frame_time`` compute the
    correct wall-clock time for every kept frame even when we skip most of them.
    """
    on_stride = frame_index % stride == 0
    in_time_window = max_seconds is None or frame_index / fps < max_seconds
    return on_stride and in_time_window


def select_frame_indices(total_frames, fps, stride, max_seconds):
    """Yield the TRUE frame indices to process out of ``total_frames`` frames.

    Walks the true source indices ``0, 1, ..., total_frames - 1`` and yields each one
    that ``should_process_frame`` keeps. The result is exactly the set the design calls
    for: ``{ i : i % stride == 0 and (max_seconds is None or i/fps < max_seconds) }``,
    with each index yielded as its own TRUE source index ``i`` (never renumbered).

    Takes ``total_frames`` (the video's frame count, an int >= 0), ``fps`` (positive
    float), ``stride`` (int >= 1), and ``max_seconds`` (positive float or ``None``).
    Yields ints in increasing order.

    WHY a generator: the detector loops over frames one at a time, so yielding indices
    lazily lets the caller pull the next index to process without first building a list
    of every frame in a long clip. The per-frame predicate and this set form are the
    same rule expressed two ways -- the predicate answers "keep this one?" inside the
    decode loop, and this generator enumerates the whole selection for the unit tests
    and for anywhere the full set is handier than a per-frame check.
    """
    for frame_index in range(total_frames):
        if should_process_frame(frame_index, fps, stride, max_seconds):
            yield frame_index

# The two class groups whose boxes get a privacy blur in the annotated video. We blur
# the TOP of a person (that is where a face sits) and the BOTTOM of a vehicle (that is
# where a license plate sits). Keeping these names here — matched to the COCO class
# names the model reports — means the dispatcher below and task 5.3's drawing code agree
# on exactly which detections to blur, and nothing is hard-coded in two places.
PERSON_CLASS = "person"  # COCO class 0: a human; we blur the top 25% to hide the face.
VEHICLE_CLASSES = frozenset({"car", "motorcycle", "bus", "truck"})  # we blur the bottom 30% to hide the plate.

# How much of each box we blur, as a fraction of the box HEIGHT.
FACE_BLUR_FRACTION = 0.25  # top 25% of a person box -> the face band.
PLATE_BLUR_FRACTION = 0.30  # bottom 30% of a vehicle box -> the plate band.


def face_blur_region(x1, y1, x2, y2):
    """Return the face sub-rectangle: the TOP 25% of a person box (Req 6.2, Property 10).

    BOX CONVENTION: a box is given as ``(x1, y1, x2, y2)`` in pixels, where ``(x1, y1)``
    is the TOP-LEFT corner and ``(x2, y2)`` is the BOTTOM-RIGHT corner. Image y grows
    DOWNWARD (y1 is the top edge, y2 is the bottom edge), which is how OpenCV and image
    arrays index rows. The returned sub-region uses the same ``(x1, y1, x2, y2)`` shape
    and the same convention, so task 5.3 can hand it straight to ``cv2.GaussianBlur``.

    WHY the top 25%: on a person detection the head — and therefore the face — sits at
    the top of the box. Blurring only that band hides the identity while leaving the rest
    of the person visible, which is exactly what the steering rule asks for: every
    exported image or video must have faces (and plates) blurred so we never expose who
    was on the street. We blur a band rather than the whole box so the annotated demo
    still shows what is happening.

    This helper is PURE geometry: it takes four numbers and returns four numbers. It
    never touches ``cv2`` or any pixels, so it is trivial to unit-test without a real
    image. The caller is the one that later blurs the pixels inside this rectangle.

    Returns ``(rx1, ry1, rx2, ry2)`` as ints (pixel coordinates). The band spans the
    full width of the box and the top ``FACE_BLUR_FRACTION`` of its height, and is
    clamped to stay within the box bounds so it can never spill outside the detection.
    """
    # Normalize the corners so the math does not care which way round the caller passed
    # them: top is the smaller y, bottom is the larger y (same idea for left/right x).
    top = min(y1, y2)
    bottom = max(y1, y2)
    left = min(x1, x2)
    right = max(x1, x2)

    height = bottom - top
    # The face band starts at the box top and extends down by 25% of the height. We clamp
    # the lower edge to ``bottom`` so a tiny or zero-height box can never produce an edge
    # below the box itself (the region always stays within the box bounds, Property 10).
    band_bottom = min(bottom, top + height * FACE_BLUR_FRACTION)

    # Round to ints because these index pixels. The x range is the full box width.
    return (int(round(left)), int(round(top)), int(round(right)), int(round(band_bottom)))


def plate_blur_region(x1, y1, x2, y2):
    """Return the plate sub-rectangle: the BOTTOM 30% of a vehicle box (Req 6.3, Property 10).

    BOX CONVENTION: identical to ``face_blur_region`` — ``(x1, y1, x2, y2)`` is the
    top-left/bottom-right box in pixels, y grows downward, and the result is the same
    ``(x1, y1, x2, y2)`` shape ready for ``cv2.GaussianBlur``.

    WHY the bottom 30%: a car, motorcycle, bus, or truck carries its license plate low —
    at the back/front bumper — so the plate lives in the bottom slice of the detection
    box. Blurring that band hides the plate (and with it the vehicle's identity) while
    leaving the rest of the vehicle visible for the demo, matching the steering rule that
    plates are blurred in every exported image or video. A vehicle plate band is a little
    taller (30%) than the face band (25%) because plates sit a bit further up from the
    very bottom edge than faces do from the very top.

    Like its sibling this is PURE geometry (four numbers in, four numbers out, no ``cv2``
    and no pixels), so it unit-tests without a real image.

    Returns ``(rx1, ry1, rx2, ry2)`` as ints. The band spans the full width of the box
    and the bottom ``PLATE_BLUR_FRACTION`` of its height, clamped within the box bounds.
    """
    # Normalize corners so the result is correct regardless of the input corner order.
    top = min(y1, y2)
    bottom = max(y1, y2)
    left = min(x1, x2)
    right = max(x1, x2)

    height = bottom - top
    # The plate band ends at the box bottom and starts 30% of the height above it. We
    # clamp the upper edge to ``top`` so a tiny box can never produce an edge above the
    # box itself — the region always stays within the box bounds (Property 10).
    band_top = max(top, bottom - height * PLATE_BLUR_FRACTION)

    # Round to ints for pixel use. The x range is the full box width.
    return (int(round(left)), int(round(band_top)), int(round(right)), int(round(bottom)))


def blur_region_for_class(cls_name, x1, y1, x2, y2):
    """Pick the right privacy blur sub-region for a detection, by its class name.

    Given the COCO class NAME (as it appears in the detections CSV / the model's
    ``names``) and the box ``(x1, y1, x2, y2)``, return the sub-rectangle to blur:

    - a person -> the face band from ``face_blur_region`` (top 25%),
    - a vehicle (car, motorcycle, bus, truck) -> the plate band from
      ``plate_blur_region`` (bottom 30%),
    - anything else -> ``None`` (we do not blur it).

    WHY a dispatcher: task 5.3 draws the annotated MP4 and needs, for each detection,
    "which rectangle do I blur, if any?". Centralizing that decision here keeps the
    drawing loop simple — it just calls this one function per box — and keeps the list of
    "what counts as a person vs a vehicle" in a single place (``PERSON_CLASS`` and
    ``VEHICLE_CLASSES``) so the two blur rules never drift apart. ``None`` means "leave
    this detection unblurred", which is the correct default for every class we are not
    asked to protect.

    Still PURE: it only calls the two pure geometry helpers, so it needs no image to test.
    """
    if cls_name == PERSON_CLASS:
        return face_blur_region(x1, y1, x2, y2)
    if cls_name in VEHICLE_CLASSES:
        return plate_blur_region(x1, y1, x2, y2)
    return None


def positive_int(value):
    """argparse ``type`` for ``--stride``: parse a string into an int that is >= 1.

    argparse hands every option value to us as a raw string. This function turns that
    string into an int and rejects anything that is not a whole number of 1 or greater,
    raising ``argparse.ArgumentTypeError`` whose message NAMES the offending value.
    argparse catches that exception, prints the message, and exits non-zero — so a bad
    ``--stride`` stops the run before any video is opened and nothing is ever written
    (Req 3.5, design "report-and-stop" for CLI errors).

    WHY stride must be an int >= 1: ``--stride N`` means "process every Nth frame". The
    frame loop walks the TRUE source indices and keeps the ones where ``i % N == 0``.
    ``N = 1`` keeps every frame (the default). A value below 1 — ``0`` would divide the
    work by zero frames, a negative N is meaningless — and a fractional value like ``1.5``
    cannot name "every Nth frame", so none of those describe a real sampling step. We
    reject them here, up front, instead of letting them cause confusing behavior deep in
    the decode loop.
    """
    # Parse as an int. We deliberately do NOT accept "1.5" or "abc": int() on either
    # raises ValueError, which we turn into a clear, value-naming CLI error. A float that
    # happens to be whole ("2.0") is also rejected — stride is conceptually a count of
    # frames, so we require it to be written as a plain integer.
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"--stride must be an integer >= 1, got: {value!r}")

    # Enforce the ">= 1" half of the contract. 0 and negatives are not valid strides.
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"--stride must be an integer >= 1, got: {value!r}")

    return parsed


def positive_float(value):
    """argparse ``type`` for ``--max-seconds``: parse a string into a float that is > 0.

    Like ``positive_int``, this runs on the raw string argparse passes in. It parses the
    value as a float and rejects anything that is not a strictly positive number, raising
    ``argparse.ArgumentTypeError`` whose message NAMES the offending value so argparse can
    print it and exit non-zero without writing any output (Req 13.4, design "report-and-stop").

    WHY max-seconds must be strictly positive: ``--max-seconds N`` limits a run to the
    first N seconds of footage — a frame ``i`` sits ``i / fps`` seconds in and is kept
    only while ``i / fps < N``. A zero or negative N would select NO frames (an empty
    run that writes an empty CSV for no reason), and a non-number cannot express a time
    limit at all. Requiring ``N > 0`` keeps the option meaning what it says: "process at
    least the opening slice of the clip". (Leaving the option off entirely, not passing
    0, is how you ask for the whole video — that is handled by the default of ``None``.)
    """
    # Parse as a float so values like "2.5" seconds are allowed. A non-numeric string
    # raises ValueError, which becomes a value-naming CLI error.
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"--max-seconds must be a positive number, got: {value!r}")

    # Strictly greater than zero: 0 and negatives would select no frames, so they are errors.
    # We also guard against NaN, which is not a usable time limit (NaN fails every compare).
    if not (parsed > 0):
        raise argparse.ArgumentTypeError(f"--max-seconds must be a positive number, got: {value!r}")

    return parsed


def parse_args(argv):
    """Parse the detector's command-line options from an explicit ``argv`` list.

    Takes ``argv`` — the list of command-line tokens WITHOUT the program name (exactly
    what you would get from ``sys.argv[1:]``) — and returns an ``argparse.Namespace`` with
    one attribute per option below. Taking ``argv`` as a parameter (instead of reading
    ``sys.argv`` directly) is what lets the unit tests drive this function with a plain
    list, so the CLI can be tested without actually launching a process.

    WHY each option exists:
    - ``--input`` (required): the single ``.mp4`` file OR the folder of ``.mp4`` files to
      score. Required because there is nothing to do without something to process, and
      paths are always explicit (typically Google Drive paths) so the script never guesses
      a location.
    - ``--output`` (required): the folder where the per-video detections CSVs (and, with
      ``--save-annotated``, the annotated MP4s) are written. Required for the same reason —
      we must be told where results go.
    - ``--config-dir`` (default ``"config/"``): where the per-camera ``<camID>.yaml`` files
      live. It has a sensible default because the configs normally sit in the repo's
      ``config/`` folder, but it stays overridable for runs where they live elsewhere.
    - ``--stride`` (int >= 1, default 1): sample every Nth frame to trade completeness for
      speed; 1 means "every frame". Validated by ``positive_int`` (see why there).
    - ``--max-seconds`` (positive number, optional): cap a run to the first N seconds for a
      quick test pass; omitted (``None``) means "the whole video". Validated by
      ``positive_float`` (see why there).
    - ``--overwrite`` (flag, default off): replace an existing detections CSV instead of
      skipping it. Off by default so a re-run never silently destroys earlier results.
    - ``--save-annotated`` (flag, default off): also write a blurred annotated MP4 for a
      shareable demo. Off by default because it is slower and usually not needed.

    WHY validation lives in the ``type=`` functions: a bad ``--stride`` or ``--max-seconds``
    is a whole-run CLI error — the design's "report-and-stop" policy says to report it and
    stop WITHOUT writing anything. Because ``positive_int`` / ``positive_float`` run during
    parsing (before this function returns, and long before any video is opened), an invalid
    value makes argparse print an error naming the value and exit non-zero on its own, so no
    output is ever produced (Req 3.5, 13.4).
    """
    parser = argparse.ArgumentParser(
        prog="detect.py",
        description="Headless batch detector: turn recorded bike-lane video into per-video detections CSVs.",
    )

    # The two paths we cannot run without. Both are required so the script never has to
    # guess what to read or where to write.
    parser.add_argument(
        "--input",
        required=True,
        help="Single .mp4 file OR a folder of .mp4 files to process.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for detections CSVs (and annotated MP4s with --save-annotated).",
    )

    # Where the per-camera <camID>.yaml configs live. Defaults to the repo's config/ folder.
    parser.add_argument(
        "--config-dir",
        default="config/",
        help="Directory holding the per-camera <camID>.yaml files (default: config/).",
    )

    # Frame sampling step. type=positive_int rejects anything that is not an int >= 1,
    # naming the bad value and stopping the run before anything is written.
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=1,
        help="Process every Nth frame; integer >= 1 (default: 1).",
    )

    # Optional time cap. type=positive_float rejects anything that is not a positive
    # number. No default value means it stays None when omitted, which means "whole video".
    parser.add_argument(
        "--max-seconds",
        type=positive_float,
        default=None,
        help="Process only the first N seconds; positive number (default: whole video).",
    )

    # Two on/off switches. store_true gives each a default of False, so the safe behavior
    # (do not overwrite, do not spend time rendering a video) is what you get unless asked.
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing detections CSV instead of skipping it (default: off).",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Also write a blurred annotated MP4 (default: off).",
    )

    # parse_args on the EXPLICIT argv list (not sys.argv), which is what makes this testable.
    # argparse turns "--config-dir" into args.config_dir, "--max-seconds" into args.max_seconds,
    # and "--save-annotated" into args.save_annotated (dashes become underscores).
    return parser.parse_args(argv)


# How often, in wall-clock seconds, we print a progress line while decoding a video.
# We throttle with time.monotonic() (not a frame counter) so the cadence is steady no
# matter how fast or slow the model runs on a given machine (Req 7.1).
PROGRESS_INTERVAL_SECONDS = 5.0


def process_video(video_path, output_dir, cfg, camera_id, start, *, stride, max_seconds, overwrite, save_annotated):
    """Score one video and write its per-video detections CSV. Returns a summary dict.

    This is the heart of the detector: it decodes a single clip, runs YOLO tracking on
    the frames we choose to keep, turns every tracked box into one CSV row following the
    fixed ``CSV_COLUMNS`` contract, and writes ``<output_dir>/<video_name>_detections.csv``.

    PARAMETERS
    ----------
    - ``video_path``: the clip to process (a ``str`` or ``pathlib.Path``). Its *stem*
      (no directory, no extension) becomes the ``video`` column and part of every uid.
    - ``output_dir``: the folder the CSV is written into (created if absent).
    - ``cfg``: the loaded per-camera config dict from ``load_camera_config`` — we read
      ``cfg["model"]["weights"]`` (the YOLO weights), ``cfg["model"]["conf"]`` (the
      confidence threshold), and ``cfg["model"]["classes"]`` (the COCO class ids to keep).
    - ``camera_id``: the camera id parsed from the filename. It is written verbatim into
      the ``camera`` column of EVERY row, which is how we guarantee the two cameras are
      never combined (Req 2.2) — a video's rows all carry its own camera and no other.
    - ``start``: the recording start time (tz-aware America/New_York) from
      ``parse_video_name``; every row's ``time`` is computed from this plus the frame offset.
    - ``stride`` (keyword-only, int >= 1): process every Nth frame.
    - ``max_seconds`` (keyword-only, positive float or ``None``): cap the run to the first
      N seconds; ``None`` means the whole clip.
    - ``overwrite`` (keyword-only, bool): accepted here for a stable signature; the real
      skip/overwrite branch lands in task 5.2. See the SEAM note below.
    - ``save_annotated`` (keyword-only, bool): accepted here for a stable signature; the
      annotated-MP4 writing lands in task 5.3. See the SEAM note below.

    RETURNS a per-video summary dict with keys ``frames_processed`` (int),
    ``total_detections`` (int), ``unique_tracks_per_class`` (dict class-name -> count of
    distinct track ids), and ``skipped`` (bool). The same numbers are also printed as a
    human-readable summary line (Req 7.2).

    WHY the heavy imports are INSIDE this function: ultralytics (YOLO), OpenCV, and pandas
    are slow to import and only needed once we actually decode a video. Keeping them out of
    the module top level means importing ``src.detect`` for its pure helpers (filename
    parsing, config loading) stays cheap and never pulls in the whole CV/ML stack. This is
    the deliberate design noted in the module's import comments.

    SKIP/OVERWRITE (Req 5.1, 5.2, 5.3): before any heavy work, if the target CSV already
    exists and ``overwrite`` is off, this reports the named video, returns a summary with
    ``skipped=True`` (and zero/empty counts), and leaves the existing CSV untouched. When
    ``overwrite`` is on (or no CSV exists yet) the video is processed and any existing CSV
    is replaced by the write below. A skip returns cleanly rather than raising, so the
    batch loop (task 5.4) just continues to the next clip.

    ANNOTATED MP4 (Req 6): when ``save_annotated`` is set, this also writes a blurred
    annotated video to ``<output_dir>/<video_name>_annotated.mp4``. For every detection
    written to the CSV it draws the box and the track id, and for privacy it blurs faces
    (top 25% of a person box) and license plates (bottom 30% of a vehicle box) so a demo
    clip can be shared without exposing identities (the steering rule: faces/plates are
    blurred in EVERY exported video). If the MP4 cannot be written, the error names the
    target path and the CSV is still written (Req 6.5). Without the flag, only the CSV is
    written and no video file is ever created (Req 6.4). No GUI calls anywhere (Req 9.1).
    """
    # --- Heavy, lazy imports (see the WHY above) ---------------------------------
    from ultralytics import YOLO  # the detector/tracker; loading weights is expensive.
    import cv2  # OpenCV: decode the video frame by frame.
    import pandas as pd  # assemble and write the CSV in the exact column order.

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    video_name = video_path.stem  # the Video_Name: no directory, no extension (Req 4 contract).

    # The target CSV path. Downstream code always looks for "<video_name>_detections.csv",
    # so this name is part of the contract, not an arbitrary choice.
    csv_path = output_dir / f"{video_name}_detections.csv"

    # --- Skip/overwrite handling (Req 5.1, 5.2, 5.3) ----------------------------
    # WHY we check this FIRST, before loading the model or opening the video: decoding a
    # clip and running YOLO on it is the slow, expensive part of the whole pipeline. If
    # we already produced this video's CSV on an earlier run and the caller did NOT ask to
    # redo it (--overwrite is off), there is nothing to gain by re-scoring it — the result
    # would be the same file. Returning here, up front, means a finished video costs almost
    # nothing on a re-run: we never load weights, never open the stream, never touch the
    # existing CSV. That is exactly what lets a big batch be re-run cheaply (Req 5.1).
    #
    # The rule (design Property 9): process the video IFF `overwrite or not csv_path.exists()`.
    # So we skip only when the CSV already exists AND overwrite is off. When overwrite is on
    # (or the CSV is not there yet) we fall through and process normally; the pandas write
    # further down already overwrites any existing file, which satisfies Req 5.2.
    if csv_path.exists() and not overwrite:
        # Report, by name, that THIS video was skipped so a batch log makes it obvious which
        # clips were reused vs freshly scored. We name the CSV too so it is clear what is
        # being kept untouched.
        print(f"[{video_name}] skipped: detections CSV already exists, leaving it unchanged (use --overwrite to redo) -> {csv_path}")

        # Return cleanly with skipped=True and zero/empty values for everything else, so the
        # batch loop (task 5.4) can count this as a skipped video and the folder summary math
        # stays simple. We return instead of raising so a skip never stops the batch — the
        # loop just moves on to the next clip (Req 5.3). The summary dict keeps the SAME shape
        # as the normal return below so main() can rely on it unconditionally.
        return {
            "frames_processed": 0,
            "total_detections": 0,
            "unique_tracks_per_class": {},
            "skipped": True,
        }

    # We are going to process this video (either the CSV did not exist, or --overwrite is on).
    skipped = False

    # Pull the three model settings out of config. These are DECISIONS living in the YAML
    # (weights, threshold, which classes to keep), never hard-coded here.
    model_cfg = cfg["model"]
    weights = model_cfg["weights"]
    conf_threshold = model_cfg["conf"]
    classes = model_cfg["classes"]

    # Load the YOLO model ONCE for this whole video. Reloading weights per frame would be
    # absurdly slow; one load up front is reused for every frame we track.
    model = YOLO(weights)

    # Open the clip. cv2.VideoCapture is how OpenCV decodes frames; we read fps and the
    # total frame count straight off the stream so progress and timestamps are honest.
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        # CAP_PROP_FRAME_COUNT can be a float and is occasionally unreliable; we coerce to
        # int and clamp negatives to 0 so a bad count means "unknown total", not a crash.
        total_frames_raw = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(total_frames_raw) if total_frames_raw and total_frames_raw > 0 else 0

        # Guard against a missing/zero fps. Every timestamp and the max-seconds cutoff
        # divide by fps, so fps <= 0 (some containers report it as 0) would poison all the
        # math. We fail loudly with a clear message rather than silently writing wrong times.
        if not fps or fps <= 0:
            raise ValueError(
                f"Could not read a valid frame rate (fps) from video {video_path} "
                f"(got {fps!r}); cannot compute frame timestamps."
            )

        rows = []  # one dict per tracked detection, in CSV_COLUMNS order.
        frames_processed = 0  # count of frames we actually ran the model on.
        # Per class name, the SET of distinct track ids seen — len() of each set at the end
        # gives the "unique track count per class" the summary reports (Req 7.2).
        unique_tracks_per_class = {}

        # Progress throttle: remember when we last printed, and only print again once at
        # least PROGRESS_INTERVAL_SECONDS of wall-clock time have passed (Req 7.1).
        last_progress = time.monotonic()

        # --- Annotated MP4 state (Req 6) ----------------------------------------
        # We only produce an annotated video when the caller asked for it with
        # --save-annotated; otherwise none of this runs and no .mp4 is ever created (Req 6.4).
        # The target path follows the fixed "<video_name>_annotated.mp4" naming so a demo
        # clip is easy to find next to its CSV.
        annotated_path = output_dir / f"{video_name}_annotated.mp4"
        # The cv2.VideoWriter is created LAZILY on the first frame we process, because only
        # then do we know the real frame size (width/height from the decoded frame's shape).
        # fps is already read above. None means "not created yet".
        writer = None
        # annotated_ok tracks whether the annotated video is still being written
        # successfully. If anything goes wrong (the writer will not open, or an error is
        # raised while drawing/writing), we flip this to False, report the target path, and
        # keep going so the DETECTIONS CSV is never lost to a video failure (Req 6.5).
        annotated_ok = save_annotated
        if save_annotated:
            # Make sure the output folder exists before the first write attempt. We also
            # create it again before the CSV write below; doing it here keeps the writer
            # creation self-contained.
            output_dir.mkdir(parents=True, exist_ok=True)

        # Decode by TRUE source index i = 0, 1, 2, ... We read (or grab) EVERY frame so the
        # true index stays aligned with the real video, but only run the model on the frames
        # the stride + max-seconds rule selects. Keeping the true index is what lets
        # frame_time report when each kept frame actually happened (Req 13.1, 13.2, 13.5).
        i = 0
        while True:
            # Stop early once we are past the time window: frame i sits i/fps seconds in,
            # and everything after the cutoff is also past it, so there is nothing left to
            # keep — no point decoding the rest of a long clip (Req 3.4).
            if max_seconds is not None and (i / fps) >= max_seconds:
                break

            if should_process_frame(i, fps, stride, max_seconds):
                # A frame we keep: fully DECODE it (read gives us the pixels) and track.
                ok, frame = capture.read()
                if not ok:
                    break  # end of stream (or an unreadable frame): stop cleanly.

                # Run tracking. persist=True keeps track ids stable across frames (the same
                # object keeps the same id). We pass the config classes and conf so the model
                # itself only returns the classes we want at/above the threshold; verbose=False
                # keeps ultralytics from flooding stdout and drowning our progress lines.
                results = model.track(
                    frame,
                    persist=True,
                    classes=classes,
                    conf=conf_threshold,
                    verbose=False,
                )

                # For the annotated video we collect, per frame, two things:
                #  - blur_boxes: EVERY person/vehicle detection's corner box, so we can blur
                #    faces/plates even on detections the tracker has not given an id yet.
                #    Privacy must NOT depend on tracking (Req 6.2, 6.3).
                #  - draw_boxes: the (corner box, label) for each detection written to the
                #    CSV, so the drawn boxes/track-id labels match the CSV rows (Req 6.1).
                # Both stay empty (and cheap) when save_annotated is off.
                blur_boxes = []  # list of (cls_name, x1, y1, x2, y2)
                draw_boxes = []  # list of (x1, y1, x2, y2, label)

                # results is a list (one entry per image); we passed a single frame, so we
                # read result[0]. A frame with no detections still returns a result with an
                # empty boxes set, which the loop below simply skips.
                for result in results:
                    boxes = getattr(result, "boxes", None)
                    if boxes is None or len(boxes) == 0:
                        continue

                    # model.names maps class id -> COCO class NAME (e.g. 0 -> "person").
                    # We write the NAME, never the raw id, so the CSV is human-readable and
                    # downstream code filters on names (Req 4.4).
                    names = model.names

                    # Each box carries: a track id, a class id, a confidence, and an xywh
                    # (center x, center y, width, height) in pixels. We pull them per box.
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])

                        # Defensive threshold filter (Req 3.3 / Property 5): we already asked
                        # the model for conf>=threshold, but we re-check here so a row can
                        # never slip below the configured confidence even if the model's own
                        # filtering changes. No at-or-above detection is dropped.
                        if confidence < conf_threshold:
                            continue

                        # xywh is center-x, center-y, width, height in PIXELS. This is exactly
                        # the cx, cy, w, h the CSV contract wants (not corner coords).
                        cx, cy, w, h = (float(v) for v in box.xywh[0])

                        # Map the class id to its name; fall back to the raw id as a string if
                        # the model somehow lacks the name, so we never crash on a stray class.
                        cls_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]

                        # PRIVACY FIRST: record this detection for blurring BEFORE the track-id
                        # check. A face or a plate must be hidden whether or not the tracker has
                        # committed to an id yet, so we never let a missing id leak an identity
                        # into the exported video (Req 6.2, 6.3). The corner box is derived from
                        # the xywh center/size: left = cx - w/2, top = cy - h/2, etc.
                        if save_annotated:
                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            blur_boxes.append((cls_name, x1, y1, x2, y2))

                        # A box without a track id (the tracker has not committed to one yet)
                        # is not a stable tracked detection, so it is not written to the CSV and
                        # gets no drawn box/label — every CSV row must have a real integer track
                        # id for its uid to mean anything. (It was already queued for blurring
                        # above, so privacy is preserved regardless.)
                        if box.id is None:
                            continue

                        track_id = int(box.id[0])

                        # Build the uid from video + camera + track id so it is unique across
                        # the whole dataset (track ids repeat across clips and cameras).
                        uid = build_uid(video_name, camera_id, track_id)

                        # One row, values placed to match CSV_COLUMNS. camera is ALWAYS
                        # camera_id (cameras never combined). time uses the TRUE index i.
                        # Numeric fields are rounded exactly as the contract requires:
                        # conf to 3 decimals, the box geometry to 1 decimal.
                        rows.append(
                            {
                                "video": video_name,
                                "camera": camera_id,
                                "time": frame_time(start, i, fps),
                                "frame": i,
                                "track_id": track_id,
                                "uid": uid,
                                "cls": cls_name,
                                "conf": round(confidence, 3),
                                "cx": round(cx, 1),
                                "cy": round(cy, 1),
                                "w": round(w, 1),
                                "h": round(h, 1),
                            }
                        )

                        # Queue the drawn box + label for the annotated video. We draw the box
                        # in CORNER coords (same derivation as the blur box) and label it with
                        # the track id (and class for readability) so the picture matches the
                        # CSV rows (Req 6.1).
                        if save_annotated:
                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            draw_boxes.append((x1, y1, x2, y2, f"{cls_name} #{track_id}"))

                        # Track this id under its class name for the unique-per-class summary.
                        unique_tracks_per_class.setdefault(cls_name, set()).add(track_id)

                # --- Annotated MP4 writing (Req 6.1, 6.2, 6.3, 6.5) --------------
                # Everything below is guarded by save_annotated AND annotated_ok, so when the
                # flag is off we write nothing (Req 6.4), and once a video failure has been
                # reported we stop trying but keep decoding so the CSV still completes (Req 6.5).
                if save_annotated and annotated_ok:
                    try:
                        # Lazily create the writer on the first processed frame, now that we
                        # know the real frame size from the decoded frame's shape (rows, cols).
                        # mp4v is a widely-playable fourcc for .mp4 output.
                        if writer is None:
                            frame_height, frame_width = frame.shape[0], frame.shape[1]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (frame_width, frame_height))
                            # VideoWriter does not raise when it fails to open; it returns an
                            # object whose isOpened() is False. Treat that as a write failure so
                            # we degrade gracefully instead of silently producing an empty file.
                            if not writer.isOpened():
                                raise IOError(f"cv2.VideoWriter could not open {annotated_path}")

                        # Work on a COPY so our boxes/labels/blur never alter the frame the
                        # model saw (not that we reuse it, but it keeps intent clear).
                        annotated = frame.copy()
                        annotated_h, annotated_w = annotated.shape[0], annotated.shape[1]

                        # BLUR FIRST, DRAW SECOND. We blur the privacy sub-regions before
                        # drawing boxes/labels so the blurred pixels (faces/plates) are what
                        # ends up UNDER the drawn box — the box outline sits on top of an
                        # already-anonymized region, so an identity can never show through.
                        for cls_name, bx1, by1, bx2, by2 in blur_boxes:
                            # Pick the right privacy sub-rectangle for this class: top 25% of a
                            # person (the face), bottom 30% of a vehicle (the plate), or None
                            # for classes we do not blur.
                            region = blur_region_for_class(cls_name, bx1, by1, bx2, by2)
                            if region is None:
                                continue
                            rx1, ry1, rx2, ry2 = region
                            # Clamp to the frame bounds so slicing can never run off the image
                            # (a box can sit partly outside the frame edges).
                            rx1 = max(0, min(int(rx1), annotated_w))
                            rx2 = max(0, min(int(rx2), annotated_w))
                            ry1 = max(0, min(int(ry1), annotated_h))
                            ry2 = max(0, min(int(ry2), annotated_h))
                            if rx2 <= rx1 or ry2 <= ry1:
                                continue  # empty region after clamping: nothing to blur.
                            sub = annotated[ry1:ry2, rx1:rx2]
                            # A strong Gaussian blur over the sub-region hides the detail. We
                            # size the kernel to the region (odd, >= 3) so small and large boxes
                            # are both effectively anonymized.
                            kx = max(3, (rx2 - rx1) // 2 * 2 + 1)
                            ky = max(3, (ry2 - ry1) // 2 * 2 + 1)
                            annotated[ry1:ry2, rx1:rx2] = cv2.GaussianBlur(sub, (kx, ky), 0)

                        # Now draw the tracked boxes + track-id labels ON TOP of the (already
                        # blurred) frame, so the annotations match the CSV rows (Req 6.1).
                        for dx1, dy1, dx2, dy2, label in draw_boxes:
                            px1 = max(0, min(int(round(dx1)), annotated_w))
                            py1 = max(0, min(int(round(dy1)), annotated_h))
                            px2 = max(0, min(int(round(dx2)), annotated_w))
                            py2 = max(0, min(int(round(dy2)), annotated_h))
                            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
                            # Put the label just above the box (or just below the top edge if
                            # the box hugs the top of the frame) so it stays readable.
                            text_y = py1 - 6 if py1 - 6 > 6 else py1 + 16
                            cv2.putText(annotated, label, (px1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        writer.write(annotated)
                    except Exception as annotate_error:
                        # Degrade gracefully (Req 6.5): report the target MP4 path by name, stop
                        # trying to write the video, but DO NOT re-raise — the detections CSV
                        # below must still be written and the normal summary returned.
                        print(
                            f"[{video_name}] warning: could not write annotated video -> {annotated_path}: {annotate_error}",
                            file=sys.stderr,
                        )
                        annotated_ok = False
                        # Release a half-open writer so we do not leak the handle.
                        if writer is not None:
                            writer.release()
                            writer = None

                frames_processed += 1

                # Throttled progress line: at most once every PROGRESS_INTERVAL_SECONDS.
                now = time.monotonic()
                if now - last_progress >= PROGRESS_INTERVAL_SECONDS:
                    total_label = total_frames if total_frames else "?"
                    print(f"[{video_name}] processed {frames_processed} frames (at true frame {i} / {total_label})")
                    last_progress = now
            else:
                # A frame we are NOT keeping: GRAB it (advance the decoder) without the cost
                # of fully decoding the pixels. This keeps the true index i aligned with the
                # real stream so skipped-over frames do not shift later timestamps.
                if not capture.grab():
                    break  # no more frames to advance past: end of stream.

            i += 1
    finally:
        # Always release the capture so the file handle / decoder is freed even if an
        # error bubbled up mid-decode.
        capture.release()
        # Release the annotated-video writer too (if one was created), so the .mp4 is
        # flushed/finalized and its handle freed even when an error bubbled up mid-decode.
        # `writer` is defined before the loop whenever we reached it; guard with locals()
        # in case an early error (e.g. bad fps) happened before it was set.
        if "writer" in locals() and writer is not None:
            writer.release()

    # Make sure the output folder exists before writing (no assumption that a local data/
    # dir is present — we create the target folder ourselves).
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the DataFrame with an EXPLICIT columns=CSV_COLUMNS so the header is always the
    # 12-column contract in the right order, even when there are zero rows (an empty clip
    # still produces a valid, correctly-headed CSV). index=False keeps pandas' row numbers
    # out of the file. This is the one write of the detections CSV for this video.
    frame_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    frame_df.to_csv(csv_path, index=False)

    # Collapse the per-class sets of track ids into plain counts for the summary/return.
    unique_counts = {cls_name: len(ids) for cls_name, ids in unique_tracks_per_class.items()}
    total_detections = len(rows)

    # Human-readable per-video summary (Req 7.2): what we processed and found.
    print(
        f"[{video_name}] done: {frames_processed} frames processed, "
        f"{total_detections} detections, unique tracks per class: {unique_counts} "
        f"-> {csv_path}"
    )

    # The machine-readable summary the batch loop (task 5.4) will aggregate. skipped is
    # False here; task 5.2 sets it True on the skip path.
    return {
        "frames_processed": frames_processed,
        "total_detections": total_detections,
        "unique_tracks_per_class": unique_counts,
        "skipped": skipped,
    }


def main(argv=None):
    """Dispatch a file or folder of videos, score each one, and return a process exit code.

    This is the detector's top-level entry point: it ties the pure helpers and
    ``process_video`` together into a single run. It takes ``argv`` — the command-line
    tokens WITHOUT the program name (exactly what ``sys.argv[1:]`` gives). When ``argv``
    is ``None`` we fall back to ``sys.argv[1:]`` so the real CLI works, while tests can
    still drive ``main`` with an explicit list and never touch the real process args.

    RETURNS an int process exit code: ``0`` when the run completed (even if some videos
    were skipped or individually errored and were reported), and non-zero ONLY for the
    whole-run stop conditions below.

    TWO ERROR POLICIES (see design "Error Handling"):

    - **Report-and-STOP (non-zero exit, nothing written):** a missing ``--input`` path.
      There is simply nothing to do — we cannot invent videos to score — so we name the
      missing path on stderr and bail out before creating any output. (Bad ``--stride`` /
      ``--max-seconds`` are the same kind of whole-run error, but argparse already caught
      those inside ``parse_args`` and exited non-zero before we ever get here.)

    - **Report-and-CONTINUE (exit 0, batch keeps going):** a single video that will not
      parse, has no config, is missing a config field, or cannot be decoded. WHY continue:
      detection runs are long overnight batches over a folder of clips. If one corrupt or
      mis-named file killed the whole run, you would lose a night's work to a single bad
      clip. Instead we name the offending file/path/field, skip just that one video, and
      move on — so the other 99 clips still get their CSVs.
    """
    # Default to the real process args when called as a script; take the explicit list
    # when a test (or any caller) passes one. This is the same testability trick parse_args
    # uses — the CLI can be exercised without launching a process.
    if argv is None:
        argv = sys.argv[1:]

    # parse_args validates --stride / --max-seconds via their type= functions; an invalid
    # value makes argparse print an error naming it and exit non-zero on its own, before
    # this returns. So by the time we have args, the numeric options are known-good.
    args = parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # --- Report-and-STOP: a missing --input path (Req 1.4) ----------------------
    # If the path we were told to read does not exist, there is nothing to process. We
    # name the missing path on stderr (so the fix — correct the path — is obvious) and
    # return non-zero WITHOUT creating the output folder or writing anything. This is the
    # whole-run stop: a missing input is a user/CLI mistake, not a per-video hiccup.
    if not input_path.exists():
        print(f"error: --input path does not exist: {input_path}", file=sys.stderr)
        return 1

    # --- Resolve the batch: one file, or every .mp4 directly in a folder (Req 1.1, 1.2) --
    if input_path.is_dir():
        # Folder mode: every file with a .mp4 extension DIRECTLY inside the folder, NOT
        # recursive (Req 1.2). We match the suffix case-insensitively (".MP4" counts too)
        # and sort the result so a batch processes clips in a stable, predictable order.
        is_folder = True
        videos = sorted(
            child
            for child in input_path.iterdir()
            if child.is_file() and child.suffix.lower() == ".mp4"
        )
    else:
        # File mode: the batch is just the single file we were handed (Req 1.1).
        is_folder = False
        videos = [input_path]

    # Create the output folder up front if it is missing (Req 1.3). process_video also
    # ensures it before writing, but making it here means an empty folder run (no .mp4s)
    # still leaves the destination ready, and we never assume a local data/ dir exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Batch tallies for the final folder summary (Req 7.3):
    #  - processed: videos that were scored and got a fresh CSV.
    #  - skipped:   videos left untouched because their CSV already existed (--overwrite off).
    #  - errored:   videos reported-and-skipped for a bad name/config/decode. These are
    #               NEITHER cleanly processed NOR "skipped due to an existing CSV", so we
    #               count them on their own so the summary math stays honest.
    videos_processed = 0
    videos_skipped = 0
    videos_errored = 0

    for video in videos:
        # Each video runs inside its own try/except so ONE bad clip never stops the batch
        # (report-and-continue, Req 1.5, 2.3, 2.4, 2.5, 5.3). Every except arm names the
        # offending file/path/field on stderr and then falls through to the next clip.
        try:
            # Parse the camera id + recording start out of the filename. A name that does
            # not fit the commave_<camID>_YYYY-MM-DD_HHMM pattern raises ValueError (Req 2.3).
            camera_id, start = parse_video_name(video)

            # Load that camera's config. A missing file raises FileNotFoundError naming the
            # expected path (Req 2.4); a missing weights/conf/classes field raises KeyError
            # naming the field and the config path (Req 2.5).
            cfg = load_camera_config(args.config_dir, camera_id)

            # Score the clip. process_video handles skip/overwrite, the CSV write, and the
            # optional annotated MP4, and returns a summary whose "skipped" flag tells us
            # whether an existing CSV was left in place.
            summary = process_video(
                video,
                args.output,
                cfg,
                camera_id,
                start,
                stride=args.stride,
                max_seconds=args.max_seconds,
                overwrite=args.overwrite,
                save_annotated=args.save_annotated,
            )

            # Count this clip as skipped (existing CSV kept) or freshly processed, using the
            # summary's own flag so the tally always matches what process_video actually did.
            if summary.get("skipped"):
                videos_skipped += 1
            else:
                videos_processed += 1

        except ValueError as exc:
            # Two report-and-continue cases land here:
            #  - an unparseable filename (parse_video_name, Req 2.3), and
            #  - an unreadable/undecodable video (process_video raises ValueError when it
            #    cannot read a valid fps, which also stands in for any decode failure, Req 1.5).
            # Both name the offending file in their message; we print it and skip this clip.
            print(f"error: skipping {video}: {exc}", file=sys.stderr)
            videos_errored += 1
        except FileNotFoundError as exc:
            # The camera config for this clip's camId is missing (Req 2.4). The message
            # already names the expected config path. Skip and continue.
            print(f"error: skipping {video}: {exc}", file=sys.stderr)
            videos_errored += 1
        except KeyError as exc:
            # The config loaded but is missing a required model.<field> (Req 2.5). The
            # KeyError message names both the field and the config path. Skip and continue.
            print(f"error: skipping {video}: {exc}", file=sys.stderr)
            videos_errored += 1
        except Exception as exc:
            # Any other failure while opening/decoding/scoring one clip (e.g. an OpenCV
            # error) is still just one bad video (Req 1.5) — report it by name and keep the
            # batch alive rather than letting an unexpected error take down the whole run.
            print(f"error: skipping {video}: {exc}", file=sys.stderr)
            videos_errored += 1

    # --- Final folder summary (Req 7.3) -----------------------------------------
    # Only meaningful for a folder run: print how many clips were processed vs skipped
    # (and, as a nicety, how many errored out and were skipped). A single-file run does
    # not need a folder-level tally, so we keep stdout quiet there.
    if is_folder:
        print(
            f"folder summary: {videos_processed} processed, "
            f"{videos_skipped} skipped, {videos_errored} errored"
        )

    # The batch completed. Per the design's Error Handling, a completed batch exits 0 even
    # when some videos were skipped or individually errored and were reported — the only
    # non-zero exits are the whole-run stop conditions handled above (missing --input, and
    # the bad --stride/--max-seconds that argparse already stopped on).
    return 0


if __name__ == "__main__":
    # raise SystemExit(main()) makes the process exit code equal main()'s return value, so
    # the shell/Colab sees 0 on a completed run and non-zero on a whole-run stop.
    raise SystemExit(main())
