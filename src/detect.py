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
