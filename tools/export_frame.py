"""Export a video's first frame as a PNG so it can be downloaded and picked on a laptop.

WHY THIS SCRIPT EXISTS
----------------------
Picking the counting line and lane polygon is a point-and-click job that needs a real
GUI, so it happens on a laptop (see ``tools/pick_points.py``). But the raw videos live
on Google Drive and detection runs in Google Colab, where opening a GUI window is not
practical and downloading a whole multi-gigabyte clip just to grab one frame is wasteful.

So in Colab we run this tiny script to pull the FIRST frame out of a video and save it
as a single PNG. You download that small PNG, open it on your laptop, and pick points
against it — no full video download, no GUI in Colab. It is deliberately simple and
headless: it never calls ``cv2.imshow`` or opens any window.

INPUTS
------
- ``--video``: path to the input video (required). Typically a Google Drive path mounted
  in Colab; this script never assumes a local ``data/`` directory exists.
- ``--output``: where to write the PNG (required). This may be either an exact PNG file
  path OR a folder. When it names a folder, the frame is written inside it as
  ``<Video_Name>.png`` (the video filename without its directory or extension).

OUTPUTS
-------
- A single PNG of the video's first frame at the resolved output path.

On a missing or unreadable video, or if the first frame cannot be read, the script
reports an error that names the path and writes nothing.
"""

# --- Standard library -------------------------------------------------------
# These are cheap to import and keep the path logic easy to reason about.
import argparse  # the CLI: --video and --output
import sys  # write errors to stderr and return a process exit code
from pathlib import Path  # path handling that works the same on every OS

# --- Third-party ------------------------------------------------------------
# OpenCV (cv2) is the heavy import here — it is what actually decodes the video frame.
# We import it LAZILY inside export_frame() rather than at module top level so that
# importing this module (for example, to unit-test the pure path rule) stays cheap and
# does not require OpenCV to be installed. cv2 is already a project dependency.


def _resolve_output_path(video_path, output):
    """Decide the exact PNG file path to write to, applying the folder rule (Req 11.2).

    WHY this is its own function: the "is --output a file or a folder?" decision is pure
    path logic with no video or OpenCV involved, so it is easy to reason about and test on
    its own, separate from the frame-reading code.

    ``--output`` is treated as a FOLDER when any of these is true:
      - it already exists as a directory, OR
      - it ends with a path separator (e.g. ``out/``), OR
      - it has no file suffix (e.g. ``out`` with no ``.png``).
    In the folder case we write ``<Video_Name>.png`` inside it, where Video_Name is the
    video filename's stem (no directory, no extension). Otherwise ``--output`` is taken as
    the exact PNG file path to write.
    """
    video_path = Path(video_path)
    output = Path(output)

    # ``os.fspath`` via str() lets us peek at the trailing character the user typed. A
    # trailing separator ("out/") is a strong signal they mean a folder even if it does
    # not exist yet, so we honour it before touching the filesystem.
    ends_with_sep = str(output).endswith(("/", "\\"))

    is_folder = output.is_dir() or ends_with_sep or output.suffix == ""
    if is_folder:
        # Video_Name is the filename with no directory and no extension, matching the
        # rest of the pipeline's naming (e.g. commave_camA_2024-05-17_0830).
        return output / f"{video_path.stem}.png"

    # Otherwise the user gave us the exact PNG path they want.
    return output


def export_frame(video_path, output):
    """Read a video's first frame and write it as a PNG, returning the written path.

    PARAMETERS
    - ``video_path``: path to the input video (str or Path).
    - ``output``: either an exact PNG file path or a folder (str or Path). The folder rule
      (Req 11.2) is applied by ``_resolve_output_path``.

    BEHAVIOR
    - Opens the video and reads its FIRST frame via ``cv2.VideoCapture.read()``.
    - Creates the output folder if it does not exist (Req 11.3).
    - Writes the frame as a PNG via ``cv2.imwrite`` and returns the resolved Path (Req 11.1).

    ERRORS (Req 11.4): if the video is missing or cannot be opened, or its first frame
    cannot be read, this raises an error whose message NAMES the offending path and writes
    nothing. ``main`` catches that, prints it, and returns a non-zero exit code.
    """
    video_path = Path(video_path)

    # Fail fast with a clear, path-naming error when the file simply is not there. This is
    # the common Colab mistake (wrong Drive path), so naming the path saves a lot of guessing.
    # We do this BEFORE importing OpenCV so a missing/unreadable path is reported as a clean,
    # path-naming error regardless of the environment (Req 11.4).
    if not video_path.exists():
        raise FileNotFoundError(f"input video does not exist: {video_path}")

    # Work out where the PNG should land BEFORE we spend time decoding, so a bad output
    # value surfaces quickly. This is pure path logic (the folder-vs-file rule, Req 11.2).
    output_path = _resolve_output_path(video_path, output)

    # Lazy import: see the note at the top of the module — this keeps importing the module
    # cheap and only pulls in OpenCV when we actually decode a frame. cv2 is a project
    # dependency (requirements.txt), so in Colab it is always present.
    import cv2

    # Open the video. cv2 will not raise on a bad/unsupported file — it just reports that
    # the capture never opened — so we check isOpened() and name the path ourselves (Req 11.4).
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video (unreadable or unsupported): {video_path}")

        # ``read()`` returns (ok, frame). The very first call gives us the FIRST frame.
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"could not read the first frame of video: {video_path}")
    finally:
        # Always release the capture, even if reading failed, so we never leak the handle.
        capture.release()

    # Create the output folder if absent (Req 11.3). parents=True handles nested folders;
    # exist_ok=True means a pre-existing folder is fine and not an error.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the PNG. cv2.imwrite picks the encoder from the ".png" suffix. It returns False
    # on failure (e.g. a path the OS refuses), so we check and name the path (Req 11.4).
    if not cv2.imwrite(str(output_path), frame):
        raise ValueError(f"could not write PNG to: {output_path}")

    return output_path


def main(argv=None):
    """Parse CLI arguments, export the first frame, and return a process exit code.

    Returns 0 on success and a non-zero code on any error. All errors are reported to
    stderr with a message that names the offending path, so a failed run is easy to debug
    from the Colab output (Req 11.4).
    """
    parser = argparse.ArgumentParser(
        prog="export_frame",
        description="Save a video's first frame as a PNG (headless; for Colab -> laptop point picking).",
    )
    # --video: the clip to grab a frame from. Required — there is no sensible default.
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video.",
    )
    # --output: the PNG file path OR a folder. When a folder, we write <Video_Name>.png into it.
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG file path, or a folder (then written as <Video_Name>.png).",
    )
    args = parser.parse_args(argv)

    try:
        output_path = export_frame(args.video, args.output)
    except (FileNotFoundError, ValueError, OSError) as exc:
        # Report and stop with a non-zero exit code; nothing is written on these paths.
        # The exception messages already name the offending path (Req 11.4).
        print(f"export_frame: error: {exc}", file=sys.stderr)
        return 1

    # Success: tell the user exactly where the PNG landed so they can download it.
    print(f"wrote first frame -> {output_path}")
    return 0


# Make the script runnable both as ``python3 -m tools.export_frame ...`` and as
# ``python3 tools/export_frame.py ...``. Both routes land here and hand off to main(),
# whose return value becomes the process exit code.
if __name__ == "__main__":
    sys.exit(main())
