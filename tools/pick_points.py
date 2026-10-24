"""Interactive local tool: click the counting line and lane polygon on a frame.

WHY THIS TOOL EXISTS
--------------------
Every camera config (``config/<camID>.yaml``) needs two things that are impossible to
guess by hand: a **counting line** (two pixel points drawn across the bike lane) and a
**lane polygon** (4+ pixel points outlining the lane on the ground). Eyeballing pixel
coordinates from a screenshot is painful and error-prone. This tool shows you the actual
frame, lets you *click* those points with the mouse, and then prints ready-to-paste YAML
in exactly the shape the config file expects.

WHY IT IS LOCAL AND INTERACTIVE (NOT HEADLESS)
----------------------------------------------
``src/detect.py`` runs on Google Colab and must never open a GUI window (Req 9.1). This
tool is the opposite: it is meant to run on your laptop with a real display, because the
whole point is to *see* the frame and click on it (Req 9.2). That split is deliberate —
heavy detection uses free cloud compute; one-time point picking uses a local GUI.

INPUTS
------
Exactly ONE of:
- ``--video PATH``: open the first frame of that video and click on it.
- ``--image PATH``: open that PNG/JPG (e.g. a frame exported from Drive) and click on it.
Passing neither or both is an error (Req 8.1).

OUTPUTS
-------
- YAML printed to **stdout** with ``counting_line.points`` (2 points) and ``lane_polygon``
  (the clicked points), matching ``config/camA.yaml`` so you can paste it straight in.
- A copy of the frame with the line and polygon **drawn on it**, saved to ``data/frames/``
  as ``<source_stem>_points.png`` so you have a visual record of what you picked.

Nothing is printed or saved if the source can't be read, there's no display, or you
cancel before picking enough points (Req 8.2, 8.3, 9.3).

HOW TO RUN
----------
From the repo root (so ``import src.detect`` works):
    python3 tools/pick_points.py --video data/raw_video/commave_camA_2024-05-17_0830.mp4
    python3 -m tools.pick_points --image data/frames/commave_camA_2024-05-17_0830.png
"""

# --- Standard library -------------------------------------------------------
# argparse builds the tiny CLI; os lets us check display environment variables on Linux;
# sys gives us stderr + an exit code; pathlib handles paths the same way on every OS.
import argparse
import os
import sys
from pathlib import Path

# --- Third-party ------------------------------------------------------------
# cv2 (OpenCV) does the image loading, the GUI window, the mouse callback, and the
# drawing. numpy is the array type OpenCV hands us for the image. yaml lets us emit YAML
# in exactly the nested shape the config file uses, so there's no hand-built string to
# get wrong. We import these at module top level because this tool genuinely needs them
# to do anything useful (unlike detect.py, which defers its heavy imports).
import cv2
import numpy as np
import yaml

# --- Shared project helper --------------------------------------------------
# Camera-id parsing lives in exactly ONE place, the detector, so the picker and the
# detector can never disagree about what a filename means. We reuse it here to infer the
# camera id from the source filename (Req 8.5).
#
# WHY this sys.path line: when you run ``python3 -m tools.pick_points`` the repo root is
# already importable and ``import src.detect`` just works. But when you run the file
# directly as ``python3 tools/pick_points.py``, Python puts the *tools/* folder on the
# path instead of the repo root, so ``src`` can't be found. Adding the repo root (the
# parent of this file's folder) to ``sys.path`` makes BOTH invocation styles work, which
# is exactly what the module docstring promises.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detect import parse_video_name


# How many points each stage needs. These bounds come straight from Req 8.3: the counting
# line is always exactly 2 points, and the lane polygon is anywhere from 4 to 20 points.
LINE_POINT_COUNT = 2
POLY_MIN_POINTS = 4
POLY_MAX_POINTS = 20

# Where the annotated frame is saved. A plain default keeps the common case one command.
DEFAULT_OUT_DIR = Path("data/frames")

# Drawing colors (OpenCV uses BGR, not RGB). Kept here so the on-screen preview and the
# saved PNG look identical and the intent is obvious at a glance.
_COLOR_LINE = (0, 0, 255)      # red counting line
_COLOR_POLY = (0, 255, 0)      # green lane polygon
_COLOR_POINT = (0, 255, 255)   # yellow dots on each clicked point


def ensure_display() -> None:
    """Stop early with a clear error if there's no interactive display (Req 9.3).

    WHY check first: opening a cv2 window with no display fails deep inside OpenCV with a
    cryptic message. We'd rather tell the user plainly, *before* touching any GUI call,
    that this tool needs a real screen — it is the interactive counterpart to the detector
    being fully headless.

    WHY the OS split: on macOS and Windows a desktop session always has a display, so we
    assume one is present. On Linux a GUI only exists if an X11 (``DISPLAY``) or Wayland
    (``WAYLAND_DISPLAY``) server is advertised in the environment; a bare SSH session or a
    CI box has neither, which is exactly the "no display" case we must refuse.
    """
    if sys.platform.startswith("linux"):
        display = os.environ.get("DISPLAY", "").strip()
        wayland = os.environ.get("WAYLAND_DISPLAY", "").strip()
        if not display and not wayland:
            raise RuntimeError(
                "No interactive display found (DISPLAY and WAYLAND_DISPLAY are both unset). "
                "tools/pick_points.py needs a GUI to click on; run it on a local machine with a screen."
            )
    # macOS ("darwin") and Windows ("win32"): assume a desktop display is available.


def load_base_image(args: argparse.Namespace) -> np.ndarray:
    """Load the frame to click on: a video's first frame OR an image file (Req 8.2).

    Returns the image as a numpy/BGR array. Raises an error that *names the path* when the
    source is missing, unreadable, or its frame/image can't be decoded, so the user knows
    exactly which file to check — and so ``main`` can stop without saving anything.
    """
    if args.video is not None:
        video_path = Path(args.video)
        # Check existence ourselves so the message names the path; VideoCapture would just
        # silently fail to open and read() would return no frame, which is harder to debug.
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Could not open video (is it a valid .mp4?): {video_path}")
            # read() returns (ok, frame); ok is False at end-of-stream or on a decode error.
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Could not read the first frame of the video: {video_path}")
            return frame
        finally:
            # Always release the capture handle, even if reading raised, so we don't leak it.
            capture.release()

    # Otherwise we're in --image mode (main guarantees exactly one of the two is set).
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    # imread returns None (not an exception) for an unreadable or non-image file.
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read the image (unsupported or corrupt file?): {image_path}")
    return image


def _draw_overlay(base: np.ndarray, line_pts, poly_pts, stage: str) -> np.ndarray:
    """Return a copy of ``base`` with the points/line/polygon drawn so far drawn on it.

    WHY a copy: we redraw the whole overlay on every mouse event and keypress so the user
    sees live feedback, but we must never scribble on the original frame — otherwise an
    'undo' couldn't erase anything. Drawing on a fresh copy each time keeps undo trivial.
    """
    canvas = base.copy()

    # Counting line: dots on each point, plus the connecting segment once both are placed.
    for (x, y) in line_pts:
        cv2.circle(canvas, (x, y), 5, _COLOR_POINT, -1)
    if len(line_pts) == 2:
        cv2.line(canvas, line_pts[0], line_pts[1], _COLOR_LINE, 2)

    # Lane polygon: dots on each point, segments between consecutive points, and a dashed-
    # looking closing hint once there are at least 3 points so the shape reads as an area.
    for (x, y) in poly_pts:
        cv2.circle(canvas, (x, y), 5, _COLOR_POINT, -1)
    for i in range(1, len(poly_pts)):
        cv2.line(canvas, poly_pts[i - 1], poly_pts[i], _COLOR_POLY, 2)
    if len(poly_pts) >= 3:
        cv2.line(canvas, poly_pts[-1], poly_pts[0], _COLOR_POLY, 1)

    # A short on-image instruction so the user never has to remember the key bindings.
    hint = {
        "line": f"LINE: click {LINE_POINT_COUNT} points | u=undo  Enter=next  Esc=cancel",
        "poly": f"POLYGON: click {POLY_MIN_POINTS}-{POLY_MAX_POINTS} points | u=undo  Enter=finish  Esc=cancel",
    }[stage]
    cv2.putText(canvas, hint, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def collect_points(image: np.ndarray):
    """Open a GUI window and collect the counting-line then lane-polygon points (Req 8.3, 9.2).

    Flow: first the user clicks exactly 2 counting-line points, then 4-20 lane-polygon
    points. Left-click adds a point, ``u`` undoes the last one, ``Enter`` finishes the
    current stage, ``Esc`` cancels everything. Returns ``(line_pts, poly_pts)``; on cancel
    or an incomplete stage it returns whatever was collected so the caller can detect the
    shortfall and save nothing (Req 8.3).
    """
    window = "pick_points"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    line_pts: list = []
    poly_pts: list = []
    # Mutable state the mouse callback writes into. A dict avoids Python's "can't assign to
    # an outer variable from a nested function" gotcha without reaching for ``nonlocal``.
    state = {"stage": "line", "cancelled": False}

    def on_mouse(event, x, y, flags, param):
        # Only left-button clicks add a point; everything else (moves, right-clicks) is
        # ignored so stray mouse activity never pollutes the picked coordinates.
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["stage"] == "line":
            if len(line_pts) < LINE_POINT_COUNT:
                line_pts.append((int(x), int(y)))
        elif state["stage"] == "poly":
            if len(poly_pts) < POLY_MAX_POINTS:
                poly_pts.append((int(x), int(y)))

    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            current = line_pts if state["stage"] == "line" else poly_pts
            cv2.imshow(window, _draw_overlay(image, line_pts, poly_pts, state["stage"]))
            # Poll keys ~every 20ms so the window stays responsive to clicks and redraws.
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # Esc -> cancel the whole picking session.
                state["cancelled"] = True
                break
            if key == ord("u"):  # undo the most recent point in the current stage.
                if current:
                    current.pop()
                continue
            if key in (13, 10):  # Enter / Return -> finish the current stage.
                if state["stage"] == "line":
                    # Only advance once the line is complete; otherwise keep collecting.
                    if len(line_pts) == LINE_POINT_COUNT:
                        state["stage"] = "poly"
                    # If the line stage isn't done, Enter does nothing (user keeps clicking).
                else:
                    # Polygon stage: Enter finishes the whole session. Validity (4-20
                    # points) is judged by the caller, so we just stop collecting here.
                    break
    finally:
        # Always tear the window down, even on an exception, so no ghost window lingers.
        cv2.destroyWindow(window)
        # A few extra waitKey pumps make sure the window actually closes on macOS.
        for _ in range(4):
            cv2.waitKey(1)

    if state["cancelled"]:
        # Signal a cancel as "nothing collected" so the caller reports incomplete and saves
        # nothing, regardless of how many points were clicked before Esc.
        return [], []
    return line_pts, poly_pts


def to_yaml(line_pts, poly_pts) -> str:
    """Render the picked points as YAML matching ``config/camA.yaml`` (Req 8.4).

    The target shape (copy-paste ready) is::

        counting_line:
          points:
          - - 200
            - 400
          - - 700
            - 400
        lane_polygon:
        - - 100
          - 700
        ...

    WHY ``yaml.safe_dump``: hand-building YAML text is easy to get subtly wrong (indent,
    quoting). Dumping a plain Python dict guarantees it parses back to the same structure,
    which is exactly what the config loader will read. We coerce every coordinate to a
    plain ``int`` so no numpy types leak into the output.
    """
    data = {
        "counting_line": {
            "points": [[int(x), int(y)] for (x, y) in line_pts],
        },
        "lane_polygon": [[int(x), int(y)] for (x, y) in poly_pts],
    }
    # sort_keys=False keeps counting_line before lane_polygon, matching the config file.
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def save_annotated_frame(image, line_pts, poly_pts, out_dir: Path = DEFAULT_OUT_DIR, stem: str = "frame") -> Path:
    """Draw the final line + polygon on the frame and save it as a PNG (Req 8.4).

    Saves to ``<out_dir>/<stem>_points.png``, creating ``out_dir`` if it doesn't exist, and
    returns the written path. This gives the user a visual record of exactly what they
    picked, next to the YAML they'll paste into the config.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas = image.copy()
    # Draw the completed counting line (red) with its endpoint dots.
    if len(line_pts) == 2:
        cv2.line(canvas, line_pts[0], line_pts[1], _COLOR_LINE, 2)
    for (x, y) in line_pts:
        cv2.circle(canvas, (x, y), 5, _COLOR_POINT, -1)
    # Draw the closed lane polygon (green). np.int32 is what cv2.polylines expects.
    if len(poly_pts) >= 2:
        pts = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=_COLOR_POLY, thickness=2)
    for (x, y) in poly_pts:
        cv2.circle(canvas, (x, y), 5, _COLOR_POINT, -1)

    out_path = out_dir / f"{stem}_points.png"
    cv2.imwrite(str(out_path), canvas)
    return out_path


def _infer_camera(source_path: Path) -> None:
    """Warn about the matching camera config inferred from the filename (Req 8.5).

    Best-effort and never fatal: if the name follows the ``commave_<camID>_...`` pattern we
    point the user at ``config/<camID>.yaml`` (and warn if that file is missing); if it
    doesn't (e.g. a downloaded PNG with an arbitrary name) we just say we couldn't infer it.
    Either way the picking continues — inferring the camera is a convenience, not a gate.
    """
    try:
        camera_id, _start = parse_video_name(source_path)
    except ValueError:
        print(
            f"WARNING: could not infer the camera id from '{source_path.name}'. "
            "Paste the YAML below into the correct config/<camID>.yaml by hand.",
            file=sys.stderr,
        )
        return

    config_path = Path("config") / f"{camera_id}.yaml"
    if config_path.exists():
        print(f"Detected camera '{camera_id}'. Paste the YAML below into {config_path}.", file=sys.stderr)
    else:
        print(
            f"WARNING: detected camera '{camera_id}' but {config_path} does not exist. "
            "Create it (or check the camera id) before pasting the YAML below.",
            file=sys.stderr,
        )


def parse_args(argv=None) -> argparse.Namespace:
    """Build the CLI: exactly one of --video / --image is required (Req 8.1)."""
    parser = argparse.ArgumentParser(
        prog="pick_points",
        description=(
            "Interactively click the counting line and lane polygon on a video's first "
            "frame or on an image, then print config-ready YAML and save a drawn frame."
        ),
    )
    parser.add_argument("--video", metavar="PATH", help="video file; pick on its first frame")
    parser.add_argument("--image", metavar="PATH", help="image file (PNG/JPG); pick on it directly")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Wire the steps together; return a process exit code (0 ok, non-zero on any problem)."""
    args = parse_args(argv)

    # Req 8.1: exactly one source. XOR check — error and stop (saving nothing) on 0 or 2.
    if bool(args.video) == bool(args.image):
        print("ERROR: provide exactly one of --video or --image (not neither, not both).", file=sys.stderr)
        return 2

    # Req 9.3: refuse before opening any window if there's no display to open it on.
    try:
        ensure_display()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Req 8.2: load the frame/image, erroring by name if it can't be read. This runs
    # before any GUI window so an unreadable path fails fast and saves nothing.
    try:
        image = load_base_image(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    source_path = Path(args.video if args.video else args.image)
    # Req 8.5: best-effort camera inference (warnings only; never stops the run).
    _infer_camera(source_path)

    # Req 8.3 + 9.2: collect the points interactively.
    line_pts, poly_pts = collect_points(image)

    # Req 8.3: gate output on a complete collection. Fewer than 2 line points OR fewer than
    # 4 polygon points means the user cancelled or stopped early -> report and save nothing.
    if len(line_pts) != LINE_POINT_COUNT or not (POLY_MIN_POINTS <= len(poly_pts) <= POLY_MAX_POINTS):
        print(
            "INCOMPLETE: need exactly 2 counting-line points and 4-20 lane-polygon points; "
            f"got {len(line_pts)} line point(s) and {len(poly_pts)} polygon point(s). "
            "Nothing was printed or saved.",
            file=sys.stderr,
        )
        return 1

    # Req 8.4: print the config-ready YAML to stdout...
    print(to_yaml(line_pts, poly_pts))
    # ...and save the drawn frame next to it for a visual record.
    out_path = save_annotated_frame(image, line_pts, poly_pts, stem=source_path.stem)
    print(f"Saved annotated frame to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    # Return main()'s exit code to the shell so scripts/CI can tell success from failure.
    raise SystemExit(main())
