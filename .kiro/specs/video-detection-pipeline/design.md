# Design Document

## Overview

The video-detection-pipeline turns recorded bike lane clips into per-video detections CSVs so the
rest of the team works in pandas, never in video. Heavy CV runs once per clip in Google Colab;
analysis code reads the CSVs. The feature is four deliverables plus setup files:

- `src/detect.py` — headless Colab batch detector (video -> `<name>_detections.csv`, optional blurred annotated MP4).
- `tools/export_frame.py` — saves a video's first frame as a PNG for download from Drive.
- `tools/pick_points.py` — interactive local GUI to click the counting line and lane polygon.
- `notebooks/run_detection.ipynb` — a ready-to-run Colab notebook that ties a Drive run together.
- `requirements.txt`, `.gitignore` additions, and a README "How to run detection" section.

Design rules pulled from steering: never hard-code pixels, thresholds, or paths (they come from
config files and CLI args); every script has an argparse CLI and a docstring stating inputs and
outputs; faces and plates are blurred in every exported image or video; the detections CSV column
contract is fixed and never changed; cameras are never combined. The filename pattern is
`commave_<camID>_YYYY-MM-DD_HHMM.mp4` per requirements.md (this is authoritative over the older
pattern shown in tech.md).

## Architecture

Three independent scripts, no shared runtime. They communicate only through files on disk (CSVs,
PNGs) and a small shared config format. A tiny internal helper module avoids duplicating the
filename-parsing and config-loading logic between the detector and the point picker.

```
tools/pick_points.py (local GUI)  --prints YAML-->  config/<camID>.yaml  (hand-edited)
tools/export_frame.py (Colab)     --PNG-->          data/frames or Drive  --> pick_points --image
src/detect.py (Colab, headless)   --reads config/<camID>.yaml-->  <name>_detections.csv [+ _annotated.mp4]
notebooks/run_detection.ipynb     --orchestrates export_frame.py then detect.py for camA then camB-->
```

Shared helpers live in `src/detect.py` as small module-level functions (the picker imports them).
No new third-party dependency is introduced; everything uses the packages already listed in
tech.md and requirements.txt (ultralytics, opencv-python, pandas, numpy, pyyaml, pytest) plus the
Python standard library (`argparse`, `pathlib`, `re`, `datetime`, `zoneinfo`, `time`, `sys`).

## Components and Interfaces

### src/detect.py (Requirements 1, 2, 3, 4, 5, 6, 7, 9, 13)

Headless batch detector. CLI (argparse):

```
--input PATH          # single .mp4 file OR a folder of .mp4 files (required)
--output FOLDER       # output folder for CSVs / annotated MP4s (required)
--config-dir DIR      # where <camID>.yaml live (default: config/)
--stride N            # process every Nth frame, integer >= 1 (default: 1)
--max-seconds N       # process only the first N seconds, positive number (optional)
--overwrite           # replace an existing detections CSV (flag, default off)
--save-annotated      # also write a blurred annotated MP4 (flag, default off)
```

Key function signatures:

```python
def parse_args(argv: list[str]) -> argparse.Namespace: ...

# Filename -> (camera_id, start_time). Raises ValueError if the pattern does not match. (Req 2.1, 2.3)
# Pattern: commave_<camID>_YYYY-MM-DD_HHMM.mp4 ; start_time is tz-aware America/New_York.
FILENAME_RE = re.compile(r"^commave_(?P<cam>[A-Za-z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{4})$")
def parse_video_name(video_path: Path) -> tuple[str, datetime]: ...

# Loads <config_dir>/<camID>.yaml. Raises FileNotFoundError (Req 2.4) or
# KeyError naming the missing field (Req 2.5) for weights/conf/classes.
def load_camera_config(config_dir: Path, camera_id: str) -> dict: ...

# Build the iso timestamp for a true frame index. (Req 4.2, 13.2)
def frame_time(start: datetime, frame_index: int, fps: float) -> str: ...  # ISO 8601 + offset, ms precision

# The 12-column contract, in order. Single source of truth. (Req 4.1)
CSV_COLUMNS = ["video","camera","time","frame","track_id","uid","cls","conf","cx","cy","w","h"]

# Process one video -> write one CSV (and optional annotated MP4). Returns a per-video summary dict. (Req 1,3,4,5,6,7,13)
def process_video(video_path: Path, output_dir: Path, cfg: dict, camera_id: str,
                  start: datetime, *, stride: int, max_seconds: float | None,
                  overwrite: bool, save_annotated: bool) -> dict: ...

def main(argv: list[str]) -> int: ...  # folder/file dispatch, batch loop, folder summary
```

Behavior notes:

- **Input dispatch (Req 1):** if `--input` is a file, process it; if a folder, process every `.mp4`
  directly inside it (not recursive). Missing `--input` path -> error naming the path, exit, write
  nothing (Req 1.4). Output folder is created if absent; no local `data/` dir is assumed (Req 1.3).
  A file that cannot be opened/decoded is reported by name and skipped; folder mode continues (Req 1.5).
- **Per-camera config (Req 2):** parse `camID` from the filename, load `<config-dir>/<camID>.yaml`,
  read `model.weights`, `model.conf`, `model.classes`. Detections are restricted to those class IDs;
  the `camera` column is always the parsed `camID`, so cameras are never combined. Unparseable name,
  missing config, or missing field -> error naming the cause, skip that video, continue the batch
  (Req 2.3-2.5).
- **Tracking (Req 3):** `model.track(source=frame, persist=True, classes=cfg_classes, conf=cfg_conf,
  verbose=False)` using ultralytics YOLO + the default ByteTrack tracker, with the weights from
  config. Detections below the config confidence are discarded (ultralytics `conf` plus a defensive
  filter when building rows). `persist=True` keeps track IDs stable across frames.
- **Frame selection (Req 3.4, 13):** iterate frames by true index `i = 0,1,2,...`. With `--stride N`
  process frames where `i % N == 0`; frames in between are read and skipped (or `grab()`-ped) so the
  true index is preserved. With `--max-seconds N` stop once `i / fps >= N`. When both are set, process
  every Nth frame whose `i / fps < N`. Every recorded row carries the **true** frame index `i`, and
  `time` is computed from that true index — not from a compacted counter (Req 13.2). Invalid stride
  (not an int >= 1) or non-positive max-seconds -> error naming the value, write nothing (Req 3.5, 13.4).
- **CSV assembly (Req 4):** one row per tracked detection at or above the config threshold, columns in
  the fixed order. `uid = f"{video_name}_{camera_id}_{track_id}"`; `track_id` is the integer from YOLO;
  `cls` is the COCO class **name** mapped from the model's `names`; `conf` rounded to 3 decimals;
  `cx, cy, w, h` are box center-x, center-y, width, height in pixels rounded to 1 decimal.
- **Skip/overwrite (Req 5):** if the target CSV exists and `--overwrite` is off, skip and report the
  named video; otherwise process and replace. Folder batches continue after a skip.
- **Annotated MP4 (Req 6):** only with `--save-annotated`. Draw box + track ID for every row written.
  Blur the **top 25%** of each person box (COCO class 0) for faces; blur the **bottom 30%** of each
  vehicle box (car, motorcycle, bus, truck) for plates, using `cv2.GaussianBlur` on the sub-region.
  Written with `cv2.VideoWriter` (mp4v). If the MP4 cannot be written, report the target path and still
  write the CSV (Req 6.5). Without the flag, only the CSV is written (Req 6.4).
- **Progress & summaries (Req 7):** a progress line to stdout at least every 5 seconds (wall clock via
  `time.monotonic()`) reporting frames processed / total frames; a per-video summary (frames processed,
  total detections, unique track count per class); a final folder summary (videos processed, videos
  skipped).
- **Headless (Req 9.1):** no `cv2.imshow`, `cv2.namedWindow`, `cv2.waitKey`, or any GUI call anywhere
  in the detector path.

### tools/export_frame.py (Requirement 11)

```
--video PATH     # input video (required)
--output PATH    # output PNG file, or a folder (required)
```

```python
def export_frame(video_path: Path, output: Path) -> Path: ...  # returns the written PNG path
def main(argv: list[str]) -> int: ...
```

Reads the first frame with `cv2.VideoCapture.read()`. If `--output` is a folder (exists as a dir or
ends with a separator), write `<Video_Name>.png` into it (Req 11.2); otherwise treat it as the PNG
path. Create the output folder if absent (Req 11.3). Missing/unreadable video or unreadable first
frame -> error naming the path, write nothing (Req 11.4). Headless-safe (no GUI).

### tools/pick_points.py (Requirements 8, 9.2, 9.3)

```
--video PATH     # pick on the video's first frame
--image PATH     # pick on a PNG/JPG
# exactly one of --video / --image is required (Req 8.1)
```

```python
def ensure_display() -> None: ...       # raise/exit if no GUI display (Req 9.3)
def load_base_image(args) -> np.ndarray: ...  # first video frame OR image; errors name the path (Req 8.2)
def collect_points(image: np.ndarray) -> tuple[list[LinePt], list[PolyPt]]: ...  # cv2 window + mouse callback (Req 8.3, 9.2)
def to_yaml(line_pts, poly_pts) -> str: ...  # matches Camera_Config counting_line.points / lane_polygon (Req 8.4)
def save_annotated_frame(image, line_pts, poly_pts, out_dir: Path = Path("data/frames")) -> Path: ...  # (Req 8.4)
def main(argv: list[str]) -> int: ...
```

Behavior:

- **Display guard first (Req 9.3):** before opening a window, check for a display. On macOS/Windows a
  display is assumed present; on Linux require a non-empty `DISPLAY`/`WAYLAND_DISPLAY`. If unavailable,
  print an error that an interactive display is required and stop. This is the counterpart to the
  detector being fully headless (Req 9.1).
- **Exactly one source (Req 8.1):** neither or both of `--video`/`--image` -> error, stop, save nothing.
- **Load base image (Req 8.2):** `--video` opens the first frame; `--image` loads the file. Missing,
  unreadable, or undecodable -> error naming the path, stop, save nothing.
- **Click collection (Req 8.3, 9.2):** open a `cv2.namedWindow`, register a mouse callback that records
  left-clicks. Collect exactly **2** counting-line points first, then **4 to 20** lane-polygon points.
  Keys: a point confirms on click, `u` undoes the last point, `Enter` finishes the current stage, `Esc`
  cancels. If collection ends before 2 line points **and** at least 4 polygon points, report incomplete
  and do **not** print YAML or save a frame.
- **Output (Req 8.4):** print YAML whose shape matches `camA.yaml`:
  `counting_line: {points: [[x,y],[x,y]]}` and `lane_polygon: [[x,y], ...]`. Also save the clicked
  image with the line and polygon drawn on it to `data/frames/` (folder created if absent).
- **Camera inference (Req 8.5):** try to parse `camID` from the source filename (reusing
  `parse_video_name`). If parsed, reference `config/<camID>.yaml` and warn (naming the path) if it does
  not exist. If not parseable, continue and warn that the config cannot be inferred.

### notebooks/run_detection.ipynb (Requirement 12)

Ordered cells, Drive-based, cameras processed separately:

1. Mount Google Drive (`google.colab.drive.mount('/content/drive')`).
2. Clone or copy the repo into the Colab runtime.
3. `!pip install -r requirements.txt`.
4. Export frames with `tools/export_frame.py` for sample clips (before detection) (Req 12.3).
5. Run `src/detect.py` for **camA**: `--input <Drive camA folder> --output <Drive out> --config-dir config`.
6. Run `src/detect.py` for **camB** in a separate cell with camB Drive paths (Req 12.4).
7. CSVs land on Drive; camA and camB runs are distinct cells so detections/counts are never combined
   (Req 12.5).

### Setup files (Requirement 10)

- `requirements.txt`: exactly ultralytics, opencv-python, pandas, numpy, matplotlib, plotly, requests,
  pyyaml, pytest — each pinned to a single exact version (`==`), no open ranges (Req 10.1, 10.2, 10.5).
- `.gitignore`: add `data/raw_video/`, `demo/media/`, `*.mp4` (Req 10.3).
- README "How to run detection" section, in order: mount Drive in Colab, `pip install -r
  requirements.txt`, run the detector command, and reference `notebooks/run_detection.ipynb` (Req 10.4).

## Data Models

**Camera_Config** (`config/<camID>.yaml`) — existing format; the detector reads `model.weights`
(str), `model.conf` (float), `model.classes` (list[int], COCO IDs). Other sections (`counting_line`,
`lane_polygon`, `blockage`, etc.) are not used by the detector but must remain valid YAML.

**Detections_CSV** — fixed 12-column contract, in order (never changed):

| column | type | notes |
|---|---|---|
| video | str | Video_Name (no dir, no extension) |
| camera | str | parsed Camera_ID; always matches the filename camID |
| time | str | ISO 8601 with UTC offset, America/New_York, ms precision |
| frame | int | **true** source frame index |
| track_id | int | integer from YOLO tracking |
| uid | str | `video_camera_track_id` |
| cls | str | COCO class name |
| conf | float | 0.0–1.0, 3 decimals |
| cx, cy | float | box center x/y in px, 1 decimal |
| w, h | float | box width/height in px, 1 decimal |

**Per-video summary** (dict, printed and returned): `frames_processed`, `total_detections`,
`unique_tracks_per_class`, `skipped` (bool). **Folder summary**: `videos_processed`, `videos_skipped`.

**Timestamp math (Req 4.2, 13.2):** `time = start + timedelta(seconds = frame_index / fps)` formatted
with `datetime.isoformat(timespec="milliseconds")`, `start` built with
`zoneinfo.ZoneInfo("America/New_York")` so the result is timezone-aware.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a
system — a formal statement about what the system should do. Properties are the bridge between
human-readable specifications and machine-verifiable correctness guarantees.*

The properties below cover the pure, input-varying logic of the pipeline (path building, filename
parsing, timestamp math, frame selection, CSV row formatting, blur geometry, and picker output).
Infrastructure-heavy behavior (real YOLO tracking, MP4 rendering, GUI clicks, progress timing, file
content of setup files) is covered by example and integration tests in the Testing Strategy instead.

### Property 1: CSV column contract is fixed

*For any* produced Detections_CSV, the header row equals exactly
`["video","camera","time","frame","track_id","uid","cls","conf","cx","cy","w","h"]` in that order.

**Validates: Requirements 4.1**

### Property 2: UID is deterministic and unique per track

*For any* Video_Name, Camera_ID, and track_id, the uid equals `f"{video}_{camera}_{track_id}"`, and
two detections produce the same uid *if and only if* they share the same (video, camera, track_id).

**Validates: Requirements 4.3**

### Property 3: Timestamp equals start + frame/fps and is timezone-aware

*For any* start time, non-negative true frame index, and positive fps, `frame_time` equals
`start + frame/fps` seconds, is timezone-aware in America/New_York, and is formatted to millisecond
precision.

**Validates: Requirements 4.2, 13.2**

### Property 4: Frame selection picks every Nth true frame within the time bound

*For any* stride M ≥ 1, optional max-seconds N > 0, fps > 0, and frame count, the set of selected
frame indices equals `{ i : i % M == 0 and (N is None or i/fps < N) }`, and each selected index is
recorded as its own true source index.

**Validates: Requirements 3.4, 13.1, 13.3, 13.5**

### Property 5: Confidence threshold filter

*For any* set of detections and a config confidence threshold, every row written to the CSV has
`conf >= threshold`, and no at-or-above-threshold detection is dropped.

**Validates: Requirements 3.3, 4.1**

### Property 6: CSV numeric fields are rounded and in range

*For any* detection, `cx, cy, w, h` equal their source values rounded to 1 decimal, `conf` is in
`[0.0, 1.0]` rounded to 3 decimals, and `cls` is a class name drawn from the Camera_Config class list.

**Validates: Requirements 4.4, 4.5**

### Property 7: The camera column always matches the parsed camID (cameras never combined)

*For any* video whose filename parses to Camera_ID X, every row in its Detections_CSV has
`camera == X`, and no row carries a different camera value.

**Validates: Requirements 2.1, 2.2**

### Property 8: Filename parser accepts exactly the valid pattern

*For any* string, `parse_video_name` succeeds *if and only if* the name matches
`commave_<camID>_YYYY-MM-DD_HHMM`, returning that camID and a tz-aware America/New_York start time;
non-matching names raise an error.

**Validates: Requirements 2.1, 2.3**

### Property 9: Skip/overwrite decision

*For any* booleans (csv_exists, overwrite), the detector processes the video *if and only if*
`overwrite or not csv_exists`; otherwise it skips and leaves the existing CSV unchanged.

**Validates: Requirements 5.1, 5.2**

### Property 10: Blur region geometry

*For any* bounding box, the person face region is the top 25% of the box height anchored at the box
top, and the vehicle plate region is the bottom 30% of the box height anchored at the box bottom;
each region stays within the box bounds.

**Validates: Requirements 6.2, 6.3**

### Property 11: Point picker output gate and YAML round-trip

*For any* collected counting-line points (count L) and lane-polygon points (count P), the picker
produces output *if and only if* `L == 2 and 4 <= P <= 20`; and when it does, parsing the printed
YAML yields `counting_line.points` equal to the line points and `lane_polygon` equal to the polygon
points.

**Validates: Requirements 8.3, 8.4**

### Property 12: Frame export path rule

*For any* Video_Name and output that names a folder, the exported PNG path equals
`<folder>/<Video_Name>.png`.

**Validates: Requirements 11.2**

## Error Handling

The detector follows a **report-and-continue** policy in folder/batch mode and a
**report-and-stop** policy for whole-run configuration errors:

- **Stop the whole run, write nothing:** missing `--input` path (Req 1.4), invalid `--stride` (Req
  3.5), invalid `--max-seconds` (Req 13.4). These are user/CLI errors caught before any processing.
- **Skip one video, continue the batch:** undecodable video (Req 1.5), unparseable filename (Req
  2.3), missing Camera_Config (Req 2.4), missing config field (Req 2.5). Each prints an error that
  names the offending file/path/field and the batch proceeds to the next `.mp4`.
- **Degrade, still deliver the CSV:** if `--save-annotated` is set but the MP4 cannot be written,
  the error names the target MP4 path and the Detections_CSV is still written (Req 6.5).
- **export_frame:** missing/unreadable video or unreadable first frame -> error naming the path,
  nothing written (Req 11.4).
- **pick_points:** not exactly one source (Req 8.1), unreadable source (Req 8.2), no display (Req
  9.3), or incomplete collection (Req 8.3) -> error/warning naming the cause, nothing saved.

Errors are reported to stderr with a clear message; the CLIs return a non-zero exit code on a
run-stopping error and zero when a batch completes even if some videos were skipped.

## Testing Strategy

Tests are a small, focused set of fast `pytest` unit tests over the pure helper functions. The
Correctness Properties above remain the formal spec of intended behavior; here each is checked with
a few example-based `pytest` cases rather than exhaustive generation. There is NO property-based
testing, NO `hypothesis`, and NO new dependency — a handful of hand-written example cases per
behavior is enough. Table-driven `pytest.mark.parametrize` is fine as long as each case list stays
short (a handful of cases, not dozens).

Tests MUST NOT require a real video file or the YOLO model. They exercise the pure helpers directly:
filename parsing, timestamp math, CSV row/column assembly from synthetic detection records, config
selection/loading from small temporary YAML files, path-building, blur-region geometry, and picker
YAML formatting. Any behavior that needs real decode or model inference (YOLO tracking/track-ID
persistence, annotated MP4 rendering, headless window behavior, progress timing) is NOT unit-tested
here; it is verified manually on a short clip during the Colab run.

**Core set (highest priority):**

1. **Filename parsing (Req 2.1, 2.3):** `parse_video_name` extracts the correct camera ID and a
   timezone-aware America/New_York start time from `commave_<camID>_YYYY-MM-DD_HHMM.mp4`, and rejects
   malformed names with an error.
2. **Timestamp calculation (Req 4.2, 13.2):** `frame_time(start, frame_index, fps)` equals
   `start + frame/fps` seconds, is timezone-aware in America/New_York, and is formatted to
   millisecond precision.
3. **CSV column contract (Req 4.1):** the written header equals exactly the 12 columns
   `["video","camera","time","frame","track_id","uid","cls","conf","cx","cy","w","h"]` in order, and
   a row built from a synthetic detection record places each value in the correct column.
4. **Per-camera config selection (Req 2.2, 2.4, 2.5):** given a camA filename the detector resolves
   `<config-dir>/camA.yaml`, given camB it resolves `camB.yaml`, and a missing config or missing
   field (`weights`/`conf`/`classes`) is reported clearly (loaded from small temp YAML files).

**Secondary set (a few extra cheap unit tests, no video/model):**

- Skip/overwrite decision: process iff `overwrite or not csv_exists` (Req 5.1, 5.2).
- Invalid `--stride` (not int ≥ 1) and invalid `--max-seconds` (non-positive) are rejected with an
  error and nothing is written (Req 3.5, 13.4).
- Frame-selection index math for stride + max-seconds: selected indices equal
  `{ i : i % M == 0 and (N is None or i/fps < N) }` (Req 3.4, 13.1, 13.3).
- `requirements.txt` lists the expected packages, each pinned with `==` (Req 10.1, 10.2, 10.5).
- `.gitignore` contains the three required entries (Req 10.3).
