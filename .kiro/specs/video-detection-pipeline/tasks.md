# Implementation Plan: Video Detection Pipeline

## Overview

Build the pipeline bottom-up: scaffolding and pinned dependencies first, then the pure,
testable helpers in `src/detect.py` with their fast `pytest` unit tests right after, then the
video/model-dependent processing code, then the `export_frame` and `pick_points` tools, and
finally the Colab notebook and README. Each step builds on the previous one and ends wired into
the detector's `main()` or a runnable script. Per the Testing Strategy, tests are a small set of
example-based `pytest` cases over the pure helpers only — no property-based testing, no
`hypothesis`, and no real video or YOLO model in the test suite. Anything needing real decode or
model inference is verified manually on a short clip during the Colab run.

## Tasks

- [x] 1. Scaffold project structure and pinned setup files
  - Create `src/` and `tools/` packages (each with an `__init__.py`) and a `tests/` folder.
  - Create `requirements.txt` listing exactly ultralytics, opencv-python, pandas, numpy,
    matplotlib, plotly, requests, pyyaml, pytest, each pinned with `==` to a single exact version
    (no open or minimum-bound ranges).
  - Add `data/raw_video/`, `demo/media/`, and `*.mp4` entries to `.gitignore`.
  - _Requirements: 10.1, 10.2, 10.3, 10.5_

- [x] 2. Implement pure helper functions in `src/detect.py`
  - [x] 2.1 Add module docstring, imports, and the CSV column contract
    - Write the `src/detect.py` module docstring stating inputs (video file/folder, per-camera
      config) and outputs (detections CSV, optional annotated MP4), with beginner-facing comments.
    - Define `CSV_COLUMNS = ["video","camera","time","frame","track_id","uid","cls","conf","cx","cy","w","h"]`
      as the single source of truth for the column order.
    - _Requirements: 4.1_

  - [x] 2.2 Implement `parse_video_name` and the filename regex
    - Define `FILENAME_RE` for `commave_<camID>_YYYY-MM-DD_HHMM` and
      `parse_video_name(video_path) -> (camera_id, start_time)` returning a tz-aware
      America/New_York (`zoneinfo.ZoneInfo`) start time; raise `ValueError` on a non-matching name.
    - _Requirements: 2.1, 2.3_

  - [x] 2.3 Implement `load_camera_config`
    - Load `<config_dir>/<camID>.yaml` with `pyyaml`; read `model.weights`, `model.conf`,
      `model.classes`; raise `FileNotFoundError` naming the path when missing and `KeyError`
      naming the missing field.
    - _Requirements: 2.2, 2.4, 2.5_

  - [x] 2.4 Implement `frame_time` and the `uid` builder
    - `frame_time(start, frame_index, fps)` returns `start + frame_index/fps` seconds as an ISO 8601
      string with UTC offset at millisecond precision (`isoformat(timespec="milliseconds")`).
    - Add a small helper that builds `uid = f"{video}_{camera}_{track_id}"`.
    - _Requirements: 4.2, 4.3, 13.2_

  - [x] 2.5 Implement frame-selection index math
    - A pure helper deciding which true frame indices are processed: `{ i : i % stride == 0 and
      (max_seconds is None or i/fps < max_seconds) }`, preserving the true source index `i`.
    - _Requirements: 3.4, 13.1, 13.3, 13.5_

  - [x] 2.6 Implement blur-region geometry helper
    - A pure helper returning the face sub-region (top 25% of a person box) and the plate
      sub-region (bottom 30% of a vehicle box), each clamped within the box bounds.
    - _Requirements: 6.2, 6.3_

  - [x] 2.7 Implement `parse_args` with stride/max-seconds validation
    - argparse CLI with `--input`, `--output`, `--config-dir` (default `config/`), `--stride`
      (int ≥ 1, default 1), `--max-seconds` (positive, optional), `--overwrite`, `--save-annotated`.
    - Reject a `--stride` that is not an int ≥ 1 and a `--max-seconds` that is not positive with an
      error naming the invalid value and no output written.
    - _Requirements: 3.5, 13.4_

- [x] 3. Write core unit tests for the pure helpers
  - [x]* 3.1 Test filename parsing
    - `parse_video_name` extracts the correct Camera_ID and a tz-aware America/New_York start time
      from valid names and rejects malformed names with an error.
    - _Requirements: 2.1, 2.3_

  - [x]* 3.2 Test timestamp calculation
    - `frame_time(start, frame_index, fps)` equals `start + frame/fps` seconds, is tz-aware in
      America/New_York, and is formatted to millisecond precision.
    - _Requirements: 4.2, 13.2_

  - [x]* 3.3 Test CSV column contract and the uid builder
    - The header equals exactly the 12 `CSV_COLUMNS` in order, and `uid` equals
      `f"{video}_{camera}_{track_id}"`.
    - _Requirements: 4.1, 4.3_

  - [x]* 3.4 Test per-camera config selection
    - A camA filename resolves `camA.yaml` and camB resolves `camB.yaml` (small temp YAML files);
      a missing config and a missing `weights`/`conf`/`classes` field are reported clearly.
    - _Requirements: 2.2, 2.4, 2.5_

  - [x]* 3.5 Test secondary helpers (cheap, no video/model)
    - Skip/overwrite decision: process iff `overwrite or not csv_exists`.
    - Invalid `--stride` and invalid `--max-seconds` are rejected with an error and no output.
    - Frame-selection index math equals `{ i : i % M == 0 and (N is None or i/fps < N) }`.
    - `requirements.txt` lists the expected packages each pinned with `==`; `.gitignore` contains
      the three required entries.
    - _Requirements: 3.4, 3.5, 5.1, 5.2, 10.1, 10.2, 10.3, 13.1, 13.3, 13.4_

- [ ] 4. Checkpoint - Ensure all helper tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement video/model processing in `src/detect.py`
  - [ ] 5.1 Implement `process_video` detection and CSV assembly
    - Run ultralytics YOLO `model.track(..., persist=True, classes=cfg_classes, conf=cfg_conf,
      verbose=False)` with the config weights; iterate frames by true index applying the
      stride + max-seconds selection from task 2.5; discard below-threshold detections; build one
      CSV row per tracked detection using `CSV_COLUMNS`, `frame_time`, the uid builder, the COCO
      class name from `model.names`, `conf` to 3 decimals, and `cx,cy,w,h` to 1 decimal; set
      `camera` to the parsed Camera_ID so cameras are never combined; print a progress line at
      least every 5 seconds (`time.monotonic()`) and return a per-video summary dict.
    - _Requirements: 2.2, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5, 7.1, 7.2, 13.1, 13.2, 13.5_

  - [ ] 5.2 Implement skip/overwrite handling in `process_video`
    - When the target CSV exists and `--overwrite` is off, skip and report the named video; else
      process and replace. Reflect the skip in the returned summary.
    - _Requirements: 5.1, 5.2, 5.3_

  - [ ] 5.3 Implement annotated MP4 writing with face/plate blur
    - When `--save-annotated` is set, draw each written detection's box and track ID and blur the
      face/plate sub-regions from task 2.6 with `cv2.GaussianBlur`, writing an `_annotated.mp4` via
      `cv2.VideoWriter`; if the MP4 cannot be written, report the target path and still write the
      CSV; when the flag is off, write only the CSV. No GUI calls anywhere in the detector path.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 9.1_

  - [ ] 5.4 Implement `main` file/folder dispatch and batch loop
    - Resolve `--input` as a single `.mp4` or a folder of `.mp4` files (non-recursive); create the
      Output_Folder if absent without assuming a local `data/`; error-and-stop on a missing
      `--input` path; for each video, report-and-continue on undecodable video, unparseable
      filename, missing config, or missing field; print the final folder summary (videos
      processed, videos skipped); return the process exit code.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.3, 2.4, 2.5, 7.3, 9.1_

  - [ ]* 5.5 Manual verification of detection on a short clip
    - Verify on a short real clip during a Colab run: track IDs persist, the CSV matches the
      contract, `--stride`/`--max-seconds` preserve true frame indices, and the annotated MP4
      renders with faces/plates blurred. Not an automated test (needs a real video and the model).
    - _Requirements: 3.1, 6.1, 6.2, 6.3, 7.1_

- [ ] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement `tools/export_frame.py`
  - [ ] 7.1 Implement the frame exporter
    - Add a docstring and argparse CLI (`--video`, `--output`). Read the first frame via
      `cv2.VideoCapture.read()`; when `--output` names a folder, write `<Video_Name>.png` into it,
      else treat it as the PNG path; create the output folder if absent; error naming the path and
      write nothing on a missing/unreadable video or unreadable first frame. Headless-safe.
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [ ]* 7.2 Test the frame export path rule
    - For an output that names a folder, the exported PNG path equals `<folder>/<Video_Name>.png`
      (pure path logic, no real video).
    - _Requirements: 11.2_

- [ ] 8. Implement `tools/pick_points.py`
  - [ ] 8.1 Implement the interactive point picker
    - Add a docstring and argparse CLI requiring exactly one of `--video`/`--image`. Add a display
      guard that stops with an error when no interactive display is available. Load the base image
      (first video frame or image), erroring by path when unreadable. Collect exactly 2
      counting-line points then 4–20 lane-polygon points via a `cv2` window and mouse callback;
      on incomplete collection report it and save nothing. On completion print YAML matching
      `camA.yaml` (`counting_line.points`, `lane_polygon`) and save the drawn frame to
      `data/frames/` (created if absent). Infer Camera_ID from the filename via `parse_video_name`
      and warn naming the config path when it cannot be found or inferred.
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 9.2, 9.3_

  - [ ]* 8.2 Test the picker output gate and YAML round-trip
    - Output is produced iff `line_count == 2 and 4 <= poly_count <= 20`; parsing the printed YAML
      yields `counting_line.points` equal to the line points and `lane_polygon` equal to the
      polygon points (pure logic, no GUI).
    - _Requirements: 8.3, 8.4_

- [ ] 9. Create `notebooks/run_detection.ipynb`
  - Ordered cells: mount Google Drive, clone/copy the repo into the runtime,
    `!pip install -r requirements.txt`, export frames with `tools/export_frame.py` before
    detection, then run `src/detect.py` for camA in one cell and camB in a separate cell (each with
    Drive `--input`/`--output` and `--config-dir`), writing CSVs to Drive and keeping the cameras
    separate so detections and counts are never combined.
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 10. Add the README "How to run detection" section
  - Add a README section titled "How to run detection" with, in order, the steps to mount Google
    Drive in Colab, install dependencies via `pip install -r requirements.txt`, and run the
    Detector command, referencing `notebooks/run_detection.ipynb`.
  - _Requirements: 10.4_

- [ ] 11. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test or manual-verification sub-tasks and can be skipped for a
  faster MVP; core implementation tasks are never optional.
- Testing is example-based `pytest` over the pure helpers only — no property-based testing, no
  `hypothesis`, and no real video or YOLO model in the suite.
- Tasks 5.5 is manual verification on a short clip during the Colab run because it needs a real
  video and the model; it is not an automated test.
- Each task references the specific requirement clauses it implements for traceability.
- Checkpoints ensure incremental validation at natural breaks.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"] },
    { "id": 2, "tasks": ["3.1", "3.2", "3.3", "3.4", "3.5", "7.1", "8.1"] },
    { "id": 3, "tasks": ["5.1", "7.2", "8.2"] },
    { "id": 4, "tasks": ["5.2", "5.3"] },
    { "id": 5, "tasks": ["5.4"] },
    { "id": 6, "tasks": ["5.5", "9", "10"] }
  ]
}
```
