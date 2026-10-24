# Requirements Document

## Introduction

The video detection pipeline turns recorded bike lane video chunks into per-video CSV files of
tracked objects, so the rest of the team works in pandas without touching video. Raw videos live on
Google Drive and detection runs in Google Colab, never locally in this repo; `src/detect.py` reads
input videos and writes outputs through CLI paths (typically Drive paths mounted at
`/content/drive/MyDrive/...`) and never assumes a local `data/` directory exists. Two cameras film
the same street at the same time from different angles; each camera has its own config file, and
counts are never combined across cameras. The pipeline has these scripts: `src/detect.py` (headless,
Colab batch detection and tracking), `tools/export_frame.py` (saves a video's first frame as a PNG so
it can be downloaded from Drive), and `tools/pick_points.py` (an interactive local tool to pick the
counting line and lane polygon from a video or a downloaded PNG). A ready-to-run Colab notebook at
`notebooks/run_detection.ipynb` ties the Drive-based detection run together. All pixel coordinates,
thresholds, model weights and config/output paths come from config files and CLI arguments, never
from hard-coded local paths.

## Glossary

- **Detector**: The `src/detect.py` script that reads video and writes a detections CSV.
- **Frame_Exporter**: The `tools/export_frame.py` script that saves a video's first frame as a PNG.
- **Point_Picker**: The `tools/pick_points.py` interactive tool that captures counting line and lane polygon points from a video or an image.
- **Colab_Notebook**: The ready-to-run notebook `notebooks/run_detection.ipynb` that performs a full Drive-based detection run in Google Colab.
- **Output_Folder**: The folder passed to the Detector via `--output` where it writes its outputs.
- **Config_Dir**: The directory passed via `--config-dir` where per-camera config files live (default `config/`).
- **Camera_Config**: The per-camera YAML file `<Config_Dir>/<camID>.yaml` (for example `config/camA.yaml`).
- **Camera_ID**: The camera identifier parsed from the video filename (for example `camA`, `camB`).
- **Detections_CSV**: The output file `<Output_Folder>/<Video_Name>_detections.csv`.
- **Video_Name**: The video filename without its directory path or extension.
- **UID**: A globally unique track identifier, formed as Video_Name + Camera_ID + track_id.
- **Start_Time**: The recording start time parsed from the filename pattern `commave_<camID>_YYYY-MM-DD_HHMM.mp4`, interpreted in the America/New_York timezone.
- **Annotated_MP4**: An optional output video `<Output_Folder>/<Video_Name>_annotated.mp4` with detection boxes, track IDs, and blurred faces and license plates.

## Requirements

### Requirement 1: Detect from a file or folder

**User Story:** As a team member, I want to run detection on one video or a whole folder with explicit input and output paths, so that I can produce detections CSVs for every clip from Google Drive in one command.

#### Acceptance Criteria

1. WHEN the Detector is invoked with an `--input` path to a single video file and an `--output` folder path, THE Detector SHALL process that video and write exactly one Detections_CSV for it to the Output_Folder as `<Output_Folder>/<Video_Name>_detections.csv`.
2. WHEN the Detector is invoked with an `--input` path to a folder and an `--output` folder path, THE Detector SHALL process every file with a `.mp4` extension directly in the input folder and write exactly one Detections_CSV per such file to the Output_Folder.
3. WHEN the Detector writes a Detections_CSV, THE Detector SHALL first create the Output_Folder if it does not exist, and SHALL NOT require any local `data/` directory to exist.
4. IF the `--input` path does not exist, THEN THE Detector SHALL report an error that names the missing path and stop without writing any Detections_CSV.
5. IF a video file cannot be opened or decoded, THEN THE Detector SHALL report an error that names that file, skip it without writing its Detections_CSV, and continue processing any remaining `.mp4` files in folder mode.

### Requirement 2: Per-camera configuration

**User Story:** As a team member, I want the pipeline to pick each camera's settings automatically, so that I never combine cameras or hard-code values.

#### Acceptance Criteria

1. WHEN the Detector processes a video, THE Detector SHALL parse the Camera_ID from the video filename using the pattern `commave_<camID>_YYYY-MM-DD_HHMM.mp4` and load the Camera_Config located at `<Config_Dir>/<camID>.yaml`, WHERE Config_Dir comes from the `--config-dir` option and defaults to `config/`.
2. WHEN a Camera_Config is loaded, THE Detector SHALL read the model weights, confidence threshold, and detection classes from that Camera_Config and restrict detections to only the classes listed in it, without combining detections or counts across different Camera_IDs.
3. IF the Camera_ID cannot be parsed from the filename, THEN THE Detector SHALL report an error that names the filename, skip that video, and continue processing the remaining videos in the batch.
4. IF the Camera_Config file for a parsed Camera_ID does not exist at `<Config_Dir>/<camID>.yaml`, THEN THE Detector SHALL report an error that names the expected config path, skip that video, and continue processing the remaining videos in the batch.
5. IF a loaded Camera_Config is missing any of the model weights, confidence threshold, or detection classes, THEN THE Detector SHALL report an error that names the missing field and the config path, skip that video, and continue processing the remaining videos in the batch.

### Requirement 3: Object tracking with YOLO

**User Story:** As a team member, I want objects tracked across frames, so that each object keeps a stable identity in the CSV.

#### Acceptance Criteria

1. WHEN the Detector processes a video, THE Detector SHALL run ultralytics YOLO tracking with `persist=True` so that track identifiers persist across frames.
2. THE Detector SHALL use the model weights named in the Camera_Config.
3. THE Detector SHALL discard detections whose confidence is below the threshold named in the Camera_Config.
4. WHERE the `--stride N` option is set to an integer N of 1 or greater, THE Detector SHALL process every Nth frame, skip the frames in between, and record each processed detection with the true frame index it holds in the source video; WHERE the `--stride N` option is not set, THE Detector SHALL process every frame (equivalent to N of 1).
5. IF the `--stride N` option is set to a value that is not an integer of 1 or greater, THEN THE Detector SHALL report an error that names the invalid value and stop without writing output.

### Requirement 4: Detections CSV contract

**User Story:** As a team member, I want a CSV with a fixed set of columns, so that downstream pandas code reads every camera the same way.

#### Acceptance Criteria

1. THE Detector SHALL write the Detections_CSV with a header row followed by exactly these columns in this order: video, camera, time, frame, track_id, uid, cls, conf, cx, cy, w, h, and SHALL write one row for each tracked detection whose confidence is at or above the Camera_Config threshold.
2. THE Detector SHALL set the `time` column to an ISO 8601 timestamp with a UTC offset in the America/New_York timezone, computed as Start_Time plus (frame index divided by the video's frames per second) and rounded to millisecond precision.
3. THE Detector SHALL set the `uid` column to the Video_Name, Camera_ID, and track_id joined in that order with a single underscore separator, and SHALL set the `track_id` column to the integer identifier from YOLO tracking.
4. THE Detector SHALL record `cx`, `cy`, `w`, and `h` as the box center x, center y, width, and height in pixels, each as a floating-point value rounded to one decimal place.
5. THE Detector SHALL set the `cls` column to the detected class name string from the Camera_Config class list and the `conf` column to the detection confidence as a floating-point value between 0.0 and 1.0 rounded to three decimal places.

### Requirement 5: Overwrite and skip behavior

**User Story:** As a team member, I want to re-run detection without redoing finished videos, so that I save time on large batches.

#### Acceptance Criteria

1. IF a Detections_CSV already exists in the Output_Folder for a video AND the `--overwrite` option is not set, THEN THE Detector SHALL skip that video, leave the existing Detections_CSV unchanged, and report that the named video was skipped.
2. WHERE the `--overwrite` option is set, THE Detector SHALL process the video and replace any existing Detections_CSV for that video.
3. WHEN the Detector skips a video during a folder batch, THE Detector SHALL continue processing the remaining videos in that folder.

### Requirement 6: Annotated video with privacy blurring

**User Story:** As a team member, I want an optional annotated video with faces and plates blurred, so that I can share a demo clip without exposing identities.

#### Acceptance Criteria

1. WHERE the `--save-annotated` option is set, THE Detector SHALL write an Annotated_MP4 to `<Output_Folder>/<Video_Name>_annotated.mp4` that draws a detection box and its track ID for every tracked detection written to the Detections_CSV.
2. WHERE the `--save-annotated` option is set, THE Detector SHALL blur the top 25 percent of each person box (COCO class 0) in the Annotated_MP4.
3. WHERE the `--save-annotated` option is set, THE Detector SHALL blur the bottom 30 percent of each vehicle box (classes car, motorcycle, bus, and truck) in the Annotated_MP4.
4. WHERE the `--save-annotated` option is not set, THE Detector SHALL write only the Detections_CSV and SHALL NOT write an Annotated_MP4.
5. IF the Annotated_MP4 cannot be written while the `--save-annotated` option is set, THEN THE Detector SHALL report an error that names the target Annotated_MP4 path and SHALL still write the Detections_CSV.

### Requirement 7: Progress and summary output

**User Story:** As a team member, I want to see progress and a summary, so that I can confirm a run finished and inspect the counts.

#### Acceptance Criteria

1. WHILE the Detector processes a video, THE Detector SHALL print to stdout, at least once every 5 seconds, a progress line that reports the number of frames processed so far and the total number of frames in the video.
2. WHEN the Detector finishes a video, THE Detector SHALL print to stdout a summary line that reports frames processed, total detections, and unique track count per class for that video.
3. WHEN the Detector is invoked on a folder and finishes processing all videos in that folder, THE Detector SHALL print to stdout a final line that reports the number of videos processed and the number of videos skipped.

### Requirement 8: Pick counting line and lane polygon

**User Story:** As a team member, I want to click points on a video frame or a downloaded PNG to get coordinates, so that I can fill in the camera config on a laptop without guessing pixels.

#### Acceptance Criteria

1. THE Point_Picker SHALL require exactly one of `--video` or `--image`; IF neither or both are provided, THEN THE Point_Picker SHALL report an error and stop without saving output.
2. WHEN the Point_Picker is invoked with a `--video` path, THE Point_Picker SHALL open the first frame of the video for the user to click on; WHEN the Point_Picker is invoked with an `--image` path, THE Point_Picker SHALL load that image for the user to click on; IF the given path is missing, unreadable, or its frame or image cannot be read, THEN THE Point_Picker SHALL report an error that names the path and stop without saving output.
3. THE Point_Picker SHALL capture exactly 2 points for the counting line and between 4 and 20 points inclusive for the lane polygon, and IF point collection ends before 2 counting-line points and at least 4 lane-polygon points are captured, THEN THE Point_Picker SHALL report that collection was incomplete and SHALL NOT print YAML or save a frame.
4. WHEN point collection is complete, THE Point_Picker SHALL print the counting line and lane polygon as YAML matching the Camera_Config `counting_line.points` and `lane_polygon` structure, and SHALL save the clicked-on frame or image with the counting line and lane polygon drawn on it to `data/frames/`.
5. WHERE a Camera_ID can be parsed from the `--video` or `--image` filename, THE Point_Picker SHALL reference the matching Camera_Config and warn naming the expected config path if none exists; WHERE a Camera_ID cannot be parsed from the filename, THE Point_Picker SHALL continue and warn that the Camera_Config cannot be inferred.

### Requirement 9: Headless and interactive execution

**User Story:** As a team member, I want detection to run on Google Colab and point picking to run locally, so that heavy processing uses free compute and setup uses a GUI.

#### Acceptance Criteria

1. THE Detector SHALL complete a full detection run without invoking any GUI display call and without opening any GUI window at any processing stage.
2. WHEN the Point_Picker is launched on a local machine with an available display, THE Point_Picker SHALL open an interactive GUI window showing the video frame and SHALL record each point the user clicks in the window.
3. IF the Point_Picker is launched in an environment where no GUI display is available, THEN THE Point_Picker SHALL stop before opening a window and return an error indicating that an interactive display is required.

### Requirement 10: Project setup and documentation

**User Story:** As a team member, I want pinned dependencies, git ignore rules, and run instructions, so that I can set up and run the pipeline reliably.

#### Acceptance Criteria

1. THE video-detection-pipeline SHALL provide a `requirements.txt` that lists exactly these packages: ultralytics, opencv-python, pandas, numpy, matplotlib, plotly, requests, pyyaml, and pytest.
2. THE video-detection-pipeline SHALL pin each package in `requirements.txt` to an exact version (a single resolvable version per package, not an open or minimum-bound range).
3. THE video-detection-pipeline SHALL list `data/raw_video/`, `demo/media/`, and `*.mp4` in `.gitignore` so that raw video files are excluded from git tracking.
4. THE video-detection-pipeline SHALL provide a README section titled "How to run detection" that contains, in order, the steps to mount Google Drive in Colab, install dependencies via `pip install -r requirements.txt`, and run the Detector command, and that references the Colab_Notebook at `notebooks/run_detection.ipynb`.
5. IF `requirements.txt` is missing any of the required packages or omits an exact version for any listed package, THEN the setup SHALL be considered non-conforming to this requirement.

### Requirement 11: Export a frame as PNG

**User Story:** As a team member, I want to export a video's first frame as a PNG in Colab, so that I can download it and pick points on a laptop without the full video.

#### Acceptance Criteria

1. WHEN the Frame_Exporter is invoked with a `--video` path and an `--output` path, THE Frame_Exporter SHALL read the first frame of the video and write it as a PNG to the given output location.
2. WHERE the `--output` path names a folder, THE Frame_Exporter SHALL write the PNG into that folder as `<Video_Name>.png`.
3. WHEN the Frame_Exporter writes the PNG, THE Frame_Exporter SHALL first create the output folder if it does not exist.
4. IF the `--video` path is missing or unreadable, or its first frame cannot be read, THEN THE Frame_Exporter SHALL report an error that names the path and stop without writing a PNG.

### Requirement 12: Colab detection notebook

**User Story:** As a team member, I want a ready-to-run Colab notebook, so that I can run the full Drive-based detection for both cameras without assembling commands by hand.

#### Acceptance Criteria

1. THE video-detection-pipeline SHALL provide the Colab_Notebook at `notebooks/run_detection.ipynb`.
2. THE Colab_Notebook SHALL, in order, mount Google Drive, clone or copy the repository, and install dependencies via `pip install -r requirements.txt`.
3. THE Colab_Notebook SHALL export frames using `tools/export_frame.py` before running detection.
4. THE Colab_Notebook SHALL run `src/detect.py` for camera camA and for camera camB, each with Drive `--input` and `--output` paths and a `--config-dir` option, writing the Detections_CSV files to Google Drive.
5. THE Colab_Notebook SHALL process camera camA and camera camB separately so that detections and counts are never combined across Camera_IDs.

### Requirement 13: Limit processing duration

**User Story:** As a team member, I want to limit detection to the first N seconds of each video, so that I can test on a short prefix before running the full file.

#### Acceptance Criteria

1. WHERE the `--max-seconds N` option is set to a positive number N, THE Detector SHALL process only frames whose timestamp offset from the start of the video is less than N seconds, then stop and finalize the Detections_CSV for that video.
2. WHEN the Detector limits processing with the `--max-seconds N` option, THE Detector SHALL record each processed detection with the true frame index it holds in the source video and set the `time` column to the timestamp computed from that true frame index, consistent with the Requirement 3 stride behavior and the Requirement 4 timestamp math.
3. WHERE the `--max-seconds N` option is not set, THE Detector SHALL process the entire video.
4. IF the `--max-seconds N` option is set to a value that is not a positive number, THEN THE Detector SHALL report an error that names the invalid value and stop without writing output.
5. WHERE both the `--max-seconds N` option and the `--stride M` option are set, THE Detector SHALL process every Mth frame whose timestamp offset from the start of the video is less than N seconds.
