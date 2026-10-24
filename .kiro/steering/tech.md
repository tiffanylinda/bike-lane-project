# Tech stack and conventions

## Libraries
- Python 3.10+: ultralytics (YOLO + ByteTrack), opencv-python, pandas, numpy, matplotlib,
  plotly, requests, pyyaml, pytest. Ask before adding any other Python package.
- Demo front end: one plain HTML file with vanilla JavaScript and CSS. No React, no build
  step, no CDN links (it must work offline). Serve it with `python -m http.server`.

## Computer vision
- Pretrained COCO YOLO, default yolov8n.pt (configurable).
  COCO class IDs: person 0, bicycle 1, car 2, motorcycle 3, bus 5, truck 7, skateboard 36.
- Video chunks are named commave_YYYY-MM-DD_HHMM.mp4. Start time comes from the filename.
  Every detection's timestamp = start time + frame_index / fps, in America/New_York.

## Rules
- Never hard-code pixel coordinates, thresholds or paths. They live in config/camera.yaml.
- Raw video never goes in git. data/raw_video/ and demo/media/ are in .gitignore.
- Heavy CV runs once per video and writes a CSV. Analysis code reads CSVs, never video.
- Blockage logic lives in ONE class, LaneMonitor (src/lane_monitor.py), that processes
  detections one frame at a time and emits events. Batch analysis, the replay demo and
  any live mode all use this same class, so their results always agree.
- Every script has an argparse CLI and a docstring stating inputs and outputs. Comments
  explain *why*, for beginners.
- Faces and license plates must be blurred in every exported image or video.

## Shared data contracts (do not change without telling the whole team)
- detections CSV: video, time, frame, track_id, uid, cls, conf, cx, cy, w, h
- events.json: see "Spec 4" in KIRO_PROMPTS.md
