# Tech stack and conventions

## Libraries
- Python 3.10+: ultralytics (YOLO + ByteTrack), opencv-python, pandas, numpy, matplotlib,
  plotly, requests, pyyaml, pytest. fastapi and uvicorn are also approved, for the app/
  backend only. Ask before adding any other Python package.
- demo/ replay demo front end: one plain HTML file with vanilla JavaScript and CSS. No
  React, no build step, no CDN links (it must work offline). Serve it with
  `python -m http.server`. This rule applies ONLY to the existing demo/ folder.
- app/ "Comm Ave Lane Watch" front end: React + Vite + Recharts with plain CSS (no other
  UI library). Backend: FastAPI + uvicorn on Python 3. The app/ front end MUST run without
  the backend: if the API is unreachable it loads the same data from static JSON files in
  app/frontend/public/data/, so the front end can be hosted on GitHub Pages as a public
  testing link for judges.

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
