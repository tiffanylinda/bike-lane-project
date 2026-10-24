# Project structure

bike-lane-project/
  config/camera.yaml          counting line, lane polygon, thresholds, model, location
  data/raw_video/             input video chunks (gitignored)
  data/frames/                frames used to pick the line and polygon
  data/detections/            one *_detections.csv per video chunk
  data/telraam/               downloaded Telraam counts
  data/public/                crash records, counter data
  src/detect.py               video -> detections CSV (+ optional annotated, blurred MP4)
  src/lane_monitor.py         LaneMonitor class: frame-by-frame lane status + events
  src/count.py                detections -> line crossings, cyclist vs pedestrian, hourly counts
  src/blockages.py            runs LaneMonitor over CSVs -> blockages.csv + events
  src/telraam.py              download Telraam hourly counts
  src/validate.py             compare our counts with Telraam and manual counts
  src/dashboard.py            Plotly charts and the evidence dashboard HTML
  src/build_demo.py           builds the replay demo: clip, overlay video, events.json
  tools/pick_points.py        click on a frame to get line/polygon coordinates
  demo/index.html             replay demo page (video + phone-style rider app)
  demo/media/                 demo clip (gitignored)
  demo/data/events.json       events the demo page plays back
  tests/                      pytest tests using small synthetic CSVs
  outputs/                    charts, dashboard, CSV summaries
  app/                        "Comm Ave Lane Watch" demo app (separate from demo/)
    backend/                  FastAPI + uvicorn API (lane status, blockages, stats, reports)
    backend/data/             sample blockages.csv + telraam.csv the API reads
    backend/tests/            pytest tests calling the pure functions directly
    frontend/                 React + Vite + Recharts front end (built in a later task)
    frontend/public/data/     static fallback JSON so the front end runs without the backend
    README.md                 how to run the app/ backend and front end

Specs: keep requirements.md, design.md and tasks.md short. No more than 5 acceptance
criteria per requirement. Skip diagrams and long test plans.
