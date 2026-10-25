# Comm Ave Lane Watch (app/)

A small demo app, separate from the existing `demo/` replay page. It shows bike-lane
blockage status, a blockage list, simple stats, and lets a rider submit a report.

- `backend/`  FastAPI + uvicorn API (Python 3). Pure, unit-tested rule/stats logic.
- `frontend/` React + Vite replay-demo front end. It runs standalone from static files
  (no backend required), reading events from `frontend/public/data/`, so it can be hosted
  on GitHub Pages as a public testing link for judges.

## Run the frontend (replay demo)

```bash
cd app/frontend
npm install
npm run dev        # Vite prints the local URL

npm run build      # production build into app/frontend/dist/
npm run preview    # serve the built dist/ to check it
```

The replay clip (`app/frontend/public/media/demo_clip.mov`) is local-only and gitignored;
the app falls back to a placeholder stage when it is absent (e.g. on GitHub Pages). To ship
a playable clip to Pages, add a compressed `demo_clip.mp4` and point `CONFIG.videoSrc`
(`app/frontend/src/config.js`) at it. See `app/frontend/README.md` for details.

## Run the backend

```bash
# from the repo root
python -m venv .venv && source .venv/bin/activate   # if you don't already have one
pip install -r app/backend/requirements.txt

cd app/backend
uvicorn main:app --reload
# open http://127.0.0.1:8000/docs to try the endpoints
```

### Endpoints
- `GET /`              hint pointing to `/docs`
- `GET /api/status?t=` lane status at second `t` of the recording (default `t=0`)
- `GET /api/blockages` confirmed in-bike-lane blockages with `duration_seconds`
- `GET /api/stats`     totals, average/longest duration, count by vehicle, bikes by hour
- `POST /api/reports`  submit a rider report; `GET /api/reports` lists them (in-memory)

## Run the tests

```bash
python -m pytest app/backend/tests -q
```
