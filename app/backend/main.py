"""FastAPI app for Comm Ave Lane Watch.

Run it from the app/backend directory:
    uvicorn main:app --reload
then open http://127.0.0.1:8000/docs to try the endpoints.

The app loads its data once on startup and serves it read-only, except for rider reports
which are kept in a simple in-memory list (no database -- this is a demo).

Import note: the sibling modules (config, data_loader, ...) are imported flatly so the
pure modules can be unit-tested and uvicorn can run with `main:app`. We also support
`import app.backend.main` by adding this folder to sys.path when imported as a package.
"""

import os
import sys
from datetime import datetime, timezone

# Make flat imports (config, data_loader, ...) work whether this file is run by uvicorn
# from app/backend, or imported as app.backend.main from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

import data_loader  # noqa: E402
import stats as stats_module  # noqa: E402
from lane_status import status_at  # noqa: E402
from models import (  # noqa: E402
    ActiveEvent,
    Blockage,
    Report,
    ReportIn,
    Stats,
    Status,
)

app = FastAPI(
    title="Comm Ave Lane Watch API",
    description="Bike lane blockage status, blockages, stats and rider reports for the demo.",
    version="0.1.0",
)

# CORS: the React/Vite dev server runs on :5173. Allowing all origins is fine for this
# offline demo; it keeps the public GitHub Pages front end able to reach a local backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # includes http://localhost:5173 and http://127.0.0.1:5173
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at import/startup and reused by every request.
_STATE = {"blockages": [], "telraam": []}

# In-memory store for rider reports. Resets when the server restarts (fine for a demo).
_REPORTS = []


@app.on_event("startup")
def _load_data():
    """Load CSV data into memory when the server starts."""
    _STATE["blockages"] = data_loader.load_blockages()
    _STATE["telraam"] = data_loader.load_telraam()


# Load immediately too, so importing the module (e.g. for a smoke test) has data ready
# even without the ASGI startup event firing.
try:
    _load_data()
except Exception:  # pragma: no cover - data is optional at import time
    pass


def _human_message(status, active):
    """Build a short rider-facing message, e.g. 'truck, 1 min' or 'Lane clear'."""
    if status == "clear" or active is None:
        return "Lane clear"
    minutes = active["duration_seconds"] / 60.0
    if minutes >= 1:
        length = f"{round(minutes)} min"
    else:
        length = f"{round(active['duration_seconds'])} s"
    label = "Blockage" if status == "blocked" else "Possible blockage"
    return f"{label}: {active['vehicle']}, {length}"


def _to_active_event(event):
    """Convert an internal event dict into the ActiveEvent response model."""
    if event is None:
        return None
    return ActiveEvent(
        camera=event["camera"],
        vehicle=event["vehicle"],
        start_seconds=event["start_s"],
        end_seconds=event["end_s"],
        duration_seconds=event["duration_seconds"],
    )


@app.get("/")
def root():
    """Tiny hint pointing at the interactive docs."""
    return {"app": "Comm Ave Lane Watch API", "docs": "/docs"}


@app.get("/api/status", response_model=Status)
def get_status(
    t: float = Query(
        0.0,
        description="Seconds into the recording to evaluate. Defaults to 0 so the replay "
        "demo is deterministic.",
    )
):
    """Lane status at second `t` of the recording.

    We default t=0 (not wall-clock 'now') on purpose: a replay demo must show the same
    thing every time it is played, and the browser drives the clock by passing t.
    """
    result = status_at(t, _STATE["blockages"])
    active = result["active_event"]
    return Status(
        status=result["status"],
        t_seconds=t,
        active_event=_to_active_event(active),
        message=_human_message(result["status"], active),
        rule=result["rule"],
        last_updated=datetime.now(timezone.utc),
    )


@app.get("/api/blockages", response_model=list[Blockage])
def get_blockages():
    """List confirmed in-bike-lane blockages with their durations."""
    confirmed = [b for b in _STATE["blockages"] if b["in_bike_lane"]]
    return [
        Blockage(
            camera=b["camera"],
            vehicle=b["vehicle"],
            in_bike_lane=b["in_bike_lane"],
            notes=b["notes"],
            start=b["start_dt"],
            end=b["end_dt"],
            start_seconds=b["start_s"],
            end_seconds=b["end_s"],
            duration_seconds=b["duration_seconds"],
        )
        for b in confirmed
    ]


@app.get("/api/stats", response_model=Stats)
def get_stats():
    """Aggregate blockage and bike-count statistics."""
    summary = stats_module.blockage_stats(_STATE["blockages"])
    return Stats(
        total_blockages=summary["total"],
        average_duration_seconds=summary["average_duration"],
        longest_duration_seconds=summary["longest_duration"],
        count_by_vehicle=summary["by_vehicle"],
        bikes_by_hour=stats_module.bikes_by_hour(_STATE["telraam"]),
    )


@app.post("/api/reports", response_model=Report)
def create_report(report: ReportIn):
    """Accept a rider report, store it in memory, and echo it back with an id.

    Riders think in mm:ss into the clip, so `report.time` is a free-form string that may be
    an mm:ss offset ("01:20") or an ISO datetime -- the same dual format data_loader already
    handles for blockages.csv. We resolve it to seconds-into-recording here. A garbage string
    must not 500: we still accept and store the report, echo the raw string back, and leave
    time_seconds=None.
    """
    received = datetime.now(timezone.utc)

    time_seconds = None
    if report.time is not None:
        try:
            _absolute_dt, seconds = data_loader.parse_time(report.time)
            time_seconds = seconds
        except Exception:
            # Unparseable string -> keep the report, just can't resolve a numeric offset.
            time_seconds = None

    stored = Report(
        id=len(_REPORTS) + 1,
        location_note=report.location_note,
        time=report.time,
        time_seconds=time_seconds,
        received_at=received,
    )
    _REPORTS.append(stored)
    return stored


@app.get("/api/reports", response_model=list[Report])
def list_reports():
    """Return the in-memory rider reports (handy for the front end)."""
    return _REPORTS
