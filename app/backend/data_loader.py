"""Load blockage events and bike counts from CSV into plain Python structures.

Why this module exists: the raw CSVs mix two time formats (absolute ISO datetimes and
short mm:ss offsets from the start of the clip). We normalise everything here, once, so
the rest of the backend only ever deals with clean absolute datetimes plus "seconds into
the recording". lane_status.py and stats.py then stay simple and pure.
"""

import os
import re
from datetime import datetime, timedelta

import pandas as pd

from config import VIDEO_START

# Resolve data paths relative to THIS file, not the current working directory, so tests
# and uvicorn both find the CSVs no matter where they are launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
BLOCKAGES_CSV = os.path.join(DATA_DIR, "blockages.csv")
TELRAAM_CSV = os.path.join(DATA_DIR, "telraam.csv")

# Matches short offsets like "0:12", "00:12", "1:05", "12:30" (minutes:seconds).
_MMSS = re.compile(r"^\s*(\d{1,2}):([0-5]?\d)\s*$")


def _video_start_dt():
    """Parse VIDEO_START once into a timezone-aware datetime."""
    return datetime.fromisoformat(VIDEO_START)


def parse_time(value, start_dt=None):
    """Turn one cell into (absolute_datetime, seconds_into_recording).

    A value that looks like mm:ss (e.g. "00:12" or "1:05") is treated as an offset in
    seconds from VIDEO_START. Anything else is parsed as an ISO datetime. This lets the
    sample data exercise both paths while the rest of the code sees uniform values.
    """
    if start_dt is None:
        start_dt = _video_start_dt()

    text = str(value).strip()
    match = _MMSS.match(text)
    if match:
        minutes, seconds = int(match.group(1)), int(match.group(2))
        offset = minutes * 60 + seconds
        absolute = start_dt + timedelta(seconds=offset)
        return absolute, float(offset)

    # Not an mm:ss offset -> parse as an absolute ISO datetime.
    absolute = datetime.fromisoformat(text)
    seconds = (absolute - start_dt).total_seconds()
    return absolute, float(seconds)


def load_blockages(path=BLOCKAGES_CSV):
    """Read blockages.csv into a list of normalised event dicts.

    Expected columns: camera, start, end, vehicle, in_bike_lane, notes.
    Each returned dict adds absolute datetimes (start_dt/end_dt), seconds-into-recording
    (start_s/end_s) and duration_seconds, and coerces in_bike_lane to a real bool.
    """
    start_dt = _video_start_dt()
    frame = pd.read_csv(path)

    events = []
    for _, row in frame.iterrows():
        start_abs, start_s = parse_time(row["start"], start_dt)
        end_abs, end_s = parse_time(row["end"], start_dt)
        events.append(
            {
                "camera": str(row["camera"]),
                "vehicle": str(row["vehicle"]),
                "in_bike_lane": _as_bool(row["in_bike_lane"]),
                "notes": "" if pd.isna(row.get("notes")) else str(row["notes"]),
                "start_dt": start_abs,
                "end_dt": end_abs,
                "start_s": start_s,
                "end_s": end_s,
                "duration_seconds": max(0.0, end_s - start_s),
            }
        )
    return events


def load_telraam(path=TELRAAM_CSV):
    """Read telraam.csv into a list of {hour, bikes} dicts sorted by hour.

    The sample file is intentionally simple (columns hour,bikes) so stats.py can read
    "bike counts by hour" directly without reshaping real Telraam exports.
    """
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        rows.append({"hour": int(row["hour"]), "bikes": int(row["bikes"])})
    rows.sort(key=lambda r: r["hour"])
    return rows


def _as_bool(value):
    """Coerce common truthy spellings from a CSV into a real bool."""
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}
