"""Pydantic response/request models for the Comm Ave Lane Watch API.

These give FastAPI a typed schema so /docs shows clear request and response shapes, which
is exactly what the user wants to test first.
"""

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field


class Blockage(BaseModel):
    """One confirmed blockage in the bike lane."""

    camera: str
    vehicle: str
    in_bike_lane: bool
    notes: str = ""
    start: datetime
    end: datetime
    start_seconds: float = Field(..., description="Seconds into the recording when it began")
    end_seconds: float = Field(..., description="Seconds into the recording when it ended")
    duration_seconds: float


class ActiveEvent(BaseModel):
    """The event currently affecting the lane, if any."""

    camera: str
    vehicle: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float


class Status(BaseModel):
    """Current lane status at a point in the recording."""

    status: str = Field(..., description='"clear", "possible", or "blocked"')
    t_seconds: float = Field(..., description="Seconds into the recording this status is for")
    active_event: Optional[ActiveEvent] = None
    message: str = Field(..., description="Rider-facing one-liner, e.g. 'truck, 1 min'")
    rule: str = Field(..., description="Which rule branch produced the status")
    last_updated: datetime


class Stats(BaseModel):
    """Aggregate statistics for the evidence view."""

    total_blockages: int
    average_duration_seconds: float
    longest_duration_seconds: float
    count_by_vehicle: Dict[str, int]
    bikes_by_hour: Dict[int, int]


class ReportIn(BaseModel):
    """A rider-submitted report coming in from the front end.

    `time` is a free-form string, not a datetime, because riders think in mm:ss into the
    clip (e.g. "01:20") just like blockages.csv does. We accept the same dual format the
    CSV uses -- mm:ss offset OR an ISO datetime -- and resolve it on the server. Typing it
    as Optional[datetime] would reject "01:20" with a 422.
    """

    location_note: str = Field(..., description="Where the rider saw the issue")
    time: Optional[str] = Field(
        None,
        description='When it happened, as the rider typed it: an mm:ss offset into the '
        'recording (e.g. "01:20") or an ISO datetime. Omit if unknown.',
    )


class Report(ReportIn):
    """A stored rider report echoed back with server-assigned fields.

    We echo the raw `time` string back unchanged and add `time_seconds`, the server's best
    effort at resolving that string to seconds-into-recording (None if omitted or not
    parseable), so the front end and /docs have an unambiguous numeric handle.
    """

    id: int
    time_seconds: Optional[float] = Field(
        None, description="Resolved seconds into the recording, or None if omitted/unparseable"
    )
    received_at: datetime
