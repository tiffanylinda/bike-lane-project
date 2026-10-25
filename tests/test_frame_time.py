"""Fast example-based unit tests for ``frame_time`` in ``src/detect.py`` (Task 3.2).

These tests cover the timestamp math (Req 4.2, 13.2): ``frame_time(start, frame_index, fps)``
equals ``start + frame/fps`` seconds, stays timezone-aware in America/New_York (so the ISO
string carries its UTC offset), and is formatted to millisecond precision.

Per the design Testing Strategy these are a short set of hand-written example cases: no
property-based testing, no hypothesis, and no real video or YOLO model.
"""

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.detect import frame_time, parse_video_name

NY = ZoneInfo("America/New_York")


def test_frame_zero_equals_start_with_millisecond_offset():
    """Frame 0 is exactly the start, with a .000 millisecond fraction and a UTC offset."""
    start = datetime(2024, 5, 17, 8, 30, tzinfo=NY)

    result = frame_time(start, 0, 30.0)

    # Equals the start formatted at millisecond precision, offset included.
    assert result == start.isoformat(timespec="milliseconds")
    # May is EDT, so the offset is -04:00, and the fraction shows exactly .000.
    assert result == "2024-05-17T08:30:00.000-04:00"


def test_full_second_and_half_second_offsets():
    """30 frames at 30 fps is +1.000s; 45 frames is +1.500s."""
    start = datetime(2024, 5, 17, 8, 30, tzinfo=NY)

    one_second = frame_time(start, 30, 30.0)
    one_and_half = frame_time(start, 45, 30.0)

    assert one_second == (start + timedelta(seconds=1.0)).isoformat(timespec="milliseconds")
    assert one_second == "2024-05-17T08:30:01.000-04:00"
    assert one_and_half == (start + timedelta(seconds=1.5)).isoformat(timespec="milliseconds")
    assert one_and_half == "2024-05-17T08:30:01.500-04:00"


def test_parsed_back_value_is_tz_aware_and_matches_expected():
    """Parsing the string back equals start + frame/fps and stays timezone-aware."""
    # Mirror real usage: get the start from parse_video_name.
    _camera_id, start = parse_video_name("commave_camA_2024-05-17_0830.mp4")
    frame_index, fps = 45, 30.0

    result = frame_time(start, frame_index, fps)
    parsed = datetime.fromisoformat(result)

    assert parsed == start + timedelta(seconds=frame_index / fps)
    # Timezone-aware: the offset survived the round trip.
    assert parsed.utcoffset() is not None


def test_string_has_exactly_three_fractional_second_digits():
    """The formatted string carries millisecond precision: exactly 3 fractional digits."""
    start = datetime(2024, 5, 17, 8, 30, tzinfo=NY)

    # A frame that lands on a non-round time still shows exactly 3 fraction digits.
    result = frame_time(start, 7, 30.0)

    # ...SS.mmm before the +/- offset, where mmm is exactly three digits.
    assert re.search(r"\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}$", result)
