"""Test 3: the blockage listing from real sample data.

Loads the sample CSV through data_loader (exercising both mm:ss and ISO parsing), then
confirms duration_seconds is computed correctly and that filtering to in_bike_lane keeps
only confirmed bike-lane blockages. No server, no network -- just the pure loader.
"""

from datetime import datetime

from data_loader import load_blockages, parse_time


def test_parse_time_handles_mmss_offset():
    start = datetime.fromisoformat("2024-05-17T08:30:00-04:00")
    absolute, seconds = parse_time("01:05", start)
    assert seconds == 65.0
    assert absolute == datetime.fromisoformat("2024-05-17T08:31:05-04:00")


def test_parse_time_handles_iso():
    start = datetime.fromisoformat("2024-05-17T08:30:00-04:00")
    absolute, seconds = parse_time("2024-05-17T08:31:05-04:00", start)
    assert seconds == 65.0
    assert absolute == datetime.fromisoformat("2024-05-17T08:31:05-04:00")


def test_duration_seconds_computed():
    events = load_blockages()
    # The first sample row runs 00:20 -> 01:25 = 65 seconds.
    first = next(e for e in events if e["start_s"] == 20.0)
    assert first["duration_seconds"] == 65.0


def test_filter_to_confirmed_bike_lane():
    events = load_blockages()
    confirmed = [e for e in events if e["in_bike_lane"]]
    # The sample data has exactly one non-bike-lane row (the double-parked car).
    assert len(confirmed) == len(events) - 1
    assert all(e["in_bike_lane"] for e in confirmed)


def test_longest_sample_blockage_is_90s():
    events = load_blockages()
    longest = max(e["duration_seconds"] for e in events if e["in_bike_lane"])
    assert longest == 90.0
