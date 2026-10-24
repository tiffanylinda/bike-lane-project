"""Test 1: the pure status rule.

possible for the first 10s of an event, blocked from 10s until the event ends, clear
before any event and after the active event ends. Uses a small synthetic events list --
no CSV, no server, no network.
"""

from lane_status import status_at

# One event occupying seconds 20..80 (a 60s blockage).
EVENTS = [{"start_s": 20.0, "end_s": 80.0, "vehicle": "truck", "camera": "camA"}]


def test_clear_before_any_event():
    assert status_at(0, EVENTS)["status"] == "clear"
    assert status_at(19, EVENTS)["status"] == "clear"


def test_possible_within_first_10s():
    # Right at the start and just under the 10s grace -> possible.
    assert status_at(20, EVENTS)["status"] == "possible"
    assert status_at(29, EVENTS)["status"] == "possible"


def test_blocked_from_10s_until_end():
    # At exactly 10s in and onward through the end second -> blocked.
    assert status_at(30, EVENTS)["status"] == "blocked"
    assert status_at(50, EVENTS)["status"] == "blocked"
    assert status_at(80, EVENTS)["status"] == "blocked"


def test_clear_after_event_ends():
    # One second past the end -> clear again.
    assert status_at(81, EVENTS)["status"] == "clear"
    assert status_at(200, EVENTS)["status"] == "clear"


def test_active_event_is_returned_when_not_clear():
    result = status_at(40, EVENTS)
    assert result["active_event"]["vehicle"] == "truck"
    # And clear returns no active event.
    assert status_at(5, EVENTS)["active_event"] is None
