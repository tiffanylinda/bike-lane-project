"""Pure lane-status rule logic.

This module has NO FastAPI imports on purpose: it is plain Python that the unit tests and
the API both call, so the rule is defined in exactly one place. The replay demo and the
API therefore always agree on what the lane status is at any given second.

The rule (the USER's rule for this app, see config.py):
  * "possible" -> event active and it started less than POSSIBLE_SECONDS ago
  * "blocked"  -> event active and it started POSSIBLE_SECONDS or more ago (until it ends)
  * "clear"    -> no event active (before any event, or after the active event ends)

Events are expressed in "seconds into the recording" via start_s and end_s so this logic
never has to know about dates or time zones -- data_loader does that conversion once.
"""

from config import POSSIBLE_SECONDS


def _event_active(event, t_seconds):
    """True if the event is active at time t (inclusive of both ends).

    We include the end second so an event that ends exactly at t still counts as active;
    the moment after end_s the lane becomes clear.
    """
    return event["start_s"] <= t_seconds <= event["end_s"]


def status_at(t_seconds, events):
    """Compute lane status at `t_seconds` into the recording.

    Args:
        t_seconds: float/int seconds since VIDEO_START.
        events: list of dicts, each with at least "start_s" and "end_s" (seconds into the
            recording). Extra keys (vehicle, camera, ...) are passed through untouched.

    Returns:
        dict with:
          status:       "clear" | "possible" | "blocked"
          active_event: the event dict that is active at t, or None
          elapsed:      seconds since that event started (None if clear)
          rule:         short human-readable note on which branch fired
    """
    # Find the first event that is active right now. Events in the demo do not overlap,
    # so "first active" is unambiguous; if data ever overlaps we take the earliest.
    active = None
    for event in sorted(events, key=lambda e: e["start_s"]):
        if _event_active(event, t_seconds):
            active = event
            break

    if active is None:
        return {
            "status": "clear",
            "active_event": None,
            "elapsed": None,
            "rule": "no active event",
        }

    elapsed = t_seconds - active["start_s"]

    if elapsed < POSSIBLE_SECONDS:
        status = "possible"
        rule = f"event active for {elapsed:.0f}s (< {POSSIBLE_SECONDS}s grace) -> possible"
    else:
        status = "blocked"
        rule = f"event active for {elapsed:.0f}s (>= {POSSIBLE_SECONDS}s) -> blocked"

    return {
        "status": status,
        "active_event": active,
        "elapsed": elapsed,
        "rule": rule,
    }
