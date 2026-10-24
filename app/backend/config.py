"""Configuration for the Comm Ave Lane Watch backend.

Everything time-related is anchored to VIDEO_START, the real-world clock time at which
the recording begins. The replay demo measures "seconds into the recording" from here,
so a given second always maps to the same wall-clock time -- that determinism is what
lets the browser replay and the backend agree on lane status.

The status thresholds below are the USER's rule for THIS app (intentionally simpler than
the 30s rule in product.md): a vehicle that just stopped is only "possible" for a short
grace period, then we commit to "blocked"; after it leaves we debounce briefly so a quick
flicker does not toggle the status back and forth.
"""

# Wall-clock time the recording starts. ISO 8601 with a -04:00 offset = America/New_York
# during daylight saving. mm:ss offsets in the data are added to this to get absolute times.
VIDEO_START = "2024-05-17T08:30:00-04:00"

# A stopped vehicle is only a "possible" blockage for its first 10 seconds. This grace
# period avoids flagging a car that is briefly paused (e.g. yielding) as a real blockage.
POSSIBLE_SECONDS = 10

# After this many seconds of a vehicle sitting in the lane we commit to "blocked".
# Equal to POSSIBLE_SECONDS here: possible for the first 10s, blocked from 10s onward.
BLOCKED_AFTER_SECONDS = 10

# Debounce window after an event ends before we are confident the lane is truly clear.
# Kept small so the demo clears promptly once a vehicle moves on.
CLEAR_AFTER_SECONDS = 5
