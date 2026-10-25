"""Pure statistics calculations over loaded data.

No FastAPI, no file I/O: these functions take already-loaded data (the lists produced by
data_loader) and return plain dicts. That keeps them trivial to unit-test and reusable by
both the API and any analysis script.
"""


def blockage_stats(blockages):
    """Summarise a list of blockage event dicts.

    Only events in the bike lane (in_bike_lane true) count as real blockages -- a car
    stopped in a travel lane is not what the demo cares about.

    Returns a dict with:
      total:            number of confirmed blockages
      average_duration: mean duration in seconds (0 if none)
      longest_duration: max duration in seconds (0 if none)
      by_vehicle:       {vehicle_type: count}
    """
    confirmed = [b for b in blockages if b.get("in_bike_lane")]

    if not confirmed:
        return {
            "total": 0,
            "average_duration": 0.0,
            "longest_duration": 0.0,
            "by_vehicle": {},
        }

    durations = [b["duration_seconds"] for b in confirmed]

    by_vehicle = {}
    for b in confirmed:
        vehicle = b["vehicle"]
        by_vehicle[vehicle] = by_vehicle.get(vehicle, 0) + 1

    return {
        "total": len(confirmed),
        "average_duration": sum(durations) / len(durations),
        "longest_duration": max(durations),
        "by_vehicle": by_vehicle,
    }


def bikes_by_hour(telraam_rows):
    """Return bike counts keyed by hour as {hour: bikes}.

    telraam_rows is the list of {hour, bikes} dicts from data_loader.load_telraam.
    """
    return {row["hour"]: row["bikes"] for row in telraam_rows}
