"""Test 2: stats calculations over a small synthetic set of blockages.

Checks total, average duration, longest duration and count-by-vehicle. Only in-bike-lane
events should count, so one non-bike-lane row is included to confirm it is excluded.
"""

from stats import bikes_by_hour, blockage_stats

BLOCKAGES = [
    {"vehicle": "truck", "in_bike_lane": True, "duration_seconds": 60.0},
    {"vehicle": "car", "in_bike_lane": True, "duration_seconds": 20.0},
    {"vehicle": "truck", "in_bike_lane": True, "duration_seconds": 40.0},
    # Not in the bike lane -> must be ignored by the stats.
    {"vehicle": "bus", "in_bike_lane": False, "duration_seconds": 999.0},
]


def test_total_counts_only_bike_lane():
    assert blockage_stats(BLOCKAGES)["total"] == 3


def test_average_duration():
    # (60 + 20 + 40) / 3 = 40
    assert blockage_stats(BLOCKAGES)["average_duration"] == 40.0


def test_longest_duration_ignores_non_bike_lane():
    # 60 is the longest among in-bike-lane events; the 999 bus is excluded.
    assert blockage_stats(BLOCKAGES)["longest_duration"] == 60.0


def test_count_by_vehicle():
    assert blockage_stats(BLOCKAGES)["by_vehicle"] == {"truck": 2, "car": 1}


def test_empty_is_safe():
    summary = blockage_stats([])
    assert summary["total"] == 0
    assert summary["average_duration"] == 0.0
    assert summary["longest_duration"] == 0.0
    assert summary["by_vehicle"] == {}


def test_bikes_by_hour():
    rows = [{"hour": 8, "bikes": 28}, {"hour": 9, "bikes": 19}]
    assert bikes_by_hour(rows) == {8: 28, 9: 19}
