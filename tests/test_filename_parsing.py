"""Unit tests for ``parse_video_name`` (Task 3.1, Requirements 2.1, 2.3).

These are fast, example-based ``pytest`` cases over the pure filename parser in
``src/detect.py``. Per the design Testing Strategy, there is NO property-based testing,
NO ``hypothesis``, and NO real video or YOLO model involved — just a short, focused set
of hand-written cases that check the camera ID, the timezone-aware America/New_York
start time, and that malformed names fail loudly with a ``ValueError``.
"""

import pytest

from src.detect import parse_video_name


# Valid names: (filename-or-path, expected camera id, expected y/m/d/h/m). We include a
# plain filename for each camera plus a full Drive-style path to prove the parser works
# off the stem only (no directory, no extension).
VALID_CASES = [
    ("commave_camA_2024-05-17_0830.mp4", "camA", (2024, 5, 17, 8, 30)),
    ("commave_camB_2024-12-31_2359.mp4", "camB", (2024, 12, 31, 23, 59)),
    (
        "/content/drive/MyDrive/raw/commave_camB_2024-01-02_1345.mp4",
        "camB",
        (2024, 1, 2, 13, 45),
    ),
]


@pytest.mark.parametrize("path, expected_cam, expected_fields", VALID_CASES)
def test_parse_video_name_valid(path, expected_cam, expected_fields):
    camera_id, start = parse_video_name(path)

    year, month, day, hour, minute = expected_fields
    assert camera_id == expected_cam
    assert (start.year, start.month, start.day, start.hour, start.minute) == (
        year,
        month,
        day,
        hour,
        minute,
    )

    # The start time must be timezone-aware (a UTC offset exists) and anchored to the
    # America/New_York zone, so every downstream timestamp carries the right offset.
    assert start.utcoffset() is not None
    assert start.tzinfo is not None
    assert start.tzinfo.key == "America/New_York"


# Malformed names that must each raise ValueError:
# - no pattern at all
# - missing the HHMM time part
# - empty camera id
# - invalid calendar month (13) -> datetime construction raises ValueError
# - invalid time (minute 99) -> datetime construction raises ValueError
MALFORMED_NAMES = [
    "video.mp4",
    "commave_camA_2024-05-17.mp4",
    "commave__2024-05-17_0830.mp4",
    "commave_camA_2024-13-01_0830.mp4",
    "commave_camA_2024-05-17_2599.mp4",
]


@pytest.mark.parametrize("name", MALFORMED_NAMES)
def test_parse_video_name_malformed_raises(name):
    with pytest.raises(ValueError):
        parse_video_name(name)
