"""Unit tests for the CSV column contract and the uid builder (Task 3.3, Requirements 4.1, 4.3).

These are fast, example-based ``pytest`` cases over two pure pieces of ``src/detect.py``:
``CSV_COLUMNS`` (the fixed detections-CSV header, the contract downstream pandas code
relies on) and ``build_uid`` (which joins Video_Name, Camera_ID, and track_id into one
globally unique id). Per the design Testing Strategy, there is NO property-based testing,
NO ``hypothesis``, and NO real video or YOLO model — just a short, focused set of
hand-written cases.
"""

from src.detect import CSV_COLUMNS, build_uid


# The exact column contract from Req 4.1: this list, in this order, is the single source
# of truth every detections CSV is written against. If this test fails, the contract
# with the downstream pandas code has been broken.
EXPECTED_COLUMNS = [
    "video",
    "camera",
    "time",
    "frame",
    "track_id",
    "uid",
    "cls",
    "conf",
    "cx",
    "cy",
    "w",
    "h",
]


def test_csv_columns_exact_order():
    # Full-list equality checks both the names AND their order in one assertion.
    assert CSV_COLUMNS == EXPECTED_COLUMNS


def test_csv_columns_length_is_twelve():
    assert len(CSV_COLUMNS) == 12


def test_build_uid_camA_example():
    # Req 4.3: uid = Video_Name + Camera_ID + track_id joined with single underscores.
    assert (
        build_uid("commave_camA_2024-05-17_0830", "camA", 7)
        == "commave_camA_2024-05-17_0830_camA_7"
    )


def test_build_uid_camB_example():
    assert (
        build_uid("commave_camB_2024-12-31_2359", "camB", 42)
        == "commave_camB_2024-12-31_2359_camB_42"
    )


def test_build_uid_is_deterministic():
    # Same inputs -> same output, so a given detection always maps to one uid.
    first = build_uid("commave_camA_2024-05-17_0830", "camA", 7)
    second = build_uid("commave_camA_2024-05-17_0830", "camA", 7)
    assert first == second


def test_build_uid_differs_when_track_id_differs():
    # Different track ids within the same video/camera must yield different uids, so two
    # distinct tracked objects never collide.
    uid_7 = build_uid("commave_camA_2024-05-17_0830", "camA", 7)
    uid_8 = build_uid("commave_camA_2024-05-17_0830", "camA", 8)
    assert uid_7 != uid_8


def test_build_uid_differs_when_camera_differs():
    # The same track_id on two cameras filming the same street at the same time must not
    # collide — the camera id in the uid keeps them apart.
    uid_cam_a = build_uid("commave_camA_2024-05-17_0830", "camA", 7)
    uid_cam_b = build_uid("commave_camB_2024-05-17_0830", "camB", 7)
    assert uid_cam_a != uid_cam_b
