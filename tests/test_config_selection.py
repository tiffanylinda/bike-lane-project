"""Unit tests for ``load_camera_config`` (Task 3.4, Requirements 2.2, 2.4, 2.5).

These are fast, example-based ``pytest`` cases over the pure config loader in
``src/detect.py``. Per the design Testing Strategy, there is NO property-based testing,
NO ``hypothesis``, and NO real video or YOLO model involved — just a short, focused set
of hand-written cases.

Each case writes small temporary YAML files into ``tmp_path`` (pytest's per-test temp
directory fixture) so the tests NEVER depend on the repo's real ``config/`` files. The
cases check that:
- the camera id selects the matching ``<camID>.yaml`` file (Req 2.2),
- a missing config raises ``FileNotFoundError`` naming the expected path (Req 2.4), and
- a config missing ``model.weights`` / ``model.conf`` / ``model.classes`` raises
  ``KeyError`` naming the missing field (Req 2.5).
"""

import pytest

from src.detect import load_camera_config


def _write_config(directory, camera_id, body):
    """Write ``<directory>/<camera_id>.yaml`` with the given YAML text and return its path."""
    config_path = directory / f"{camera_id}.yaml"
    config_path.write_text(body)
    return config_path


# Two distinct, complete configs — different weights AND different classes — so a correct
# loader must return the values from the file whose name matches the camera id, never the
# other camera's values.
CAM_A_YAML = """\
model:
  weights: yolov8n.pt
  conf: 0.35
  classes: [0, 1, 2]
"""

CAM_B_YAML = """\
model:
  weights: yolov8s.pt
  conf: 0.50
  classes: [3, 5, 7]
"""


def test_camera_id_selects_matching_file(tmp_path):
    """The camera id picks its own ``<camID>.yaml`` — camA and camB never get mixed (Req 2.2)."""
    _write_config(tmp_path, "camA", CAM_A_YAML)
    _write_config(tmp_path, "camB", CAM_B_YAML)

    config_a = load_camera_config(tmp_path, "camA")
    config_b = load_camera_config(tmp_path, "camB")

    assert config_a["model"]["weights"] == "yolov8n.pt"
    assert config_a["model"]["conf"] == 0.35
    assert config_a["model"]["classes"] == [0, 1, 2]

    assert config_b["model"]["weights"] == "yolov8s.pt"
    assert config_b["model"]["conf"] == 0.50
    assert config_b["model"]["classes"] == [3, 5, 7]


def test_missing_config_raises_filenotfound_naming_path(tmp_path):
    """A camera id with no config file fails loudly, naming the expected path (Req 2.4)."""
    # Only camA exists; asking for camZ must point at the exact file we looked for.
    _write_config(tmp_path, "camA", CAM_A_YAML)

    with pytest.raises(FileNotFoundError) as excinfo:
        load_camera_config(tmp_path, "camZ")

    assert "camZ.yaml" in str(excinfo.value)


# A config missing exactly one of the three required model fields, one case per field.
MISSING_FIELD_CASES = [
    (
        "classes",
        """\
model:
  weights: yolov8n.pt
  conf: 0.35
""",
    ),
    (
        "conf",
        """\
model:
  weights: yolov8n.pt
  classes: [0, 1, 2]
""",
    ),
    (
        "weights",
        """\
model:
  conf: 0.35
  classes: [0, 1, 2]
""",
    ),
]


@pytest.mark.parametrize("missing_field, body", MISSING_FIELD_CASES)
def test_missing_required_field_raises_keyerror_naming_field(tmp_path, missing_field, body):
    """An incomplete config raises ``KeyError`` naming the missing field (Req 2.5)."""
    _write_config(tmp_path, "camA", body)

    with pytest.raises(KeyError) as excinfo:
        load_camera_config(tmp_path, "camA")

    assert missing_field in str(excinfo.value)
