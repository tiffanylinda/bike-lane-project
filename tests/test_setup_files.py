"""Fast checks on the project's setup files: ``requirements.txt`` and ``.gitignore``.

These are pure file-content assertions (no video, no model, no imports from ``src``).
They lock in the pinned-dependency contract and the privacy-related ignore entries so a
later edit that loosens a pin or drops an ignore line is caught immediately.

Paths are resolved relative to the repo root (the parent of this ``tests/`` folder) so
the tests pass no matter the current working directory.

Requirements exercised: 10.1, 10.2, 10.3.
"""

import re
from pathlib import Path

# The repo root is the parent of the directory holding this test file.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Exactly the packages the design requires in requirements.txt, each pinned with "==".
EXPECTED_PACKAGES = {
    "ultralytics",
    "opencv-python",
    "pandas",
    "numpy",
    "matplotlib",
    "plotly",
    "requests",
    "pyyaml",
    "pytest",
}

# A requirements line: <name>==<version>. Names may contain letters, digits, '-', '_', '.'.
PIN_RE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[A-Za-z0-9._-]+)$")


def _requirement_lines():
    """Yield the non-blank, non-comment lines of requirements.txt (stripped)."""
    text = (REPO_ROOT / "requirements.txt").read_text()
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            yield line


def test_requirements_every_line_is_pinned_with_double_equals():
    for line in _requirement_lines():
        match = PIN_RE.match(line)
        assert match is not None, f"requirements line is not a plain '==' pin: {line!r}"
        # Guard against loose or unpinned specifiers sneaking in.
        assert ">=" not in line, f"requirements line uses a minimum bound, not '==': {line!r}"
        assert "<=" not in line, f"requirements line uses an upper bound, not '==': {line!r}"
        assert "~=" not in line, f"requirements line uses a compatible release, not '==': {line!r}"


def test_requirements_package_set_matches_expected():
    names = {PIN_RE.match(line).group("name") for line in _requirement_lines()}
    assert names == EXPECTED_PACKAGES


def test_gitignore_contains_required_entries():
    lines = {
        raw.strip()
        for raw in (REPO_ROOT / ".gitignore").read_text().splitlines()
    }
    for required in ("data/raw_video/", "demo/media/", "*.mp4"):
        assert required in lines, f".gitignore is missing required entry: {required!r}"
