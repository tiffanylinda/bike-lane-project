"""Fast, example-based unit tests for the cheap pure helpers in ``src.detect``.

These tests cover task 3.5's "secondary helpers" that exist today and need neither a
real video nor the YOLO model:

- the frame-selection index math (``select_frame_indices`` / ``should_process_frame``), and
- the CLI validation of ``--stride`` and ``--max-seconds`` (via ``parse_args``).

Per the design's Testing Strategy these are hand-written example cases (a handful each) —
there is NO property-based testing and NO ``hypothesis`` here.

NOTE on skip/overwrite (Req 5.1, 5.2): that decision ("process iff ``overwrite or not
csv_exists``") is implemented inside ``process_video`` in task 5.2, which has not landed
yet. There is no standalone pure helper for it today, so it is intentionally NOT tested
here and will be covered when task 5.2 is implemented.

Requirements exercised: 3.4, 3.5, 13.1, 13.3, 13.4.
"""

import pytest

from src.detect import parse_args, select_frame_indices, should_process_frame


# --- 1. Frame-selection index math (Req 3.4, 13.1, 13.3) --------------------
#
# The design fixes the selected set to exactly:
#   { i : i % stride == 0 and (max_seconds is None or i/fps < max_seconds) }
# We compute that reference set independently and assert the helper agrees, for a
# handful of cases that each isolate one part of the rule (stride only, time only,
# and the two composed).


def _expected_indices(total_frames, fps, stride, max_seconds):
    """Reference implementation of the selection rule, straight from the spec."""
    return [
        i
        for i in range(total_frames)
        if i % stride == 0 and (max_seconds is None or i / fps < max_seconds)
    ]


@pytest.mark.parametrize(
    "total_frames, fps, stride, max_seconds",
    [
        # stride=1, no time limit -> every index 0..total-1.
        (10, 30.0, 1, None),
        # stride=3, no time limit -> 0, 3, 6, 9.
        (10, 30.0, 3, None),
        # stride=1, fps=30, max_seconds=1.0, total=60 -> i/30 < 1.0 means i < 30,
        # so indices 0..29.
        (60, 30.0, 1, 1.0),
        # stride=2 AND max_seconds together: both predicates must compose.
        (60, 30.0, 2, 1.0),
    ],
)
def test_select_frame_indices_matches_rule(total_frames, fps, stride, max_seconds):
    expected = _expected_indices(total_frames, fps, stride, max_seconds)
    assert list(select_frame_indices(total_frames, fps, stride, max_seconds)) == expected


def test_select_frame_indices_stride_one_is_every_index():
    # Explicit spelled-out expectation for the simplest case.
    assert list(select_frame_indices(10, 30.0, 1, None)) == list(range(10))


def test_select_frame_indices_stride_three():
    assert list(select_frame_indices(10, 30.0, 3, None)) == [0, 3, 6, 9]


def test_select_frame_indices_time_window_is_strict_less_than():
    # fps=30, max_seconds=1.0 -> keep i while i/30 < 1.0, i.e. i in 0..29.
    result = list(select_frame_indices(60, 30.0, 1, 1.0))
    assert result == list(range(30))
    # Frame 30 sits at exactly 1.0s and must be excluded by the strict "<".
    assert 30 not in result


def test_select_frame_indices_stride_and_time_compose():
    # stride=2 on indices 0..29 (the time window) -> even indices 0..28.
    result = list(select_frame_indices(60, 30.0, 2, 1.0))
    assert result == list(range(0, 30, 2))


def test_should_process_frame_agrees_with_selection():
    # The per-frame predicate and the generator are the same rule two ways.
    total, fps, stride, max_seconds = 60, 30.0, 2, 1.0
    selected = set(select_frame_indices(total, fps, stride, max_seconds))
    for i in range(total):
        assert should_process_frame(i, fps, stride, max_seconds) == (i in selected)


# --- 2. Invalid --stride / --max-seconds rejection (Req 3.5, 13.4) ----------
#
# argparse validates the values via the type= functions during parsing; a bad value
# makes argparse print an error and call sys.exit(), which raises SystemExit. We give
# valid --input/--output so ONLY the value under test can trigger the exit.


def _argv(*extra):
    """Build an argv with the two required args plus whatever we're testing."""
    return ["--input", "x", "--output", "y", *extra]


@pytest.mark.parametrize("bad_stride", ["0", "-1", "1.5", "abc"])
def test_invalid_stride_exits(bad_stride):
    with pytest.raises(SystemExit):
        parse_args(_argv("--stride", bad_stride))


@pytest.mark.parametrize("bad_max_seconds", ["0", "-2", "abc"])
def test_invalid_max_seconds_exits(bad_max_seconds):
    with pytest.raises(SystemExit):
        parse_args(_argv("--max-seconds", bad_max_seconds))


def test_valid_parse_and_defaults():
    args = parse_args(
        ["--input", "in", "--output", "out", "--stride", "2", "--max-seconds", "1.5"]
    )
    # The explicitly provided values parse to the right types.
    assert args.stride == 2
    assert args.max_seconds == 1.5
    assert args.input == "in"
    assert args.output == "out"
    # The defaults the design promises when flags/options are omitted.
    assert args.overwrite is False
    assert args.save_annotated is False
    assert args.config_dir == "config/"
