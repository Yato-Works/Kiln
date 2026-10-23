"""Tests for the arena allocator (``kiln.ir.arena``).

The allocator is allowed to change the plan, but only in one direction: it may
never place a block on top of a block that is live at the same op, and it may
never produce a *larger* arena than the historical First-Fit insertion layout.
Both properties are checked against an independent baseline implementation
below, together with the two layouts that decide the Phase 1 certificate: the
shipped KWS graph (already minimal under First-Fit, so its layout must not
move) and the ``narrow_second`` twin, whose 20 B shortfall used to cost 1,344 B.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from kiln.ir.arena import Layout, Request, best_fit_offset, first_fit_offset, place

#: The shipped KWS graph as the planner sees it: merged ``CONV1D -> LUT``
#: blocks plus the pooled tensors that outlive them (see ROADMAP section 2).
SHIPPED_KWS: list[Request] = [
    ("t_conv0", (0, 2), 1324),
    ("t_pool0", (2, 3), 648),
    ("t_conv1", (3, 5), 1320),
    ("t_pool1", (5, 6), 332),
    ("t_conv2", (6, 8), 672),
    ("t_pool2", (8, 9), 224),
    ("t_conv3", (9, 11), 224),
    ("t_pool3", (11, 12), 112),
    ("t_conv4", (12, 13), 12),
    ("graph_out", (13, 13), 4),
]


def _spans(placed: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    """Merged address spans of ``(offset, size, first_op, last_op)`` blocks."""
    spans: list[tuple[int, int]] = []
    for start, end in sorted((offset, offset + size) for offset, size, _, _ in placed):
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans


def _first_fit_baseline(requests: list[Request]) -> int:
    """Independent test-side re-implementation of the historical allocator.

    Insertion order (``first_op``, then name) with the first-fit gap rule.  This
    is the layout every pre-refactor artifact was built with, so the allocator
    must never exceed its arena size.
    """
    placed: list[tuple[int, int, int, int]] = []
    arena_bytes = 0
    for _, (first_op, last_op), size in sorted(
        requests, key=lambda request: (request[1][0], request[0])
    ):
        blockers = [block for block in placed if block[2] <= last_op and first_op <= block[3]]
        offset = first_fit_offset(size, _spans(blockers))
        placed.append((offset, size, first_op, last_op))
        arena_bytes = max(arena_bytes, offset + size)
    return arena_bytes


@st.composite
def _programs(draw) -> list[Request]:
    """Random programs: unique names, valid live intervals, aligned sizes."""
    count = draw(st.integers(min_value=0, max_value=12))
    requests: list[Request] = []
    for index in range(count):
        first_op = draw(st.integers(min_value=0, max_value=12))
        length = draw(st.integers(min_value=0, max_value=6))
        size = draw(st.integers(min_value=0, max_value=64)) * 4
        requests.append((f"t{index}", (first_op, first_op + length), size))
    return requests


def test_empty_arena_has_zero_bytes() -> None:
    layout = place([])

    assert layout.arena_bytes == 0
    assert layout.peak_concurrent_bytes == 0
    assert layout.offsets == {}


def test_layout_records_the_winning_strategy() -> None:
    """Plans stay auditable: the winning layout strategy is reported."""
    layout = place([("only", (0, 0), 64)])

    assert layout == Layout({"only": 0}, 64, 64, "insertion-order/first-fit")


@pytest.mark.parametrize(
    ("size", "spans", "first_fit", "best_fit"),
    [
        (0, (), 0, 0),
        (16, (), 0, 0),
        (16, ((0, 16),), 16, 16),
        (16, ((0, 16), (32, 48)), 16, 16),
        # An early 128 B gap and a later 80 B gap: first-fit takes the first
        # one, best-fit prefers the tighter one.
        (32, ((0, 64), (192, 320), (400, 448)), 64, 320),
    ],
)
def test_gap_rules(size: int, spans, first_fit: int, best_fit: int) -> None:
    assert first_fit_offset(size, spans) == first_fit
    assert best_fit_offset(size, spans) == best_fit


def test_invalid_requests_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative tensor size"):
        place([("bad", (0, 1), -4)])

    with pytest.raises(ValueError, match="invalid lifetime"):
        place([("bad", (3, 2), 4)])


def test_shipped_kws_layout_does_not_move() -> None:
    """First-Fit is already optimal here, so the historical layout is kept."""
    layout = place(SHIPPED_KWS)

    assert layout.arena_bytes == 1972
    assert layout.peak_concurrent_bytes == 1972
    assert layout.strategy == "insertion-order/first-fit"
    assert layout.offsets["t_conv0"] == 0
    assert layout.offsets["t_pool0"] == 1324
    assert layout.offsets["t_conv1"] == 0


def test_first_fit_cliff_is_recovered_at_toy_scale() -> None:
    """``conv1`` is 4 B too large for the block ``conv0`` frees."""
    requests: list[Request] = [
        ("conv0", (0, 2), 40),
        ("pool0", (2, 3), 20),
        ("conv1", (3, 4), 44),
    ]
    layout = place(requests)

    assert _first_fit_baseline(requests) == 104
    assert layout.arena_bytes == 64  # pool0 + conv1, the true peak
    assert layout.peak_concurrent_bytes == 64
    assert layout.offsets == {"conv0": 0, "conv1": 0, "pool0": 44}
    assert layout.strategy == "size-descending/best-fit"


def test_first_fit_cliff_is_recovered_at_kws_scale() -> None:
    """The Phase 1 cliff: 56 channels need 1,344 B, the freed block has 1,324."""
    requests: list[Request] = [
        ("t_conv0", (0, 2), 1324),
        ("t_pool0", (2, 3), 648),
        ("t_conv1", (3, 4), 1344),
        ("t_pool1", (5, 6), 336),
        ("t_conv2", (6, 8), 672),
        ("t_pool2", (8, 9), 224),
        ("t_conv3", (9, 11), 224),
        ("t_pool3", (11, 12), 112),
        ("t_conv4", (12, 13), 12),
        ("graph_out", (13, 13), 4),
    ]
    layout = place(requests)

    # First-Fit appends the second convolution after everything that is live:
    # 1,324 (conv0) + 648 (pool0) + 1,344 (conv1) = 3,316 B.
    assert _first_fit_baseline(requests) == 3316
    assert layout.arena_bytes == 1992  # == pool0 + conv1, the true peak
    assert layout.peak_concurrent_bytes == 1992


@given(_programs())
@settings(max_examples=200, deadline=None)
def test_never_worse_than_first_fit(requests: list[Request]) -> None:
    """The historical layout is a fallback, never a target to beat downwards."""
    assert place(requests).arena_bytes <= _first_fit_baseline(requests)


@given(_programs())
@settings(max_examples=200, deadline=None)
def test_layout_only_depends_on_the_request_set(requests: list[Request]) -> None:
    """Placement is a pure function: input order cannot change the plan."""
    assert place(requests) == place(list(reversed(requests)))


@given(_programs())
@settings(max_examples=200, deadline=None)
def test_live_blocks_never_share_bytes(requests: list[Request]) -> None:
    """The one invariant the backend depends on, checked pairwise."""
    layout = place(requests)
    blocks = [
        (layout.offsets[name], size, first_op, last_op)
        for name, (first_op, last_op), size in requests
    ]

    for index, (offset, size, first_op, last_op) in enumerate(blocks):
        assert offset % 4 == 0
        for other_offset, other_size, other_first, other_last in blocks[index + 1 :]:
            if size == 0 or other_size == 0:
                continue
            if last_op < other_first or other_last < first_op:
                continue  # disjoint lifetimes: sharing bytes is the whole point
            assert offset + size <= other_offset or other_offset + other_size <= offset, (
                offset,
                size,
                first_op,
                last_op,
                other_offset,
                other_size,
            )


@given(_programs())
@settings(max_examples=200, deadline=None)
def test_arena_fits_every_block_and_the_live_peak(requests: list[Request]) -> None:
    """``arena_bytes`` is the highest address used, ``peak`` the RAM floor."""
    layout = place(requests)
    first = min((first_op for _, (first_op, _), _ in requests), default=0)
    last = max((last_op for _, (_, last_op), _ in requests), default=0)
    peak = 0
    for op in range(first, last + 1):
        peak = max(
            peak,
            sum(size for _, (first_op, last_op), size in requests if first_op <= op <= last_op),
        )

    assert layout.peak_concurrent_bytes == peak
    assert layout.arena_bytes >= peak
    assert layout.arena_bytes == max(
        (layout.offsets[name] + size for name, _, size in requests),
        default=0,
    )
