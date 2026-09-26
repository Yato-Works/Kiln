"""Arena placement: the allocator behind Kiln's single flat ``.bss`` buffer.

``kiln.ir.plan`` answers *what* has to live in the arena (tensor sizes, live
ranges, fused ``LUT`` pairs); this module answers *where*.  Every block is a
byte range plus a live interval ``[first_op, last_op]``, so placement is a
weighted interval-colouring problem: two blocks may share bytes only when their
intervals do not overlap, and the arena size is the highest address used.

The textbook layout is first-fit in insertion order (``first_op``, then name).
It is cheap, deterministic and often optimal -- but not always: a block that is
a few bytes too large for the gap freed by an earlier block is appended after
everything that is still live instead of reusing that gap.  Phase 1's KWS model
shows the cost exactly: the first stage frees a 1,324 B block, the second
convolution needs 1,344 B, and the 20 missing bytes pushed the arena from
1,972 B to 3,316 B (-1,344 B of RAM for a 20 B shortfall).

:func:`place` therefore evaluates a fixed list of strategies -- the historical
first-fit insertion layout first, then size-descending orders with a best-fit
gap rule -- and keeps the historical layout unless another strategy *strictly*
reduces the arena.  The guarantees this module owes the backend are:

* two blocks whose live intervals overlap never share a byte,
* offsets are multiples of the alignment the blocks were sized with,
* placement is a pure function of the requests -- no hashing, no randomness, no
  dependence on mapping order -- so re-emitting a model reproduces the same
  ``.bss`` layout bit for bit,
* the strategy search can only shrink the arena, never grow it: the historical
  First-Fit layout is always the fallback, so existing artifacts and pinned
  offsets stay valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final, Iterable, NamedTuple, Sequence

__all__ = (
    "Layout",
    "Request",
    "best_fit_offset",
    "first_fit_offset",
    "place",
)

#: One physical arena block: ``(name, (first_op, last_op), size_bytes)``.
#: Sizes are expected to be pre-aligned by the caller.
Request = tuple[str, tuple[int, int], int]

#: A gap rule maps a block size and the merged address spans of every block it
#: may not overlap onto the offset the block should use.
_GapRule = Callable[[int, Sequence[tuple[int, int]]], int]


class Layout(NamedTuple):
    """The arena layout chosen for one set of requests."""

    offsets: dict[str, int]
    arena_bytes: int
    peak_concurrent_bytes: int
    strategy: str


class _Block(NamedTuple):
    """A placed block; ``first_op``/``last_op`` are its live interval."""

    offset: int
    size: int
    first_op: int
    last_op: int
    name: str

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True, slots=True)
class _Strategy:
    """A placement order plus the gap rule used to pick an offset."""

    label: str
    descending_size: bool
    gap_rule: _GapRule


def _occupied_spans(blocks: Iterable[_Block]) -> list[tuple[int, int]]:
    """Merges block address ranges into sorted, disjoint spans."""
    spans: list[tuple[int, int]] = []
    for start, end in sorted((block.offset, block.end) for block in blocks):
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans


def first_fit_offset(size: int, spans: Sequence[tuple[int, int]]) -> int:
    """Lowest offset that fits ``size``, reusing a gap before the arena tail."""
    offset = 0
    for start, end in spans:
        if offset + size <= start:
            return offset
        offset = max(offset, end)
    return offset


def best_fit_offset(size: int, spans: Sequence[tuple[int, int]]) -> int:
    """Tightest gap that fits ``size``; ties go to the lowest offset."""
    offset = 0
    best: tuple[int, int] | None = None
    for start, end in spans:
        slack = start - offset - size
        if slack >= 0 and (best is None or slack < best[0]):
            best = (slack, offset)
        offset = max(offset, end)
    return offset if best is None else best[1]


def _peak_concurrent_bytes(requests: Sequence[Request]) -> int:
    """Peak total size of blocks live at the same op: the RAM floor.

    This is a property of the live intervals alone, so it does not depend on
    the strategy that produced the offsets.
    """
    events: list[tuple[int, int]] = []
    for _, (first_op, last_op), size in requests:
        events.append((first_op, size))
        events.append((last_op + 1, -size))
    # For equal op ids the negative (death) events sort first, so a block that
    # dies on op ``t`` is already gone when a block starting at ``t + 1`` lands.
    events.sort()

    peak = 0
    live = 0
    for _, delta in events:
        live += delta
        peak = max(peak, live)
    return peak


def _overlaps(first_op: int, last_op: int, block: _Block) -> bool:
    """True when ``[first_op, last_op]`` shares at least one op with ``block``."""
    return block.first_op <= last_op and first_op <= block.last_op


def _layout(requests: Sequence[Request], strategy: _Strategy) -> Layout:
    """Places ``requests`` with one strategy and measures the result."""
    if strategy.descending_size:
        ordered = sorted(
            requests,
            key=lambda request: (-request[2], request[1][0], request[0]),
        )
    else:
        ordered = sorted(requests, key=lambda request: (request[1][0], request[0]))

    offsets: dict[str, int] = {}
    placed: list[_Block] = []
    arena_bytes = 0

    for name, (first_op, last_op), size in ordered:
        blockers = [block for block in placed if _overlaps(first_op, last_op, block)]
        offset = strategy.gap_rule(size, _occupied_spans(blockers))
        block = _Block(offset, size, first_op, last_op, name)

        placed.append(block)
        offsets[name] = offset
        arena_bytes = max(arena_bytes, block.end)

    return Layout(offsets, arena_bytes, _peak_concurrent_bytes(requests), strategy.label)


#: The historical layout: first-fit in insertion order (``first_op``, name).
#: Every other strategy has to beat it *strictly* to be used.
_BASELINE: Final = _Strategy("insertion-order/first-fit", False, first_fit_offset)

#: Tried in this order; the first candidate that is strictly smaller than the
#: best layout so far wins, so the outcome is a deterministic function of the
#: requests alone.
_STRATEGIES: Final = (
    _Strategy("size-descending/best-fit", True, best_fit_offset),
    _Strategy("insertion-order/best-fit", False, best_fit_offset),
    _Strategy("size-descending/first-fit", True, first_fit_offset),
)


def place(requests: Sequence[Request]) -> Layout:
    """Allocates ``requests`` in one arena; the smallest layout wins.

    ``requests`` are ``(name, (first_op, last_op), size_bytes)`` triples with
    pre-aligned sizes and unique names.  The historical First-Fit layout is
    returned unless one of ``_STRATEGIES`` strictly reduces the arena, so this
    function can only ever improve on the pre-existing offsets.
    """
    for name, (first_op, last_op), size in requests:
        if size < 0:
            raise ValueError(f"negative tensor size: {name}")
        if last_op < first_op:
            raise ValueError(f"invalid lifetime: {name}")

    best = _layout(requests, _BASELINE)
    for strategy in _STRATEGIES:
        candidate = _layout(requests, strategy)
        if candidate.arena_bytes < best.arena_bytes:
            best = candidate
    return best
