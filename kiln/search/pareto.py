"""Pareto fronts over mixed-direction objectives (Phase 3, M3.2)."""

from __future__ import annotations

from typing import Mapping, Sequence

__all__ = ("dominates", "pareto_front")


def dominates(
    a: Mapping[str, float],
    b: Mapping[str, float],
    minimize: Sequence[str],
    maximize: Sequence[str],
) -> bool:
    """True when ``a`` is at least as good as ``b`` everywhere and better once."""
    strictly_better = False
    for key in minimize:
        if a[key] > b[key]:
            return False
        strictly_better = strictly_better or a[key] < b[key]
    for key in maximize:
        if a[key] < b[key]:
            return False
        strictly_better = strictly_better or a[key] > b[key]
    return strictly_better


def pareto_front(
    points: Sequence[Mapping[str, float]],
    minimize: Sequence[str] = (),
    maximize: Sequence[str] = (),
) -> list[int]:
    """Returns the indices of the non-dominated points, in a stable order.

    ``minimize`` keys are lower-is-better, ``maximize`` keys upper-is-better.
    Duplicate points never dominate each other, so twins stay on the front --
    two different shapes can legitimately audit to the same footprint.
    """
    if not points:
        return []

    for key in (*minimize, *maximize):
        for point in points:
            if key not in point:
                raise KeyError(f"objective {key!r} is missing from a search point")

    front = [
        index
        for index, point in enumerate(points)
        if not any(
            dominates(other, point, minimize, maximize)
            for position, other in enumerate(points)
            if position != index
        )
    ]
    front.sort(
        key=lambda index: (
            *(points[index][key] for key in minimize),
            *(-points[index][key] for key in maximize),
            tuple(sorted(points[index].items())),
        )
    )
    return front
