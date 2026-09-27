"""Deterministic enumeration of a declarative search grid (Phase 3, M3.2)."""

from __future__ import annotations

import itertools
from typing import Mapping

__all__ = ("enumerate_grid",)


def enumerate_grid(params: Mapping[str, tuple[int, ...]]) -> list[dict[str, int]]:
    """Expands the Cartesian product of ``params`` in a stable order.

    Parameter names are sorted first, so the candidate order depends on the
    grid's *content* only -- not on YAML mapping order, dict iteration order or
    the platform.  Kiln's determinism rule (same input, same bytes) applies to
    the search result as much as to the emitted C.
    """
    names = sorted(params)
    return [
        dict(zip(names, combination))
        for combination in itertools.product(*(params[name] for name in names))
    ]
