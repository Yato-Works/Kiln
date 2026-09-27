"""Budget-gated architecture search over a declarative spec (Phase 3, M3.2).

Kiln's footprint is a *compile-time* quantity: ``audit_graph`` derives the
arena (``.bss``) bytes, the constant ROM and the MAC count exactly from the
graph's tensor shapes and weight extents -- no training, no floating point, no
run on the target.  Two candidates that differ only in their shapes therefore
have their footprints known the instant they are built, which is what makes a
full grid search over the declared parameters both cheap and fully
deterministic (the same grid always yields the same report bytes).

Objectives.  Every candidate minimises arena RAM and constant ROM; MACs is
*maximised*, because inside one frozen task the MAC count is the compile-time
knob for how much model fits (the audit's ``estimated_worst_cycles`` grows with
it, so the front doubles as a latency/capacity skyline).  The result is the
"most model that fits the silicon" frontier -- the RAM/ROM/capacity Pareto
solutions the Phase 3 roadmap asks for.  Setting ``budget.macs_max`` turns the
MAC axis into a hard constraint instead, which recovers the pure RAM/ROM front.

A candidate that fails to build or audit (an invalid window, a kernel longer
than its input, ...) is reported as an error row and never aborts the grid: the
search exists to map the boundary, and the boundary is exactly where candidates
stop being legal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..budget.report import audit_graph
from ..spec import KilnSpec, resolve_model_graph
from .pareto import pareto_front
from .space import enumerate_grid

__all__ = ("MAXIMISE", "MINIMISE", "SearchOutcome", "SearchRow", "run_search")

#: Objectives every candidate tries to make as small as possible.
MINIMISE: tuple[str, ...] = ("arena_ram_bytes", "total_rom_bytes")

#: Objectives every candidate tries to make as large as possible (capacity).
MAXIMISE: tuple[str, ...] = ("total_macs",)


@dataclass(frozen=True, slots=True)
class SearchRow:
    """One audited grid point (or one failed candidate, with ``error`` set)."""
    params: Mapping[str, int]
    arena_ram_bytes: int
    total_rom_bytes: int
    total_macs: int
    estimated_worst_cycles: int
    feasible: bool
    violations: tuple[str, ...] = ()
    error: str | None = None

    def objectives(self) -> dict[str, int]:
        return {key: getattr(self, key) for key in (*MINIMISE, *MAXIMISE)}

    @property
    def capacity_per_arena_byte(self) -> float:
        """MACs bought per arena byte -- how efficiently RAM turns into capacity.

        A candidate can be Pareto-optimal *and* wasteful: on the hazard ladder
        this metric is ``labels * dim / (dim + labels)``, so at a fixed MAC count
        the widest embedding is always the RAM-cheapest one.  Failed rows have no
        footprint and report 0.0.
        """
        if self.arena_ram_bytes <= 0:
            return 0.0
        return self.total_macs / self.arena_ram_bytes


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """The completed grid: every row, plus which rows sit on the front."""
    rows: tuple[SearchRow, ...]
    feasible_count: int
    pareto: frozenset[int]

    @property
    def error_count(self) -> int:
        return sum(1 for row in self.rows if row.error is not None)

    def front_rows(self) -> tuple[SearchRow, ...]:
        """The Pareto rows, already ordered by the front's stable sort."""
        return tuple(self.rows[index] for index in sorted(self.pareto))


def _violations(row: Mapping[str, int], spec: KilnSpec) -> tuple[str, ...]:
    budget = spec.budget
    checks = [
        ("arena", row["arena_ram_bytes"], budget.ram_arena_max_bytes),
        ("rom", row["total_rom_bytes"], budget.rom_total_max_bytes),
    ]
    if budget.macs_max is not None:
        checks.append(("macs", row["total_macs"], budget.macs_max))
    return tuple(
        f"{label} {used:,} > {limit:,}" for label, used, limit in checks if used > limit
    )


def run_search(spec: KilnSpec, *, alignment: int = 4) -> SearchOutcome:
    """Enumerates ``spec.search`` and audits every candidate against the budget."""
    if spec.search is None:
        raise ValueError(f"spec '{spec.name}' has no 'search:' block")

    rows: list[SearchRow] = []
    for params in enumerate_grid(spec.search.params):
        try:
            loaded = resolve_model_graph(
                spec,
                entry_point=spec.search.entry_point,
                param_overrides=params,
            )
            report = audit_graph(loaded.graph, loaded.weights, alignment=alignment)
        except Exception as exc:  # one illegal candidate must not kill the grid
            rows.append(SearchRow(
                params=params,
                arena_ram_bytes=0,
                total_rom_bytes=0,
                total_macs=0,
                estimated_worst_cycles=0,
                feasible=False,
                error=f"{type(exc).__name__}: {exc}",
            ))
            continue

        measured = {
            "arena_ram_bytes": report.arena_ram_bytes,
            "total_rom_bytes": report.total_rom_bytes,
            "total_macs": report.total_macs,
        }
        violations = _violations(measured, spec)
        rows.append(SearchRow(
            params=params,
            arena_ram_bytes=report.arena_ram_bytes,
            total_rom_bytes=report.total_rom_bytes,
            total_macs=report.total_macs,
            estimated_worst_cycles=report.estimated_worst_cycles,
            feasible=not violations,
            violations=violations,
        ))

    feasible = [index for index, row in enumerate(rows) if row.feasible]
    front = pareto_front([rows[index].objectives() for index in feasible], MINIMISE, MAXIMISE)
    return SearchOutcome(
        rows=tuple(rows),
        feasible_count=len(feasible),
        pareto=frozenset(feasible[position] for position in front),
    )
