"""Markdown / CSV rendering of a completed search (Phase 3, M3.2)."""

from __future__ import annotations

import csv
import io

from ..spec import KilnSpec
from .engine import MAXIMISE, MINIMISE, SearchOutcome, SearchRow

__all__ = ("format_search_report", "search_results_csv")


def _verdict(outcome: SearchOutcome, index: int, row: SearchRow) -> str:
    if row.error is not None:
        return f"error: {row.error}"
    if not row.feasible:
        return "over budget: " + "; ".join(row.violations)
    return "fits (**Pareto**)" if index in outcome.pareto else "fits"


def _escaped(text: str) -> str:
    return text.replace("|", "/")


def capacity_classes(outcome: SearchOutcome) -> tuple[SearchRow, ...]:
    """The RAM-cheapest feasible shape at every distinct MAC count.

    Reads the front as an answer to "I need this much capacity -- which shape
    costs the least RAM?", which is the question a 2 KB MCU actually asks.
    Ties are broken by ROM, then by the grid order (stable).
    """
    best: dict[int, SearchRow] = {}
    for row in outcome.rows:
        if not row.feasible:
            continue
        current = best.get(row.total_macs)
        if (
            current is None
            or row.arena_ram_bytes < current.arena_ram_bytes
            or (
                row.arena_ram_bytes == current.arena_ram_bytes
                and row.total_rom_bytes < current.total_rom_bytes
            )
        ):
            best[row.total_macs] = row
    return tuple(best[key] for key in sorted(best))


def format_search_report(spec: KilnSpec, outcome: SearchOutcome) -> str:
    """Renders the search as a Markdown artifact next to the build outputs."""
    assert spec.search is not None, "format_search_report requires a search spec"

    grid = ", ".join(
        f"`{name}` = {{{', '.join(str(value) for value in values)}}}"
        for name, values in sorted(spec.search.params.items())
    )
    budget = [
        f"arena RAM <= {spec.budget.ram_arena_max_bytes:,} B",
        f"ROM <= {spec.budget.rom_total_max_bytes:,} B",
    ]
    if spec.budget.macs_max is not None:
        budget.append(f"MACs <= {spec.budget.macs_max:,}")

    names = sorted(spec.search.params)
    lines = [
        f"# Kiln architecture search: {spec.name}",
        "",
        "Compile-time footprint exploration (`kiln search`): arena RAM and constant",
        f"ROM are minimised, {' and '.join(MAXIMISE)} (the capacity / latency knob) is",
        "maximised, and only budget-feasible candidates can reach the front.",
        "",
        f"- grid: {grid}",
        f"- budget: {', '.join(budget)}",
        f"- candidates: {len(outcome.rows)} "
        f"({outcome.feasible_count} feasible, {outcome.error_count} errored)",
        f"- Pareto front: {len(outcome.pareto)} candidate(s)",
        "",
        "## Candidates",
        "",
        "| # | "
        + " | ".join(names)
        + " | Arena RAM (B) | ROM (B) | MACs | Cycles | MACs/B | Verdict |",
        "| ---: | "
        + " | ".join("---:" for _ in names)
        + " | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(outcome.rows):
        cells = [str(row.params[name]) for name in names]
        if row.error is not None:
            cells += ["-", "-", "-", "-", "-"]
        else:
            cells += [
                f"{row.arena_ram_bytes:,}",
                f"{row.total_rom_bytes:,}",
                f"{row.total_macs:,}",
                f"{row.estimated_worst_cycles:,}",
                f"{row.capacity_per_arena_byte:.2f}",
            ]
        lines.append(
            f"| {index} | " + " | ".join(cells) + f" | {_escaped(_verdict(outcome, index, row))} |"
        )

    lines += [
        "",
        "## Pareto front",
        "",
    ]
    front = outcome.front_rows()
    if not front:
        lines.append("No candidate fits the declared budget.")
    else:
        lines.append("| " + " | ".join(names) + " | Arena RAM (B) | ROM (B) | MACs |")
        lines.append("| " + " | ".join("---:" for _ in names) + " | ---: | ---: | ---: |")
        for row in front:
            cells = [str(row.params[name]) for name in names]
            cells += [f"{row.arena_ram_bytes:,}", f"{row.total_rom_bytes:,}", f"{row.total_macs:,}"]
            lines.append("| " + " | ".join(cells) + " |")

    classes = capacity_classes(outcome)
    lines += [
        "",
        "## Capacity classes",
        "",
        "The RAM-cheapest feasible shape at each reachable MAC count (at a fixed",
        "capacity, the widest embedding is the RAM-cheapest one):",
        "",
    ]
    if not classes:
        lines.append("No feasible capacity class.")
    else:
        lines.append("| MACs | " + " | ".join(names) + " | Arena RAM (B) | ROM (B) | MACs/B |")
        lines.append("| ---: | " + " | ".join("---:" for _ in names) + " | ---: | ---: | ---: |")
        for row in classes:
            cells = [f"{row.total_macs:,}"] + [str(row.params[name]) for name in names]
            cells += [
                f"{row.arena_ram_bytes:,}",
                f"{row.total_rom_bytes:,}",
                f"{row.capacity_per_arena_byte:.2f}",
            ]
            lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "Objectives: "
        + ", ".join(f"minimise `{key}`" for key in MINIMISE)
        + ", "
        + ", ".join(f"maximise `{key}`" for key in MAXIMISE)
        + ".",
        "",
    ]
    return "\n".join(lines)


def search_results_csv(outcome: SearchOutcome, names: list[str]) -> str:
    """Renders the raw rows as CSV (one line per candidate, audit columns exact)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            *names,
            "arena_ram_bytes",
            "total_rom_bytes",
            "total_macs",
            "estimated_worst_cycles",
            "capacity_per_arena_byte",
            "feasible",
            "pareto",
            "error",
        ]
    )
    for index, row in enumerate(outcome.rows):
        writer.writerow(
            [
                *(row.params[name] for name in names),
                row.arena_ram_bytes,
                row.total_rom_bytes,
                row.total_macs,
                row.estimated_worst_cycles,
                f"{row.capacity_per_arena_byte:.4f}",
                int(row.feasible),
                int(index in outcome.pareto),
                row.error or "",
            ]
        )
    return buffer.getvalue()
