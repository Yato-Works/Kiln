"""Text rendering of a linked spec build (``kiln link``, Phase 4).

The report's job is to put the *static* audit next to the *observed* artifact:
the arena the compiler planned against the ``.bss`` the linker actually reserved,
the audited weight bytes against the read-only data that is actually present, and
the zero-malloc guarantee against the undefined symbols of the model object.
The last section is what a certificate cannot state on its own -- the linked
binary ran and produced labels.
"""

from __future__ import annotations

from ..spec import KilnSpec
from .engine import LinkOutcome
from .harness import VECTOR_NAMES

__all__ = ("budget_violations", "format_link_report")

_RULE = "=" * 73
_THIN = "-" * 73


def budget_violations(outcome: LinkOutcome, spec: KilnSpec) -> tuple[str, ...]:
    """The spec limits the audited footprint exceeds (empty when it fits)."""
    checks = [
        ("arena RAM", outcome.report.arena_ram_bytes, spec.budget.ram_arena_max_bytes),
        ("total ROM", outcome.report.total_rom_bytes, spec.budget.rom_total_max_bytes),
    ]
    if spec.budget.macs_max is not None:
        checks.append(("MACs", outcome.report.total_macs, spec.budget.macs_max))
    return tuple(f"{label} {used:,} > {limit:,}" for label, used, limit in checks if used > limit)


def _arena_note(outcome: LinkOutcome) -> str:
    bss = outcome.bss_bytes
    arena = outcome.report.arena_ram_bytes
    if bss is None:
        return "unavailable (no objdump)"
    if bss == arena:
        return "exact match with the audited arena"
    if bss > arena:
        return f"audited arena + {bss - arena:,} B of other zero-init data"
    return "UNDER the audited arena -- investigate"


def _symbols_note(outcome: LinkOutcome) -> str:
    if not outcome.symbols_inspected:
        return "unavailable (no nm/objdump)"
    if outcome.malloc_symbols:
        return "REFERENCES " + ", ".join(sorted(outcome.malloc_symbols))
    return "no allocation entry point referenced (zero-malloc observed)"


def _budget_lines(outcome: LinkOutcome, spec: KilnSpec) -> list[str]:
    limits = [
        ("Arena RAM", outcome.report.arena_ram_bytes, spec.budget.ram_arena_max_bytes),
        ("Constant ROM", outcome.report.total_rom_bytes, spec.budget.rom_total_max_bytes),
    ]
    if spec.budget.macs_max is not None:
        limits.append(("MACs", outcome.report.total_macs, spec.budget.macs_max))
    lines = [
        f"    {label:<18}: {used:>9,} <= {limit:>8,}   {'OK' if used <= limit else 'EXCEEDED'}"
        for label, used, limit in limits
    ]
    if spec.budget.macs_max is None:
        lines.append(f"    {'MACs':<18}: {outcome.report.total_macs:>9,} (no limit declared)")
    lines.append(
        f"    {'Worst-case cycles':<18}: {outcome.report.estimated_worst_cycles:>9,} (upper bound)"
    )
    return lines


def _harness_line(outcome: LinkOutcome) -> str:
    if not outcome.predictions:
        return "not run (pass --run to execute the linked binary)"
    names = VECTOR_NAMES[: len(outcome.predictions)]
    pairs = ", ".join(f"{name}={label}" for name, label in zip(names, outcome.predictions))
    return pairs


def format_link_report(outcome: LinkOutcome, spec: KilnSpec) -> str:
    """Renders the link report: what was built, what it measures, what it ran."""
    lines = [
        _RULE,
        f"                  Kiln Link Report: {outcome.spec_name}",
        _RULE,
        f"  Toolchain            : {outcome.toolchain}",
        f"  Target profile       : {outcome.target_profile} ({outcome.execution_mode})",
        f"  Translation units    : {', '.join(path.name for path in outcome.sources)}",
        f"  Linked executable    : {outcome.executable}",
        _THIN,
        "  Static budget (compile-time audit)",
        *_budget_lines(outcome, spec),
        _THIN,
        "  Linked image",
    ]
    text = outcome.text_bytes
    rodata = outcome.rodata_bytes
    bss = outcome.bss_bytes
    if text is None:
        lines.append("    Sections            : unavailable (no objdump)")
    else:
        lines.append(f"    .text (code)        : {text:>9,} bytes")
        lines.append(
            f"    read-only data      : {rodata:>9,} bytes (.rodata/.rdata; weights + frontend)"
        )
        lines.append(f"    .bss (zero-init)    : {bss:>9,} bytes -> {_arena_note(outcome)}")
    lines.append(f"    Allocation refs     : {_symbols_note(outcome)}")
    lines += [
        _THIN,
        f"  Harness replay ({len(outcome.vectors)} vectors)",
        f"    {_harness_line(outcome)}",
        _RULE,
    ]
    return "\n".join(lines)
