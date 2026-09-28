"""Spec-driven artifact emission shared by ``kiln build`` and ``kiln link``.

Both commands start from the same three files, so the emission lives here
instead of in the CLI: the translation unit, the budget certificate and the C
frontend the spec declares.  Warning policy stays with the caller -- a missing
frontend file is *returned* rather than printed, because ``kiln build`` warns and
continues while ``kiln link`` has to decide what it can still compile.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ..backend.c99.emit_c import write_c
from ..budget.report import BudgetReport, audit_graph, format_budget_certificate
from ..ir.sem import INT32_MAX, INT32_MIN
from .loader import LoadedModel
from .schema import KilnSpec

__all__ = ("EmittedArtifacts", "SafetyViolationError", "emit_spec_artifacts")


class SafetyViolationError(RuntimeError):
    """The audited model is unsafe to emit: an accumulator can overflow int32.

    Raised *before* any artifact is written, so a refused build leaves no
    ``model.c`` behind for someone to flash by accident.  The audit report
    rides along so the caller can print the certificate that explains why.
    """

    def __init__(self, message: str, report: BudgetReport) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class EmittedArtifacts:
    """What ``emit_spec_artifacts`` wrote into one output directory."""
    model_c: Path
    report: BudgetReport
    frontend_sources: tuple[Path, ...]
    frontend_headers: tuple[Path, ...]
    missing_frontend: tuple[Path, ...]

    @property
    def certificate(self) -> str:
        return format_budget_certificate(self.report)


def emit_spec_artifacts(
    spec: KilnSpec,
    loaded: LoadedModel,
    out_dir: Path,
    *,
    alignment: int = 4,
    runtime_header: str | None = "kiln_runtime.h",
) -> EmittedArtifacts:
    """Writes ``model.c``, the certificate and the declared frontend into ``out_dir``.

    The accumulator safety gate runs first: an int32-overflowing graph or a graph
    that cannot be interval-audited raises :class:`SafetyViolationError` and
    nothing is written at all.
    """
    report = audit_graph(loaded.graph, loaded.weights, alignment=alignment)

    if not report.int32_overflow_free:
        if report.interval_audit is not None:
            audit = report.interval_audit
            raise SafetyViolationError(
                f"int32 accumulator overflow detected: accumulator range "
                f"[{audit.min_accum}, {audit.max_accum}] leaves the int32 domain "
                f"[{INT32_MIN}, {INT32_MAX}]",
                report,
            )
        else:
            raise SafetyViolationError(
                "interval boundary propagation could not run on this graph; "
                "refusing to emit C without accumulator safety proof",
                report,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    model_c = out_dir / "model.c"
    write_c(
        model_c,
        loaded.graph,
        loaded.weights,
        alignment=alignment,
        runtime_header=runtime_header,
    )
    (out_dir / "budget_certificate.txt").write_text(
        format_budget_certificate(report) + "\n", encoding="utf-8"
    )

    sources: list[Path] = []
    headers: list[Path] = []
    missing: list[Path] = []
    if spec.frontend is not None:
        declared = (
            *((path, sources) for path in spec.frontend.source_files),
            *((path, headers) for path in spec.frontend.header_files),
        )
        for relative, bucket in declared:
            source = spec.resolve_path(relative)
            if source.is_file():
                destination = out_dir / source.name
                shutil.copy2(source, destination)
                bucket.append(destination)
            else:
                missing.append(source)

    return EmittedArtifacts(
        model_c=model_c,
        report=report,
        frontend_sources=tuple(sources),
        frontend_headers=tuple(headers),
        missing_frontend=tuple(missing),
    )
