"""Phase 4's link step: compile, link, measure and run a spec's C artifacts."""

from __future__ import annotations

from .engine import LinkOutcome, link_spec
from .harness import (
    VECTOR_NAMES,
    harness_vectors,
    parse_predictions,
    render_harness,
    to_input_domain,
)
from .report import budget_violations, format_execution_certificate, format_link_report
from .runner import RunnerConfig, RunnerError, RuntimeMetrics, parse_runtime_metrics
from .toolchain import (
    MALLOC_SYMBOLS,
    STRICT_FLAGS,
    Toolchain,
    ToolchainError,
    detect_toolchain,
    inspect_sections,
    undefined_symbols,
)

__all__ = (
    "MALLOC_SYMBOLS",
    "STRICT_FLAGS",
    "LinkOutcome",
    "Toolchain",
    "ToolchainError",
    "VECTOR_NAMES",
    "budget_violations",
    "detect_toolchain",
    "format_link_report",
    "format_execution_certificate",
    "RunnerConfig",
    "RunnerError",
    "RuntimeMetrics",
    "parse_runtime_metrics",
    "harness_vectors",
    "inspect_sections",
    "link_spec",
    "parse_predictions",
    "render_harness",
    "to_input_domain",
    "undefined_symbols",
)
