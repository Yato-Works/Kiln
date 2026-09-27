"""Architecture search over the compile-time footprint (Phase 3)."""

from __future__ import annotations

from .engine import MAXIMISE, MINIMISE, SearchOutcome, SearchRow, run_search
from .pareto import pareto_front
from .report import capacity_classes, format_search_report, search_results_csv
from .space import enumerate_grid

__all__ = (
    "MAXIMISE",
    "MINIMISE",
    "SearchOutcome",
    "SearchRow",
    "capacity_classes",
    "enumerate_grid",
    "format_search_report",
    "pareto_front",
    "run_search",
    "search_results_csv",
)
