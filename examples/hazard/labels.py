"""Torch-free public labels for the Hazard example."""

from __future__ import annotations

from typing import Final

HAZARD_CLASSES: Final[tuple[str, ...]] = (
    "benign",
    "dynamic_exec",
    "deserialization",
    "shell_injection",
    "tempfile",
    "unsafe_validation",
)
