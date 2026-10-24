"""Torch-free public labels for the keyword-spotting example."""

from __future__ import annotations

from typing import Final

KWS_CLASSES: Final[tuple[str, ...]] = (
    "silence",
    "unknown",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)
