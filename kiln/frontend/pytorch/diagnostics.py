"""Structured diagnostics emitted by the PyTorch frontend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoweringDiagnostic:
    code: str
    message: str
    node: str | None = None
    source: str | None = None
    line: int | None = None

    def format(self) -> str:
        location = self.source or "model"
        if self.line is not None:
            location += f":{self.line}"
        if self.node:
            location += f" [{self.node}]"
        return f"{location}: {self.code}: {self.message}"


class UnsupportedGraphError(ValueError):
    """A model contains a graph construct outside the supported KIR subset."""

    def __init__(self, diagnostic: LoweringDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(diagnostic.format())
