from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from kiln.frontend.pytorch import UnsupportedGraphError, export_to_kir


class Unsupported(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


def test_unsupported_fx_module_is_a_strict_structured_error() -> None:
    with pytest.raises(UnsupportedGraphError, match="KILN-FX-001"):
        export_to_kir(Unsupported(), torch.tensor([1, 2], dtype=torch.int8))
