from __future__ import annotations

import pytest
import torch

from kiln.frontend.pytorch import quantize_tensor
from kiln.frontend.pytorch.quantize import QuantizationPolicy, calibration_stats


def test_calibration_records_range_and_policy() -> None:
    stats = calibration_stats([torch.tensor([-2.0, 1.0]), torch.tensor([3.0])])
    assert stats.min_value == -2.0
    assert stats.max_value == 3.0
    assert stats.max_abs == 3.0
    assert stats.scale == pytest.approx(3.0 / 127.0)
    assert stats.samples == 2


def test_quantization_policy_makes_rounding_and_saturation_explicit() -> None:
    policy = QuantizationPolicy()
    values, scale = quantize_tensor(torch.tensor([-10.0, 0.0, 10.0]), policy=policy)
    assert scale == pytest.approx(10.0 / 127.0)
    assert values == [-127, 0, 127]


def test_calibration_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        calibration_stats([torch.tensor([float("nan")])])
