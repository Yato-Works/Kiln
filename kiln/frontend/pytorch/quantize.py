"""Deterministic, auditable integer quantization helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True, slots=True)
class QuantizationPolicy:
    """Explicit policy for converting real tensors to the Kiln int8 domain."""

    symmetric: bool = True
    qmin: int = -127
    qmax: int = 127
    rounding: str = "torch.round_even"
    saturation: str = "clamp"

    def __post_init__(self) -> None:
        if not self.symmetric:
            raise ValueError("only symmetric quantization is supported in KIR v1")
        if (self.qmin, self.qmax) != (-127, 127):
            raise ValueError("KIR v1 requires the symmetric int8 range [-127, 127]")
        if self.rounding != "torch.round_even":
            raise ValueError("KIR v1 requires torch.round_even rounding")
        if self.saturation != "clamp":
            raise ValueError("KIR v1 requires clamp saturation")


@dataclass(frozen=True, slots=True)
class CalibrationStats:
    """Observed range and chosen scale for one calibration pass."""

    min_value: float
    max_value: float
    max_abs: float
    scale: float
    samples: int
    policy: QuantizationPolicy


def compute_multiplier_and_shift(scale: float) -> tuple[int, int]:
    """Encode a positive scale as a Q31 multiplier and power-of-two shift."""
    if not math.isfinite(scale):
        raise ValueError(f"quantization scale must be finite, got {scale}")
    if scale <= 0.0:
        raise ValueError(f"quantization scale must be strictly positive, got {scale}")
    if scale >= 1.0:
        raise ValueError(
            f"effective scale {scale} >= 1.0 is unsupported "
            "(requires shift < 0, but Kiln requires 0 <= shift <= 31)"
        )

    q, exp = math.frexp(scale)
    multiplier = round(q * (1 << 31))
    multiplier = max(1 << 30, min((1 << 31) - 1, multiplier))
    shift = -exp
    if shift < 0 or shift > 31:
        raise ValueError(f"effective scale {scale} produces unsupported shift {shift}")
    return int(multiplier), int(shift)


def calibration_stats(
    tensors: Iterable[torch.Tensor], *, policy: QuantizationPolicy | None = None
) -> CalibrationStats:
    """Measure a deterministic range over calibration tensors."""
    selected = policy or QuantizationPolicy()
    values = [tensor.detach().cpu().to(torch.float32) for tensor in tensors]
    if not values:
        raise ValueError("calibration requires at least one tensor")
    minimum = min(float(torch.min(tensor).item()) for tensor in values)
    maximum = max(float(torch.max(tensor).item()) for tensor in values)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("calibration values must be finite")
    max_abs = max(abs(minimum), abs(maximum))
    scale = 1.0 if max_abs < 1e-7 else max_abs / selected.qmax
    return CalibrationStats(minimum, maximum, max_abs, scale, len(values), selected)


def quantize_tensor(
    tensor: torch.Tensor,
    *,
    policy: QuantizationPolicy | None = None,
    calibration: CalibrationStats | None = None,
) -> tuple[Any, float]:
    """Quantize a tensor with explicit rounding and saturation semantics."""
    selected = policy or (calibration.policy if calibration else QuantizationPolicy())
    stats = calibration or calibration_stats((tensor,), policy=selected)
    if calibration is not None and calibration.policy != selected:
        raise ValueError("calibration policy does not match quantization policy")
    t = tensor.detach().cpu().to(torch.float32)
    q = torch.clamp(torch.round(t / stats.scale), selected.qmin, selected.qmax).to(torch.int8)
    return q.tolist(), stats.scale
