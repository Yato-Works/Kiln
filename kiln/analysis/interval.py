from __future__ import annotations

"""Interval Boundary Propagation (IBP) for Kiln IR.

Calculates static [min, max] value intervals for all tensors and intermediate
accumulators to mathematically guarantee:
  1. 32-bit signed accumulator overflow freedom (no INT32_MIN/INT32_MAX overflow).
  2. Requantization clamping / saturation behaviour (identifies whether clamping triggers).
"""

from dataclasses import dataclass
from typing import Any, Mapping

from ..ir.kir import DType, Graph, OpType
from ..ir.sem import (
    INT8_MAX,
    INT8_MIN,
    INT32_MAX,
    INT32_MIN,
    requantize,
    rounding_divide_by_pot,
    saturating_rounding_doubling_high_mul,
)
from ..ref.ref import _materialize


@dataclass(frozen=True, slots=True)
class Interval:
    """An inclusive integer interval [lo, hi]."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"invalid interval: lo ({self.lo}) > hi ({self.hi})")

    def contains(self, value: int) -> bool:
        return self.lo <= value <= self.hi

    def fits_int8(self) -> bool:
        return self.lo >= INT8_MIN and self.hi <= INT8_MAX

    def fits_int32(self) -> bool:
        return self.lo >= INT32_MIN and self.hi <= INT32_MAX


@dataclass(frozen=True, slots=True)
class NodeIntervalAudit:
    """Audit result for a single computation node."""

    node_name: str
    op_type: OpType
    # Accumulator bounds before requantization (only for CONV1D, MATMUL, ADD)
    accum_lo: int | None = None
    accum_hi: int | None = None
    accum_fits_int32: bool = True
    # Values after scaling and shifting but before int8 clamp
    preclamp_lo: int | None = None
    preclamp_hi: int | None = None
    saturates: bool = False


@dataclass(frozen=True, slots=True)
class IntervalAudit:
    """Graph-wide interval propagation audit."""

    tensor_intervals: Mapping[str, Interval]
    node_audits: Mapping[str, NodeIntervalAudit]
    int32_overflow_free: bool
    total_accum_ops: int
    saturating_ops: int
    min_accum: int | None
    max_accum: int | None


def _requantize_preclamp(accum: int, multiplier: int, shift: int) -> int:
    clamped_accum = max(INT32_MIN, min(INT32_MAX, accum))
    scaled = saturating_rounding_doubling_high_mul(clamped_accum, multiplier)
    return rounding_divide_by_pot(scaled, shift)


def _safe_requantize(accum: int, multiplier: int, shift: int) -> int:
    clamped_accum = max(INT32_MIN, min(INT32_MAX, accum))
    return requantize(clamped_accum, multiplier, shift)


def _weight_values(name: str, weights: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        raw = weights[name]
    except KeyError:
        raise KeyError(f"weight not found in weights dictionary: {name!r}") from None
    shape, flat = _materialize(raw)
    return shape, flat


def propagate_intervals(
    graph: Graph,
    weights: Mapping[str, Any],
    input_intervals: Mapping[str, Interval] | None = None,
) -> IntervalAudit:
    """Propagates inclusive value bounds across the computation graph."""
    graph.validate()
    tensor_bounds: dict[str, Interval] = {}
    node_audits: dict[str, NodeIntervalAudit] = {}

    for inp_name in graph.inputs:
        inp_desc = graph.tensors[inp_name]
        if input_intervals and inp_name in input_intervals:
            tensor_bounds[inp_name] = input_intervals[inp_name]
        elif inp_desc.dtype is DType.UINT8:
            tensor_bounds[inp_name] = Interval(0, 255)
        elif inp_desc.dtype is DType.INT8:
            tensor_bounds[inp_name] = Interval(INT8_MIN, INT8_MAX)
        else:
            tensor_bounds[inp_name] = Interval(INT32_MIN, INT32_MAX)

    for node in graph.nodes:
        out_name = node.outputs[0]
        out_desc = graph.tensors[out_name]

        if node.op_type is OpType.EMBEDDING:
            w_name = node.attrs.get("weight")
            if not isinstance(w_name, str):
                raise ValueError(f"EMBEDDING node {node.name!r} missing 'weight' attribute")
            _, flat_w = _weight_values(w_name, weights)
            if not flat_w:
                raise ValueError(f"empty embedding weights: {w_name!r}")
            tensor_bounds[out_name] = Interval(min(flat_w), max(flat_w))
            node_audits[node.name] = NodeIntervalAudit(node.name, node.op_type)

        elif node.op_type is OpType.CONV1D:
            in_name = node.inputs[0]
            in_bound = tensor_bounds[in_name]
            w_name = node.attrs.get("weight")
            if not isinstance(w_name, str):
                raise ValueError(f"CONV1D node {node.name!r} missing 'weight' attribute")
            shape_w, flat_w = _weight_values(w_name, weights)
            out_channels, in_channels, kernel = shape_w
            filter_size = in_channels * kernel
            oc_min_accs: list[int] = []
            oc_max_accs: list[int] = []

            for oc in range(out_channels):
                pos_sum = 0
                neg_sum = 0
                base = oc * filter_size
                for f in range(filter_size):
                    val = flat_w[base + f]
                    if val > 0:
                        pos_sum += val
                    else:
                        neg_sum += val
                oc_min_accs.append(in_bound.lo * pos_sum + in_bound.hi * neg_sum)
                oc_max_accs.append(in_bound.hi * pos_sum + in_bound.lo * neg_sum)

            accum_lo = min(oc_min_accs)
            accum_hi = max(oc_max_accs)
            accum_fits = (accum_lo >= INT32_MIN) and (accum_hi <= INT32_MAX)
            quant = out_desc.quant
            if quant is None:
                raise ValueError(f"CONV1D output {out_name!r} missing quantization parameters")

            preclamp_lo = _requantize_preclamp(accum_lo, quant.multiplier, quant.shift)
            preclamp_hi = _requantize_preclamp(accum_hi, quant.multiplier, quant.shift)
            saturates = (preclamp_lo < INT8_MIN) or (preclamp_hi > INT8_MAX)
            out_lo = _safe_requantize(accum_lo, quant.multiplier, quant.shift)
            out_hi = _safe_requantize(accum_hi, quant.multiplier, quant.shift)

            tensor_bounds[out_name] = Interval(out_lo, out_hi)
            node_audits[node.name] = NodeIntervalAudit(
                node.name, node.op_type, accum_lo, accum_hi, accum_fits,
                preclamp_lo, preclamp_hi, saturates,
            )

        elif node.op_type is OpType.MATMUL:
            in_name = node.inputs[0]
            in_bound = tensor_bounds[in_name]
            w_name = node.attrs.get("weight")
            if not isinstance(w_name, str):
                raise ValueError(f"MATMUL node {node.name!r} missing 'weight' attribute")
            shape_w, flat_w = _weight_values(w_name, weights)
            out_features, in_features = shape_w
            row_min_accs: list[int] = []
            row_max_accs: list[int] = []

            for o in range(out_features):
                pos_sum = 0
                neg_sum = 0
                base = o * in_features
                for f in range(in_features):
                    val = flat_w[base + f]
                    if val > 0:
                        pos_sum += val
                    else:
                        neg_sum += val
                row_min_accs.append(in_bound.lo * pos_sum + in_bound.hi * neg_sum)
                row_max_accs.append(in_bound.hi * pos_sum + in_bound.lo * neg_sum)

            accum_lo = min(row_min_accs)
            accum_hi = max(row_max_accs)
            accum_fits = (accum_lo >= INT32_MIN) and (accum_hi <= INT32_MAX)
            quant = out_desc.quant
            if quant is None:
                raise ValueError(f"MATMUL output {out_name!r} missing quantization parameters")

            preclamp_lo = _requantize_preclamp(accum_lo, quant.multiplier, quant.shift)
            preclamp_hi = _requantize_preclamp(accum_hi, quant.multiplier, quant.shift)
            saturates = (preclamp_lo < INT8_MIN) or (preclamp_hi > INT8_MAX)
            out_lo = _safe_requantize(accum_lo, quant.multiplier, quant.shift)
            out_hi = _safe_requantize(accum_hi, quant.multiplier, quant.shift)

            tensor_bounds[out_name] = Interval(out_lo, out_hi)
            node_audits[node.name] = NodeIntervalAudit(
                node.name, node.op_type, accum_lo, accum_hi, accum_fits,
                preclamp_lo, preclamp_hi, saturates,
            )

        elif node.op_type is OpType.ADD:
            left_bound = tensor_bounds[node.inputs[0]]
            right_bound = tensor_bounds[node.inputs[1]]
            accum_lo = left_bound.lo + right_bound.lo
            accum_hi = left_bound.hi + right_bound.hi
            accum_fits = (accum_lo >= INT32_MIN) and (accum_hi <= INT32_MAX)
            quant = out_desc.quant
            if quant is not None:
                preclamp_lo = _requantize_preclamp(accum_lo, quant.multiplier, quant.shift)
                preclamp_hi = _requantize_preclamp(accum_hi, quant.multiplier, quant.shift)
                saturates = (preclamp_lo < INT8_MIN) or (preclamp_hi > INT8_MAX)
                out_lo = _safe_requantize(accum_lo, quant.multiplier, quant.shift)
                out_hi = _safe_requantize(accum_hi, quant.multiplier, quant.shift)
            else:
                preclamp_lo = accum_lo
                preclamp_hi = accum_hi
                saturates = (accum_lo < INT8_MIN) or (accum_hi > INT8_MAX)
                out_lo = max(INT8_MIN, min(INT8_MAX, accum_lo))
                out_hi = max(INT8_MIN, min(INT8_MAX, accum_hi))

            tensor_bounds[out_name] = Interval(out_lo, out_hi)
            node_audits[node.name] = NodeIntervalAudit(
                node.name, node.op_type, accum_lo, accum_hi, accum_fits,
                preclamp_lo, preclamp_hi, saturates,
            )

        elif node.op_type is OpType.LUT:
            in_bound = tensor_bounds[node.inputs[0]]
            table = node.attrs.get("table")
            if not isinstance(table, (list, tuple)) or len(table) != 256:
                raise ValueError(f"LUT node {node.name!r} missing 256-entry 'table' attribute")
            idx_start = in_bound.lo - INT8_MIN
            idx_end = in_bound.hi - INT8_MIN + 1
            sliced = table[idx_start:idx_end]
            out_lo, out_hi = (min(sliced), max(sliced)) if sliced else (0, 0)
            tensor_bounds[out_name] = Interval(out_lo, out_hi)
            node_audits[node.name] = NodeIntervalAudit(node.name, node.op_type)

        elif node.op_type is OpType.MAXPOOL1D:
            tensor_bounds[out_name] = tensor_bounds[node.inputs[0]]
            node_audits[node.name] = NodeIntervalAudit(node.name, node.op_type)

        elif node.op_type is OpType.ARGMAX:
            in_desc = graph.tensors[node.inputs[0]]
            tensor_bounds[out_name] = Interval(0, max(0, in_desc.num_elements - 1))
            node_audits[node.name] = NodeIntervalAudit(node.name, node.op_type)

        else:
            raise NotImplementedError(f"Unsupported op type for interval propagation: {node.op_type}")

    all_acc_fits = all(na.accum_fits_int32 for na in node_audits.values())
    accum_audits = [na for na in node_audits.values() if na.accum_lo is not None and na.accum_hi is not None]
    min_accum = min((na.accum_lo for na in accum_audits), default=None)
    max_accum = max((na.accum_hi for na in accum_audits), default=None)
    saturating_ops = sum(1 for na in node_audits.values() if na.saturates)

    return IntervalAudit(
        tensor_intervals=tensor_bounds,
        node_audits=node_audits,
        int32_overflow_free=all_acc_fits,
        total_accum_ops=len(accum_audits),
        saturating_ops=saturating_ops,
        min_accum=min_accum,
        max_accum=max_accum,
    )

