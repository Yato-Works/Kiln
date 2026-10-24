"""Deterministic execution of a serialized KIR v1 bundle.

The VM is deliberately a separate target from the C99 emitter and from the
independent reference interpreter.  It consumes only the frozen KIR bundle,
executes the seven KIR operations, and reports deterministic resource metrics.
It is a host virtual target: its cycle count is a documented instruction-cost
model, not a claim about a particular CPU's clock cycles.
"""

from __future__ import annotations

import json
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Callable

from ..budget.report import audit_graph
from ..ir.kir import DType, Graph, Node, TensorDesc
from ..ir.memory import plan_graph_memory
from ..ir.sem import INT8_MAX, INT8_MIN, INT32_MAX, INT32_MIN, lut_lookup, requantize
from ..ir.serialization import loads_bundle

__all__ = (
    "VMError",
    "VMMetrics",
    "VMResult",
    "VMTraceEvent",
    "VirtualMachine",
    "trace_from_dict",
    "trace_to_dict",
)


_VOCAB_SIZE: Final = 256
_WEIGHT_ATTR: Final = "weight"


class VMError(RuntimeError):
    """A malformed bundle, invalid input, or refused unsafe execution."""


@dataclass(frozen=True, slots=True)
class VMTraceEvent:
    """One deterministic KIR instruction event."""

    node: str
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    cycles: int
    output_count: int
    output_min: int | None
    output_max: int | None
    output_digest: str


@dataclass(frozen=True, slots=True)
class VMMetrics:
    """Resource and execution evidence returned by one VM invocation."""

    cycles: int
    static_cycle_upper_bound: int
    instructions: int
    arena_bytes: int
    peak_concurrent_bytes: int
    rom_bytes: int
    allocation_bytes: int
    int32_overflow_free: bool


@dataclass(frozen=True, slots=True)
class VMResult:
    """Outputs, metrics, and optional instruction trace for one invocation."""

    outputs: dict[str, list[int]]
    metrics: VMMetrics
    trace: tuple[VMTraceEvent, ...]


def trace_to_dict(trace: Sequence[VMTraceEvent]) -> list[dict[str, Any]]:
    """Serialize a trace into a stable, JSON-compatible representation."""
    return [
        {
            "node": event.node,
            "op": event.op,
            "inputs": list(event.inputs),
            "outputs": list(event.outputs),
            "cycles": event.cycles,
            "output_count": event.output_count,
            "output_min": event.output_min,
            "output_max": event.output_max,
            "output_digest": event.output_digest,
        }
        for event in trace
    ]


def trace_from_dict(document: Any) -> tuple[VMTraceEvent, ...]:
    """Validate and load a trace previously produced by :func:`trace_to_dict`."""
    if not isinstance(document, list):
        raise VMError("VM trace must be an array")
    events: list[VMTraceEvent] = []
    for index, raw in enumerate(document):
        if not isinstance(raw, Mapping):
            raise VMError(f"VM trace event {index} must be an object")
        required = {
            "node",
            "op",
            "inputs",
            "outputs",
            "cycles",
            "output_count",
            "output_min",
            "output_max",
            "output_digest",
        }
        if set(raw) != required:
            raise VMError(f"VM trace event {index} has invalid fields")
        if not isinstance(raw["node"], str) or not isinstance(raw["op"], str):
            raise VMError(f"VM trace event {index} names must be strings")
        if not isinstance(raw["inputs"], list) or not all(
            isinstance(value, str) for value in raw["inputs"]
        ):
            raise VMError(f"VM trace event {index} inputs must be a string array")
        if not isinstance(raw["outputs"], list) or not all(
            isinstance(value, str) for value in raw["outputs"]
        ):
            raise VMError(f"VM trace event {index} outputs must be a string array")
        if not _is_int(raw["cycles"]) or raw["cycles"] < 0:
            raise VMError(f"VM trace event {index} cycles must be a non-negative integer")
        if not _is_int(raw["output_count"]) or raw["output_count"] < 0:
            raise VMError(f"VM trace event {index} output_count must be non-negative")
        for key in ("output_min", "output_max"):
            if raw[key] is not None and not _is_int(raw[key]):
                raise VMError(f"VM trace event {index} {key} must be an integer or null")
        if not isinstance(raw["output_digest"], str) or len(raw["output_digest"]) != 64:
            raise VMError(f"VM trace event {index} output_digest must be a SHA-256 hex digest")
        events.append(
            VMTraceEvent(
                raw["node"],
                raw["op"],
                tuple(raw["inputs"]),
                tuple(raw["outputs"]),
                raw["cycles"],
                raw["output_count"],
                raw["output_min"],
                raw["output_max"],
                raw["output_digest"],
            )
        )
    return tuple(events)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _output_digest(values: Sequence[int]) -> str:
    """Stable byte-level digest without dumping potentially large tensors."""
    return hashlib.sha256(bytes(value & 0xFF for value in values)).hexdigest()


def _materialize(value: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if _is_int(value):
        return (), (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise VMError(f"weight value must be an integer or array, got {type(value).__name__}")
    children = [_materialize(item) for item in value]
    if not children:
        return (0,), ()
    shape = children[0][0]
    if any(child_shape != shape for child_shape, _ in children):
        raise VMError("weight tensor is not rectangular")
    return (len(children), *shape), tuple(
        element for _, elements in children for element in elements
    )


def _require_shape(label: str, desc: TensorDesc, expected: Sequence[int]) -> None:
    if desc.shape != tuple(expected):
        raise VMError(f"{label} output shape mismatch: {desc.shape} != {tuple(expected)}")


def _read(env: Mapping[str, list[int]], desc: TensorDesc) -> list[int]:
    try:
        values = env[desc.name]
    except KeyError:
        raise VMError(f"tensor not available: {desc.name}") from None
    if len(values) != desc.num_elements:
        raise VMError(
            f"tensor {desc.name!r} has {len(values)} elements, expected {desc.num_elements}"
        )
    return list(values)


def _require_values(values: Sequence[int], desc: TensorDesc) -> None:
    if desc.dtype is DType.UINT8:
        low, high = 0, 255
    elif desc.dtype is DType.INT8:
        low, high = INT8_MIN, INT8_MAX
    elif desc.dtype is DType.INT32:
        low, high = INT32_MIN, INT32_MAX
    else:
        raise VMError(f"unsupported tensor dtype: {desc.dtype.value}")
    if any(not _is_int(value) or not low <= value <= high for value in values):
        raise VMError(f"tensor {desc.name!r} contains a value outside [{low}, {high}]")


def _require_int8(values: Sequence[int], desc: TensorDesc) -> None:
    if desc.dtype is not DType.INT8:
        raise VMError(f"tensor {desc.name!r} must be int8, got {desc.dtype.value}")
    _require_values(values, desc)


def _weight(node: Node, weights: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    name = node.attrs.get(_WEIGHT_ATTR)
    if not isinstance(name, str) or name not in weights:
        raise VMError(f"weight {_WEIGHT_ATTR!r} is missing for node {node.name!r}")
    shape, values = _materialize(weights[name])
    if any(not _is_int(value) or not INT8_MIN <= value <= INT8_MAX for value in values):
        raise VMError(f"weight {name!r} contains a value outside int8")
    return shape, values


def _attr(node: Node, key: str, default: int, minimum: int = 0) -> int:
    value = node.attrs.get(key, default)
    if not _is_int(value) or value < minimum:
        raise VMError(f"invalid {key} on node {node.name!r}")
    return value


def _window_length(node: Node, length: int, kernel: int, stride: int, padding: int) -> int:
    output_length = (length + 2 * padding - kernel) // stride + 1
    if output_length <= 0:
        raise VMError(f"{node.op_type.value} produces no output: {node.name}")
    return output_length


def _requantized(values: Sequence[int], output: TensorDesc) -> list[int]:
    if any(not _is_int(value) or not INT32_MIN <= value <= INT32_MAX for value in values):
        raise VMError(f"accumulator for {output.name!r} is outside int32")
    if output.quant is None:
        raise VMError(f"tensor {output.name!r} has no quantization parameters")
    return [requantize(value, output.quant.multiplier, output.quant.shift) for value in values]


Handler = Callable[[Node, Graph, Mapping[str, list[int]], Mapping[str, Any]], tuple[list[int], int]]


def _embedding(
    node: Node, graph: Graph, env: Mapping[str, list[int]], weights: Mapping[str, Any]
) -> tuple[list[int], int]:
    input_desc, output_desc = graph.tensors[node.inputs[0]], graph.tensors[node.outputs[0]]
    tokens = _read(env, input_desc)
    _require_values(tokens, input_desc)
    shape, weight = _weight(node, weights)
    if input_desc.rank != 1 or len(shape) != 2 or shape[0] != _VOCAB_SIZE:
        raise VMError(f"invalid embedding shape: {node.name}")
    _require_shape("embedding", output_desc, (input_desc.shape[0], shape[1]))
    result: list[int] = []
    for token in tokens:
        if not 0 <= token < _VOCAB_SIZE:
            raise VMError(f"embedding index outside uint8 range: {token}")
        start = token * shape[1]
        result.extend(weight[start : start + shape[1]])
    return result, len(result) * 2


def _conv1d(
    node: Node, graph: Graph, env: Mapping[str, list[int]], weights: Mapping[str, Any]
) -> tuple[list[int], int]:
    input_desc, output_desc = graph.tensors[node.inputs[0]], graph.tensors[node.outputs[0]]
    x = _read(env, input_desc)
    _require_int8(x, input_desc)
    shape, weight = _weight(node, weights)
    if input_desc.rank != 2 or len(shape) != 3:
        raise VMError(f"invalid conv1d shape: {node.name}")
    in_channels, length = input_desc.shape
    out_channels, weight_channels, kernel = shape
    if weight_channels != in_channels:
        raise VMError(f"conv1d channel mismatch: {node.name}")
    stride = _attr(node, "stride", 1, 1)
    padding = _attr(node, "padding", 0)
    output_length = _window_length(node, length, kernel, stride, padding)
    _require_shape("conv1d", output_desc, (out_channels, output_length))
    accumulators: list[int] = []
    macs = 0
    for out_channel in range(out_channels):
        filter_base = out_channel * in_channels * kernel
        for out_pos in range(output_length):
            origin = out_pos * stride - padding
            first, last = max(0, -origin), min(kernel, length - origin)
            terms = [
                x[in_channel * length + origin + tap]
                * weight[filter_base + in_channel * kernel + tap]
                for in_channel in range(in_channels)
                for tap in range(first, last)
            ]
            macs += len(terms)
            accumulators.append(sum(terms))
    return _requantized(accumulators, output_desc), macs + len(accumulators) * 3


def _matmul(
    node: Node, graph: Graph, env: Mapping[str, list[int]], weights: Mapping[str, Any]
) -> tuple[list[int], int]:
    input_desc, output_desc = graph.tensors[node.inputs[0]], graph.tensors[node.outputs[0]]
    x = _read(env, input_desc)
    _require_int8(x, input_desc)
    shape, weight = _weight(node, weights)
    if input_desc.rank != 2 or len(shape) != 2:
        raise VMError(f"invalid matmul shape: {node.name}")
    rows, features = input_desc.shape
    output_features, weight_features = shape
    if features != weight_features:
        raise VMError(f"matmul feature mismatch: {node.name}")
    _require_shape("matmul", output_desc, (rows, output_features))
    accumulators = [
        sum(
            x[row * features + feature] * weight[out * features + feature]
            for feature in range(features)
        )
        for row in range(rows)
        for out in range(output_features)
    ]
    return _requantized(accumulators, output_desc), rows * output_features * features + len(
        accumulators
    ) * 3


def _add(
    node: Node, graph: Graph, env: Mapping[str, list[int]], _weights: Mapping[str, Any]
) -> tuple[list[int], int]:
    left_desc, right_desc = graph.tensors[node.inputs[0]], graph.tensors[node.inputs[1]]
    output_desc = graph.tensors[node.outputs[0]]
    left, right = _read(env, left_desc), _read(env, right_desc)
    _require_int8(left, left_desc)
    _require_int8(right, right_desc)
    if left_desc.shape != right_desc.shape:
        raise VMError(f"add input shape mismatch: {node.name}")
    _require_shape("add", output_desc, left_desc.shape)
    values = [a + b for a, b in zip(left, right)]
    return _requantized(values, output_desc), len(values) * 4


def _lut(
    node: Node, graph: Graph, env: Mapping[str, list[int]], _weights: Mapping[str, Any]
) -> tuple[list[int], int]:
    input_desc, output_desc = graph.tensors[node.inputs[0]], graph.tensors[node.outputs[0]]
    values = _read(env, input_desc)
    _require_int8(values, input_desc)
    table = node.attrs.get("table")
    if not isinstance(table, Sequence) or len(table) != 256:
        raise VMError(f"LUT on {node.name!r} must contain 256 entries")
    _require_shape("LUT", output_desc, input_desc.shape)
    return [lut_lookup(value, table) for value in values], len(values)


def _maxpool1d(
    node: Node, graph: Graph, env: Mapping[str, list[int]], _weights: Mapping[str, Any]
) -> tuple[list[int], int]:
    input_desc, output_desc = graph.tensors[node.inputs[0]], graph.tensors[node.outputs[0]]
    values = _read(env, input_desc)
    _require_int8(values, input_desc)
    axis = _attr(node, "axis", input_desc.rank - 1)
    if input_desc.rank < 1 or axis >= input_desc.rank:
        raise VMError(f"invalid maxpool1d axis: {node.name}")
    shape = input_desc.shape
    outer, length, inner = math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1 :])
    kernel = _attr(node, "kernel_size", length, 1)
    stride = _attr(node, "stride", 1, 1)
    padding = _attr(node, "padding", 0)
    output_length = _window_length(node, length, kernel, stride, padding)
    _require_shape("maxpool1d", output_desc, (*shape[:axis], output_length, *shape[axis + 1 :]))
    result: list[int] = []
    comparisons = 0
    for group in range(outer):
        base = group * length * inner
        for out_pos in range(output_length):
            origin = out_pos * stride - padding
            first, last = max(origin, 0), min(origin + kernel, length)
            for slot in range(inner):
                window = [values[base + position * inner + slot] for position in range(first, last)]
                result.append(max(window, default=INT8_MIN))
                comparisons += max(0, len(window) - 1)
    return result, comparisons + len(result)


def _argmax(
    node: Node, graph: Graph, env: Mapping[str, list[int]], _weights: Mapping[str, Any]
) -> tuple[list[int], int]:
    input_desc, output_desc = graph.tensors[node.inputs[0]], graph.tensors[node.outputs[0]]
    values = _read(env, input_desc)
    _require_int8(values, input_desc)
    if not values or len(values) > INT8_MAX + 1:
        raise VMError(f"argmax input cannot produce an int8 index: {node.name}")
    _require_shape("argmax", output_desc, ())
    return [max(range(len(values)), key=values.__getitem__)], max(0, len(values) - 1)


_HANDLERS: Final[Mapping[str, Handler]] = {
    "embedding": _embedding,
    "conv1d": _conv1d,
    "matmul": _matmul,
    "add": _add,
    "lut": _lut,
    "maxpool1d": _maxpool1d,
    "argmax": _argmax,
}


class VirtualMachine:
    """Loads and executes a KIR v1 bundle without importing a model frontend."""

    def __init__(self, graph: Graph, weights: Mapping[str, Any], *, alignment: int = 4) -> None:
        try:
            graph.validate()
            self.graph = graph
            self.weights = dict(weights)
            self.report = audit_graph(graph, self.weights, alignment=alignment)
            if not self.report.int32_overflow_free:
                raise VMError("KIR VM refused an unverified or overflowing int32 accumulator")
            self.plan = plan_graph_memory(graph, alignment=alignment)
        except VMError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise VMError(f"KIR VM rejected the bundle: {exc}") from exc
        self._arena = bytearray(self.plan.total_arena_bytes)

    @classmethod
    def from_bundle(cls, path: str | Path, *, alignment: int = 4) -> "VirtualMachine":
        """Load a JSON KIR v1 bundle directly from disk."""
        try:
            graph, weights = loads_bundle(Path(path).read_text(encoding="utf-8"))
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise VMError(f"cannot load KIR bundle {path}: {exc}") from exc
        return cls(graph, weights, alignment=alignment)

    def run(
        self,
        inputs: Mapping[str, Sequence[int]],
        *,
        max_cycles: int | None = None,
        fault_node: str | None = None,
    ) -> VMResult:
        """Execute one input mapping and return outputs plus reproducible evidence."""
        if max_cycles is not None and (not _is_int(max_cycles) or max_cycles < 0):
            raise VMError("max_cycles must be a non-negative integer")
        expected, provided = set(self.graph.inputs), set(inputs)
        if missing := expected - provided:
            raise VMError(f"missing graph inputs: {', '.join(sorted(missing))}")
        if extra := provided - expected:
            raise VMError(f"unexpected graph inputs: {', '.join(sorted(extra))}")
        env: dict[str, list[int]] = {}
        for name in self.graph.inputs:
            desc = self.graph.tensors[name]
            values = list(inputs[name])
            if len(values) != desc.num_elements:
                raise VMError(
                    f"input {name!r} has {len(values)} elements, expected {desc.num_elements}"
                )
            _require_values(values, desc)
            env[name] = values

        trace: list[VMTraceEvent] = []
        cycles = 0
        for node in self.graph.nodes:
            if fault_node == node.name:
                raise VMError(f"fault injection triggered at node {node.name!r}")
            handler = _HANDLERS.get(node.op_type.value)
            if handler is None:
                raise VMError(f"KIR VM does not implement op {node.op_type.value!r}")
            values, node_cycles = handler(node, self.graph, env, self.weights)
            if max_cycles is not None and cycles + node_cycles > max_cycles:
                raise VMError(
                    f"VM cycle budget exceeded at node {node.name!r}: "
                    f"{cycles + node_cycles} > {max_cycles}"
                )
            output = self.graph.tensors[node.outputs[0]]
            if len(values) != output.num_elements:
                raise VMError(f"node {node.name!r} produced the wrong number of values")
            env[output.name] = values
            cycles += node_cycles
            trace.append(
                VMTraceEvent(
                    node.name,
                    node.op_type.value,
                    tuple(node.inputs),
                    tuple(node.outputs),
                    node_cycles,
                    len(values),
                    min(values) if values else None,
                    max(values) if values else None,
                    _output_digest(values),
                )
            )

        metrics = VMMetrics(
            cycles=cycles,
            static_cycle_upper_bound=self.report.estimated_worst_cycles,
            instructions=len(trace),
            arena_bytes=self.plan.total_arena_bytes,
            peak_concurrent_bytes=self.plan.peak_concurrent_bytes,
            rom_bytes=self.report.total_rom_bytes,
            allocation_bytes=0,
            int32_overflow_free=self.report.int32_overflow_free,
        )
        return VMResult(
            {name: list(env[name]) for name in self.graph.outputs}, metrics, tuple(trace)
        )

    def replay(
        self,
        inputs: Mapping[str, Sequence[int]],
        expected_trace: Sequence[VMTraceEvent],
        *,
        max_cycles: int | None = None,
    ) -> VMResult:
        """Re-execute inputs and fail if the instruction trace changed."""
        result = self.run(inputs, max_cycles=max_cycles)
        expected = tuple(expected_trace)
        if result.trace != expected:
            first_difference = next(
                (
                    index
                    for index, pair in enumerate(zip(result.trace, expected))
                    if pair[0] != pair[1]
                ),
                min(len(result.trace), len(expected)),
            )
            raise VMError(f"VM trace mismatch at event {first_difference}")
        return result
