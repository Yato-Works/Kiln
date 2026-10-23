from __future__ import annotations


import math

import re

from collections.abc import Callable, Iterable, Mapping, Sequence

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Final


from ...ir.kir import DType, Graph, Node, OpType, TensorDesc

from ...budget.report import audit_graph, format_budget_certificate
from ...ir.memory import plan_graph_memory


__all__ = ("emit_c", "write_c")


_WEIGHT_ATTR: Final = "weight"

_STRIDE_ATTR: Final = "stride"

_PADDING_ATTR: Final = "padding"

_KERNEL_ATTR: Final = "kernel_size"

_AXIS_ATTR: Final = "axis"

_TABLE_ATTR: Final = "table"


_INT8_MIN: Final = -128

_INT8_MAX: Final = 127

_VOCAB_SIZE: Final = 1 << 8

_ROW_WIDTH: Final = 12

_INDENT: Final = "    "


_WEIGHTED_OPS: Final = frozenset({OpType.EMBEDDING, OpType.CONV1D, OpType.MATMUL})


_INCLUDES: Final = (
    "#include <stdint.h>",
    "#include <stddef.h>",
    "#include <stdlib.h>",
    "#include <limits.h>",
)


_ARENA: Final = """\

static union {

    int32_t align;

    uint8_t data[ARENA_SIZE];

} _arena;"""


_HELPERS: Final = """\

static inline int32_t _require_int32(int64_t value) {

    if (value < INT32_MIN || value > INT32_MAX) abort();

    return (int32_t)value;

}



static inline int32_t _clamp_int32(int64_t value) {

    if (value < INT32_MIN) return INT32_MIN;

    if (value > INT32_MAX) return INT32_MAX;

    return (int32_t)value;

}



static inline int32_t _srdhm(int32_t a, int32_t b) {

    if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;

    int64_t product = (int64_t)a * (int64_t)b;

    int64_t nudge = product >= 0 ? (int64_t)(1 << 30) : 1 - (int64_t)(1 << 30);

    int64_t result = (product + nudge) / (int64_t)(1LL << 31);

    return _clamp_int32(result);

}



static inline int32_t _round_divide_by_pot(int32_t value, int exponent) {

    if (exponent == 0) return value;

    int64_t magnitude = value < 0 ? -(int64_t)value : (int64_t)value;

    magnitude = (magnitude + (int64_t)(1LL << (exponent - 1))) >> exponent;

    return value < 0 ? -(int32_t)magnitude : (int32_t)magnitude;

}



static inline int8_t _requantize(int32_t accum, int32_t multiplier, int shift) {

    int32_t scaled = _srdhm(accum, multiplier);

    int32_t shifted = _round_divide_by_pot(scaled, shift);

    if (shifted < INT8_MIN) return INT8_MIN;

    if (shifted > INT8_MAX) return INT8_MAX;

    return (int8_t)shifted;

}



static inline int8_t _lut_lookup(int8_t value, const int8_t table[256]) {

    return table[(int)value + 128];

}"""


Shape = tuple[int, ...]


def _identifier(value: str) -> str:

    name = re.sub(r"\W", "_", value, flags=re.ASCII) or "value"

    return f"_{name}" if name[0].isdigit() else name


def _tensor_symbol(name: str) -> str:

    return f"t_{_identifier(name)}"


def _weight_symbol(name: str) -> str:

    return f"_w_{_identifier(name)}"


def _lut_symbol(node: Node) -> str:

    return f"_lut_{_identifier(node.name)}"


def _io_symbols(node: Node) -> tuple[str, str]:

    return _tensor_symbol(node.inputs[0]), _tensor_symbol(node.outputs[0])


def _is_int(value: Any) -> bool:

    return isinstance(value, int) and not isinstance(value, bool)


def _is_uint8(dtype: DType) -> bool:

    return getattr(dtype, "value", None) == "uint8"


def _materialize(value: Any) -> tuple[Shape, tuple[int, ...]]:

    if not isinstance(value, (list, tuple)):
        if not _is_int(value):
            raise TypeError(f"weight value must be an integer, got {type(value).__name__}")

        return (), (value,)

    children = [_materialize(item) for item in value]

    if not children:
        return (0,), ()

    shape = children[0][0]

    if any(child_shape != shape for child_shape, _ in children):
        raise ValueError("weight tensor is not rectangular")

    flat = tuple(item for _, values in children for item in values)

    return (len(children), *shape), flat


def _require_int8_values(values: Iterable[int], label: str) -> None:

    for value in values:
        if not _INT8_MIN <= value <= _INT8_MAX:
            raise ValueError(f"{label} contains non-int8 value: {value}")


def _array_declaration(symbol: str, values: Sequence[int]) -> str:

    rows = ",\n".join(
        _INDENT + ", ".join(map(str, values[start : start + _ROW_WIDTH]))
        for start in range(0, len(values), _ROW_WIDTH)
    )

    return f"static const int8_t {symbol}[{len(values)}] = {{\n{rows}\n}};"


def _ctype(name: str, desc: TensorDesc) -> str:

    if _is_uint8(desc.dtype):
        return "uint8_t"

    if desc.dtype is DType.INT8:
        return "int8_t"

    if desc.dtype is DType.INT32:
        raise NotImplementedError(f"runtime int32 tensors are not supported by emit_c: {name}")

    raise TypeError(f"unsupported runtime dtype for tensor {name!r}: {desc.dtype.value}")


def _indent(lines: Iterable[str]) -> list[str]:

    return [f"{_INDENT}{line}" if line else "" for line in lines]


def _spaced(blocks: Iterable[str]) -> list[str]:

    return [line for block in blocks for line in (block, "")]


def _validate_phase0_graph(graph: Graph) -> None:

    if len(graph.inputs) != 1:
        raise ValueError("Phase 0 C emission requires exactly one graph input")

    if len(graph.outputs) != 1:
        raise ValueError("Phase 0 C emission requires exactly one graph output")

    input_desc = graph.tensors[graph.inputs[0]]

    output_desc = graph.tensors[graph.outputs[0]]

    if input_desc.dtype not in (DType.UINT8, DType.INT8):
        raise ValueError(
            f"kiln_predict requires a uint8 or int8 graph input, got {input_desc.dtype.value}"
        )

    if output_desc.dtype is not DType.INT8:
        raise ValueError("Phase 0 kiln_predict requires an int8 graph output")

    if output_desc.shape != ():
        raise ValueError("Phase 0 kiln_predict requires a scalar graph output")

    if not graph.nodes:
        raise ValueError("cannot emit an empty graph")

    final = graph.nodes[-1]

    if final.op_type is not OpType.ARGMAX:
        raise ValueError("Phase 0 kiln_predict requires ARGMAX as the final op")

    if final.outputs[0] != graph.outputs[0]:
        raise ValueError("ARGMAX output must be the graph output")

    for name, desc in graph.tensors.items():
        _ctype(name, desc)


def _weight_name(node: Node) -> str:

    name = node.attrs.get(_WEIGHT_ATTR)

    if not isinstance(name, str):
        raise TypeError(f"{_WEIGHT_ATTR!r} must name an entry in weights: {node.name}")

    return name


@dataclass(frozen=True, slots=True)
class _Weight:
    symbol: str

    shape: Shape

    values: tuple[int, ...]

    @property
    def declaration(self) -> str:

        return _array_declaration(self.symbol, self.values)


def _collect_weights(graph: Graph, weights: Mapping[str, Any]) -> dict[str, _Weight]:

    collected: dict[str, _Weight] = {}

    for node in graph.nodes:
        if node.op_type not in _WEIGHTED_OPS:
            continue

        name = _weight_name(node)

        if name in collected:
            continue

        try:
            value = weights[name]

        except KeyError:
            raise KeyError(f"weight not found: {name}") from None

        shape, values = _materialize(value)

        if not values:
            raise ValueError(f"empty weight tensor: {name}")

        _require_int8_values(values, f"weight {name!r}")

        collected[name] = _Weight(_weight_symbol(name), shape, values)

    return collected


def _lut_declarations(graph: Graph) -> list[str]:

    declarations: list[str] = []

    for node in graph.nodes:
        if node.op_type is not OpType.LUT:
            continue

        table = node.attrs.get(_TABLE_ATTR)

        if not isinstance(table, (list, tuple)):
            raise TypeError(f"invalid LUT table: {node.name}")

        if len(table) != _VOCAB_SIZE:
            raise ValueError(f"invalid LUT size: {node.name}")

        if not all(_is_int(value) for value in table):
            raise TypeError(f"invalid LUT entry: {node.name}")

        _require_int8_values(table, f"LUT {node.name!r}")

        declarations.append(_array_declaration(_lut_symbol(node), tuple(table)))

    return declarations


def _tensor_declarations(graph: Graph, offsets: Mapping[str, int]) -> list[str]:

    lines: list[str] = []

    for name, desc in graph.tensors.items():
        ctype = _ctype(name, desc)

        if name in graph.inputs:
            lines.append(
                f"const {ctype} *{_tensor_symbol(name)} = "
                f"(const {ctype} *)(const void *)input_bytes;"
            )

            continue

        lines.append(
            f"{ctype} *{_tensor_symbol(name)} = ({ctype} *)(_arena.data + {offsets[name]});"
        )

    return [*lines, ""] if lines else []


@dataclass(frozen=True, slots=True)
class _Context:
    graph: Graph

    weights: Mapping[str, _Weight]

    def input(self, node: Node, index: int = 0) -> TensorDesc:

        return self.graph.tensors[node.inputs[index]]

    def output(self, node: Node) -> TensorDesc:

        return self.graph.tensors[node.outputs[0]]

    def weight(self, node: Node) -> _Weight:

        return self.weights[_weight_name(node)]

    def quant(self, node: Node) -> str:

        quant = self.output(node).quant

        if quant is None:
            raise ValueError(f"missing quantization parameters: {node.name}")

        return f"{quant.multiplier}, {quant.shift}"


def _int_attr(node: Node, key: str, default: int, *, minimum: int) -> int:

    value = node.attrs.get(key, default)

    if not _is_int(value) or value < minimum:
        raise ValueError(f"invalid {key}: {node.name}")

    return value


def _window_length(
    label: str,
    node: Node,
    length: int,
    kernel: int,
    stride: int,
    padding: int,
) -> int:

    output_length = (length + 2 * padding - kernel) // stride + 1

    if output_length <= 0:
        raise ValueError(f"{label} produces no output: {node.name}")

    return output_length


def _window_bounds(length: int, kernel: int, stride: int, padding: int) -> list[str]:

    return [
        f"int origin = op * {stride} - {padding};",
        "int first = origin < 0 ? -origin : 0;",
        f"int last = {length} - origin < {kernel} ? {length} - origin : {kernel};",
    ]


def _require_shape(label: str, node: Node, desc: TensorDesc, expected: Shape) -> None:

    if desc.shape != expected:
        raise ValueError(f"{label} output shape mismatch: {node.name}")


def _emit_embedding(node: Node, ctx: _Context) -> list[str]:

    input_desc, output_desc = ctx.input(node), ctx.output(node)

    weight = ctx.weight(node)

    if input_desc.rank != 1:
        raise ValueError(f"embedding input must be rank-1: {node.name}")

    if len(weight.shape) != 2 or weight.shape[0] != _VOCAB_SIZE:
        raise ValueError(
            f"embedding weight must have shape ({_VOCAB_SIZE}, embedding_dim): {node.name}"
        )

    (length,), (_, dim) = input_desc.shape, weight.shape

    _require_shape("embedding", node, output_desc, (length, dim))

    x, y = _io_symbols(node)

    return [
        f"for (int i = 0; i < {length}; ++i) {{",
        f"    uint8_t token = {x}[i];",
        f"    for (int j = 0; j < {dim}; ++j) {{",
        f"        {y}[i * {dim} + j] = {weight.symbol}[(size_t)token * {dim} + j];",
        "    }",
        "}",
    ]


def _emit_conv1d(node: Node, ctx: _Context) -> list[str]:

    input_desc, output_desc = ctx.input(node), ctx.output(node)

    weight = ctx.weight(node)

    if input_desc.rank != 2:
        raise ValueError(f"conv1d input must be rank-2: {node.name}")

    if len(weight.shape) != 3:
        raise ValueError(f"conv1d weight must be rank-3: {node.name}")

    in_channels, length = input_desc.shape

    out_channels, weight_channels, kernel = weight.shape

    if weight_channels != in_channels:
        raise ValueError(f"conv1d channel mismatch: {node.name}")

    stride = _int_attr(node, _STRIDE_ATTR, 1, minimum=1)

    padding = _int_attr(node, _PADDING_ATTR, 0, minimum=0)

    output_length = _window_length("conv1d", node, length, kernel, stride, padding)

    _require_shape("conv1d", node, output_desc, (out_channels, output_length))

    x, y = _io_symbols(node)

    filter_size = in_channels * kernel

    return [
        f"for (int oc = 0; oc < {out_channels}; ++oc) {{",
        f"    for (int op = 0; op < {output_length}; ++op) {{",
        *_indent(_indent(_window_bounds(length, kernel, stride, padding))),
        "        int64_t acc = 0;",
        f"        for (int ic = 0; ic < {in_channels}; ++ic) {{",
        "            for (int k = first; k < last; ++k) {",
        f"                acc += (int64_t){x}[ic * {length} + origin + k] * "
        f"(int64_t){weight.symbol}[oc * {filter_size} + ic * {kernel} + k];",
        "            }",
        "        }",
        f"        {y}[oc * {output_length} + op] = "
        f"_requantize(_require_int32(acc), {ctx.quant(node)});",
        "    }",
        "}",
    ]


def _emit_matmul(node: Node, ctx: _Context) -> list[str]:

    input_desc, output_desc = ctx.input(node), ctx.output(node)

    weight = ctx.weight(node)

    if input_desc.rank != 2:
        raise ValueError(f"matmul input must be rank-2: {node.name}")

    if len(weight.shape) != 2:
        raise ValueError(f"matmul weight must be rank-2: {node.name}")

    rows, features = input_desc.shape

    output_features, weight_features = weight.shape

    if features != weight_features:
        raise ValueError(f"matmul feature mismatch: {node.name}")

    _require_shape("matmul", node, output_desc, (rows, output_features))

    x, y = _io_symbols(node)

    return [
        f"for (int r = 0; r < {rows}; ++r) {{",
        f"    for (int o = 0; o < {output_features}; ++o) {{",
        "        int64_t acc = 0;",
        f"        for (int f = 0; f < {features}; ++f) {{",
        f"            acc += (int64_t){x}[r * {features} + f] * "
        f"(int64_t){weight.symbol}[o * {features} + f];",
        "        }",
        f"        {y}[r * {output_features} + o] = "
        f"_requantize(_require_int32(acc), {ctx.quant(node)});",
        "    }",
        "}",
    ]


def _emit_add(node: Node, ctx: _Context) -> list[str]:

    left = _tensor_symbol(node.inputs[0])

    right = _tensor_symbol(node.inputs[1])

    y = _tensor_symbol(node.outputs[0])

    size = ctx.output(node).num_elements

    return [
        f"for (int i = 0; i < {size}; ++i) {{",
        f"    int64_t acc = (int64_t){left}[i] + (int64_t){right}[i];",
        f"    {y}[i] = _requantize(_require_int32(acc), {ctx.quant(node)});",
        "}",
    ]


def _emit_lut(node: Node, ctx: _Context) -> list[str]:

    x, y = _io_symbols(node)

    size = ctx.input(node).num_elements

    return [
        f"for (int i = 0; i < {size}; ++i) {{",
        f"    {y}[i] = _lut_lookup({x}[i], {_lut_symbol(node)});",
        "}",
    ]


def _emit_maxpool1d(node: Node, ctx: _Context) -> list[str]:

    input_desc, output_desc = ctx.input(node), ctx.output(node)

    if input_desc.rank < 1:
        raise ValueError(f"maxpool1d input must be non-scalar: {node.name}")

    shape = input_desc.shape

    axis = _int_attr(node, _AXIS_ATTR, len(shape) - 1, minimum=0)

    if axis >= len(shape):
        raise ValueError(
            f"maxpool1d axis {axis} is outside the input rank {len(shape)}: {node.name}"
        )

    outer = math.prod(shape[:axis])

    length = shape[axis]

    inner = math.prod(shape[axis + 1 :])

    kernel = _int_attr(node, _KERNEL_ATTR, length, minimum=1)

    stride = _int_attr(node, _STRIDE_ATTR, 1, minimum=1)

    padding = _int_attr(node, _PADDING_ATTR, 0, minimum=0)

    output_length = _window_length("maxpool1d", node, length, kernel, stride, padding)

    _require_shape(
        "maxpool1d", node, output_desc, (*shape[:axis], output_length, *shape[axis + 1 :])
    )

    x, y = _io_symbols(node)

    bounds = _indent(_indent(_window_bounds(length, kernel, stride, padding)))

    if inner == 1:
        # Byte-identical to the pre-``axis`` emission, which is the default case.

        body = [
            *bounds,
            "        int8_t best = INT8_MIN;",
            "        for (int k = first; k < last; ++k) {",
            f"            int8_t value = {x}[g * {length} + origin + k];",
            "            if (value > best) best = value;",
            "        }",
            f"        {y}[g * {output_length} + op] = best;",
        ]

    else:
        body = [
            *bounds,
            f"        for (int i = 0; i < {inner}; ++i) {{",
            "            int8_t best = INT8_MIN;",
            "            for (int k = first; k < last; ++k) {",
            f"                int8_t value = {x}[(g * {length} + origin + k) * {inner} + i];",
            "                if (value > best) best = value;",
            "            }",
            f"            {y}[(g * {output_length} + op) * {inner} + i] = best;",
            "        }",
        ]

    return [
        f"for (int g = 0; g < {outer}; ++g) {{",
        f"    for (int op = 0; op < {output_length}; ++op) {{",
        *body,
        "    }",
        "}",
    ]


def _emit_argmax(node: Node, ctx: _Context) -> list[str]:

    if ctx.output(node).shape != ():
        raise ValueError(f"argmax output must be scalar: {node.name}")

    size = ctx.input(node).num_elements

    if not 0 < size <= _INT8_MAX + 1:
        raise ValueError(f"invalid argmax class count: {node.name}")

    x, y = _io_symbols(node)

    return [
        "{",
        "    int best_index = 0;",
        f"    int8_t best_value = {x}[0];",
        f"    for (int i = 1; i < {size}; ++i) {{",
        f"        if ({x}[i] > best_value) {{",
        "            best_index = i;",
        f"            best_value = {x}[i];",
        "        }",
        "    }",
        f"    {y}[0] = (int8_t)best_index;",
        "}",
    ]


_Emitter = Callable[[Node, _Context], list[str]]


_EMITTERS: Final[Mapping[OpType, _Emitter]] = {
    OpType.EMBEDDING: _emit_embedding,
    OpType.CONV1D: _emit_conv1d,
    OpType.MATMUL: _emit_matmul,
    OpType.ADD: _emit_add,
    OpType.LUT: _emit_lut,
    OpType.MAXPOOL1D: _emit_maxpool1d,
    OpType.ARGMAX: _emit_argmax,
}


def _emit_node(node: Node, ctx: _Context) -> list[str]:

    emitter = _EMITTERS.get(node.op_type)

    if emitter is None:
        raise NotImplementedError(f"unsupported op type: {node.op_type.value}")

    return emitter(node, ctx)


def emit_c(
    graph: Graph,
    weights: Mapping[str, Any],
    *,
    alignment: int = 4,
    runtime_header: str | None = "kiln_runtime.h",
) -> str:

    graph.validate()

    _validate_phase0_graph(graph)

    plan = plan_graph_memory(graph, alignment=alignment)

    ctx = _Context(graph, _collect_weights(graph, weights))

    input_name, output_name = graph.inputs[0], graph.outputs[0]

    input_size = graph.tensors[input_name].num_elements

    body = [
        f"if (input_bytes == NULL || len != {input_size}) abort();",
        "",
        *_tensor_declarations(graph, plan.offsets),
        "",
    ]

    for node in graph.nodes:
        body.extend((*_emit_node(node, ctx), ""))

    body.append(f"return {_tensor_symbol(output_name)}[0];")

    includes = list(_INCLUDES)

    if runtime_header is not None:
        includes.append(f'#include "{runtime_header}"')

    audit_report = audit_graph(graph, weights, plan=plan, alignment=alignment)
    certificate = format_budget_certificate(audit_report)

    source = [
        certificate,
        "",
        *includes,
        "",
        f"#define ARENA_SIZE {plan.total_arena_bytes}",
        "",
        *_spaced((_ARENA, _HELPERS)),
        *_spaced(weight.declaration for weight in ctx.weights.values()),
        *_spaced(_lut_declarations(graph)),
        "int8_t kiln_predict(const uint8_t *input_bytes, int len) {",
        *_indent(body),
        "}",
        "",
    ]

    return "\n".join(source)


def write_c(
    path: str | Path,
    graph: Graph,
    weights: Mapping[str, Any],
    *,
    alignment: int = 4,
    runtime_header: str | None = "kiln_runtime.h",
) -> None:

    Path(path).write_text(
        emit_c(graph, weights, alignment=alignment, runtime_header=runtime_header),
        encoding="utf-8",
    )
