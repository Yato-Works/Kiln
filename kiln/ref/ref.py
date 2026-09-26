from __future__ import annotations



import math

from collections.abc import Callable, Mapping, Sequence

from dataclasses import dataclass

from typing import Any, Final



from ..ir.kir import DType, Graph, Node, OpType, TensorDesc

from ..ir.sem import (

    INT8_MAX,

    INT8_MIN,

    INT32_MAX,

    INT32_MIN,

    lut_lookup,

    requantize,

)



__all__ = ("run_ref",)





_WEIGHT_ATTR: Final = "weight"

_STRIDE_ATTR: Final = "stride"

_PADDING_ATTR: Final = "padding"

_KERNEL_ATTR: Final = "kernel_size"

_TABLE_ATTR: Final = "table"



_VOCAB_SIZE: Final = 256



Shape = tuple[int, ...]

Env = dict[str, list[int]]





def _is_int(value: Any) -> bool:

    return isinstance(value, int) and not isinstance(value, bool)





def _is_sequence(value: Any) -> bool:

    return isinstance(value, Sequence) and not isinstance(

        value,

        (str, bytes, bytearray),

    )





def _materialize(value: Any) -> tuple[Shape, tuple[int, ...]]:

    if not _is_sequence(value):

        if not _is_int(value):

            raise TypeError(

                f"weight value must be an integer, got {type(value).__name__}"

            )

        return (), (value,)



    children = [_materialize(item) for item in value]



    if not children:

        return (0,), ()



    shape = children[0][0]



    if any(child_shape != shape for child_shape, _ in children):

        raise ValueError("weight tensor is not rectangular")



    flat = tuple(element for _, elements in children for element in elements)



    return (len(children), *shape), flat





def _require_range(

    values: Sequence[int],

    low: int,

    high: int,

    label: str,

) -> None:

    for index, value in enumerate(values):

        if not low <= value <= high:

            raise ValueError(f"{label}[{index}]={value} outside [{low}, {high}]")





def _require_int8_dtype(desc: TensorDesc) -> None:

    if desc.dtype is not DType.INT8:

        raise TypeError(f"tensor {desc.name!r} must be int8, got {desc.dtype.value}")





def _require_int32(value: int) -> int:

    if not INT32_MIN <= value <= INT32_MAX:

        raise OverflowError(f"int32 accumulator overflow: {value}")

    return value





def _require_shape(label: str, desc: TensorDesc, expected: Shape) -> None:

    if desc.shape != expected:

        raise ValueError(f"{label} output shape mismatch: {desc.shape} != {expected}")





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





def _requantize_values(values: Sequence[int], output: TensorDesc) -> list[int]:

    _require_int8_dtype(output)



    quant = output.quant



    if quant is None:

        raise ValueError(f"tensor {output.name!r} has no quantization parameters")



    return [

        requantize(_require_int32(value), quant.multiplier, quant.shift)

        for value in values

    ]





@dataclass(frozen=True, slots=True)

class _Context:

    graph: Graph

    env: Env

    weights: Mapping[str, Any]



    def input(self, node: Node, index: int = 0) -> TensorDesc:

        return self.graph.tensors[node.inputs[index]]



    def output(self, node: Node) -> TensorDesc:

        return self.graph.tensors[node.outputs[0]]



    def read(self, desc: TensorDesc) -> list[int]:

        try:

            values = self.env[desc.name]

        except KeyError:

            raise KeyError(f"tensor not available: {desc.name}") from None



        if len(values) != desc.num_elements:

            raise ValueError(

                f"tensor {desc.name!r} has {len(values)} elements, "

                f"expected {desc.num_elements}"

            )



        return list(values)



    def read_int8(self, desc: TensorDesc) -> list[int]:

        values = self.read(desc)

        _require_int8_dtype(desc)

        _require_range(values, INT8_MIN, INT8_MAX, desc.name)

        return values



    def write(self, desc: TensorDesc, values: Sequence[int]) -> None:

        if len(values) != desc.num_elements:

            raise ValueError(

                f"tensor {desc.name!r} produced {len(values)} elements, "

                f"expected {desc.num_elements}"

            )



        self.env[desc.name] = list(values)



    def weight(self, node: Node) -> tuple[Shape, tuple[int, ...]]:

        name = node.attrs.get(_WEIGHT_ATTR)



        if not isinstance(name, str):

            raise TypeError(

                f"{_WEIGHT_ATTR!r} must name an entry in weights: {node.name}"

            )



        try:

            value = self.weights[name]

        except KeyError:

            raise KeyError(

                f"weight {name!r} not found for node {node.name!r}"

            ) from None



        shape, values = _materialize(value)

        _require_range(values, INT8_MIN, INT8_MAX, f"weight of {node.name}")



        return shape, values





def _embedding(node: Node, ctx: _Context) -> None:

    input_desc, output_desc = ctx.input(node), ctx.output(node)



    tokens = ctx.read(input_desc)

    weight_shape, weight = ctx.weight(node)



    if input_desc.rank != 1:

        raise ValueError(f"embedding input must be rank-1: {node.name}")



    if len(weight_shape) != 2 or weight_shape[0] != _VOCAB_SIZE:

        raise ValueError(

            f"embedding weight must have shape "

            f"({_VOCAB_SIZE}, embedding_dim): {node.name}"

        )



    _, embedding_dim = weight_shape

    _require_shape("embedding", output_desc, (input_desc.shape[0], embedding_dim))



    result: list[int] = []



    for token in tokens:

        if not 0 <= token < _VOCAB_SIZE:

            raise ValueError(f"embedding index outside uint8 range: {token}")



        start = token * embedding_dim

        result.extend(weight[start:start + embedding_dim])



    ctx.write(output_desc, result)





def _conv1d(node: Node, ctx: _Context) -> None:

    input_desc, output_desc = ctx.input(node), ctx.output(node)



    x = ctx.read_int8(input_desc)

    weight_shape, weight = ctx.weight(node)



    if input_desc.rank != 2:

        raise ValueError(

            f"conv1d input must be rank-2 (channels, length): {node.name}"

        )



    if len(weight_shape) != 3:

        raise ValueError(

            f"conv1d weight must be rank-3 "

            f"(out_channels, in_channels, kernel): {node.name}"

        )



    in_channels, length = input_desc.shape

    out_channels, weight_channels, kernel = weight_shape



    if weight_channels != in_channels:

        raise ValueError(f"conv1d channel mismatch: {node.name}")



    stride = _int_attr(node, _STRIDE_ATTR, 1, minimum=1)

    padding = _int_attr(node, _PADDING_ATTR, 0, minimum=0)

    output_length = _window_length("conv1d", node, length, kernel, stride, padding)



    _require_shape("conv1d", output_desc, (out_channels, output_length))



    accumulators: list[int] = []



    for out_channel in range(out_channels):

        filter_base = out_channel * in_channels * kernel



        for out_pos in range(output_length):

            origin = out_pos * stride - padding

            first = max(0, -origin)

            last = min(kernel, length - origin)



            accumulators.append(

                sum(

                    x[in_channel * length + origin + tap]

                    * weight[filter_base + in_channel * kernel + tap]

                    for in_channel in range(in_channels)

                    for tap in range(first, last)

                )

            )



    ctx.write(output_desc, _requantize_values(accumulators, output_desc))





def _matmul(node: Node, ctx: _Context) -> None:

    input_desc, output_desc = ctx.input(node), ctx.output(node)



    x = ctx.read_int8(input_desc)

    weight_shape, weight = ctx.weight(node)



    if input_desc.rank != 2:

        raise ValueError(f"matmul input must be rank-2: {node.name}")



    if len(weight_shape) != 2:

        raise ValueError(f"matmul weight must be rank-2: {node.name}")



    rows, features = input_desc.shape

    output_features, weight_features = weight_shape



    if features != weight_features:

        raise ValueError(f"matmul feature mismatch: {node.name}")



    _require_shape("matmul", output_desc, (rows, output_features))



    x_rows = [x[r * features:(r + 1) * features] for r in range(rows)]

    w_rows = [weight[o * features:(o + 1) * features] for o in range(output_features)]



    accumulators = [

        sum(a * b for a, b in zip(x_row, w_row, strict=True))

        for x_row in x_rows

        for w_row in w_rows

    ]



    ctx.write(output_desc, _requantize_values(accumulators, output_desc))





def _add(node: Node, ctx: _Context) -> None:

    left_desc, right_desc = ctx.input(node, 0), ctx.input(node, 1)

    output_desc = ctx.output(node)



    left = ctx.read_int8(left_desc)

    right = ctx.read_int8(right_desc)



    if left_desc.shape != right_desc.shape:

        raise ValueError(f"add input shape mismatch: {node.name}")



    _require_shape("add", output_desc, left_desc.shape)



    accumulators = [a + b for a, b in zip(left, right, strict=True)]



    ctx.write(output_desc, _requantize_values(accumulators, output_desc))





def _lut(node: Node, ctx: _Context) -> None:

    input_desc, output_desc = ctx.input(node), ctx.output(node)



    values = ctx.read_int8(input_desc)

    table = node.attrs[_TABLE_ATTR]



    ctx.write(output_desc, [lut_lookup(value, table) for value in values])





def _maxpool1d(node: Node, ctx: _Context) -> None:

    input_desc, output_desc = ctx.input(node), ctx.output(node)



    values = ctx.read_int8(input_desc)



    if input_desc.rank < 1:

        raise ValueError(f"maxpool1d input must be non-scalar: {node.name}")



    *leading, length = input_desc.shape

    kernel = _int_attr(node, _KERNEL_ATTR, length, minimum=1)

    stride = _int_attr(node, _STRIDE_ATTR, 1, minimum=1)

    padding = _int_attr(node, _PADDING_ATTR, 0, minimum=0)

    output_length = _window_length("maxpool1d", node, length, kernel, stride, padding)



    _require_shape("maxpool1d", output_desc, (*leading, output_length))



    result: list[int] = []



    for group in range(math.prod(leading)):

        row = values[group * length:(group + 1) * length]



        for out_pos in range(output_length):

            origin = out_pos * stride - padding

            window = row[max(origin, 0):max(origin + kernel, 0)]

            result.append(max(window, default=INT8_MIN))



    ctx.write(output_desc, result)





def _argmax(node: Node, ctx: _Context) -> None:

    input_desc, output_desc = ctx.input(node), ctx.output(node)



    values = ctx.read_int8(input_desc)



    if not values:

        raise ValueError(f"argmax input is empty: {node.name}")



    if len(values) > INT8_MAX + 1:

        raise ValueError(f"argmax result cannot fit in int8: {node.name}")



    best_index = max(range(len(values)), key=values.__getitem__)



    ctx.write(output_desc, [best_index])





_Handler = Callable[[Node, _Context], None]



_HANDLERS: Final[Mapping[OpType, _Handler]] = {

    OpType.EMBEDDING: _embedding,

    OpType.CONV1D: _conv1d,

    OpType.MATMUL: _matmul,

    OpType.ADD: _add,

    OpType.LUT: _lut,

    OpType.MAXPOOL1D: _maxpool1d,

    OpType.ARGMAX: _argmax,

}





def run_ref(

    graph: Graph,

    inputs: Mapping[str, Sequence[int]],

    weights: Mapping[str, Any],

) -> dict[str, list[int]]:

    graph.validate()



    provided, expected = set(inputs), set(graph.inputs)



    if missing := expected - provided:

        raise KeyError(f"missing graph inputs: {', '.join(sorted(missing))}")



    if extra := provided - expected:

        raise KeyError(f"unexpected graph inputs: {', '.join(sorted(extra))}")



    ctx = _Context(

        graph=graph,

        env={name: list(values) for name, values in inputs.items()},

        weights=weights,

    )



    for name in graph.inputs:

        ctx.read(graph.tensors[name])



    for node in graph.nodes:

        handler = _HANDLERS.get(node.op_type)



        if handler is None:

            raise NotImplementedError(

                f"reference execution is not implemented for {node.op_type.value}"

            )



        handler(node, ctx)



    return {name: list(ctx.env[name]) for name in graph.outputs}