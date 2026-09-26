from __future__ import annotations



import math

from collections import Counter

from collections.abc import Callable, Iterable, Mapping, Sequence

from collections.abc import Set as AbstractSet

from dataclasses import dataclass, field

from enum import Enum

from types import MappingProxyType

from typing import Any, Final, NamedTuple, NoReturn



from .plan import OpStep, TensorInfo



__all__ = (

    "DType",

    "QuantParam",

    "TensorDesc",

    "OpType",

    "Node",

    "Graph",

    "LoweredGraph",

)





_INT8_MIN: Final = -128

_INT8_MAX: Final = 127

_MULTIPLIER_MIN: Final = 1 << 30

_MULTIPLIER_MAX: Final = 1 << 31

_SHIFT_MIN: Final = 0

_SHIFT_MAX: Final = 31

_LUT_SIZE: Final = 1 << 8





class DType(Enum):

    UINT8 = "uint8"

    INT8 = "int8"

    INT32 = "int32"



    @property

    def size_bytes(self) -> int:

        return _DTYPE_SIZES[self]





_DTYPE_SIZES: Final[Mapping["DType", int]] = MappingProxyType(

    {

        DType.UINT8: 1,

        DType.INT8: 1,

        DType.INT32: 4,

    }

)





def _require_plain_int(value: Any, label: str) -> int:

    if not isinstance(value, int) or isinstance(value, bool):

        raise TypeError(f"{label} must be an integer, got {type(value).__name__}")



    return value





@dataclass(frozen=True, slots=True)

class QuantParam:

    multiplier: int

    shift: int



    def __post_init__(self) -> None:

        _require_plain_int(self.multiplier, "quantized multiplier")

        _require_plain_int(self.shift, "quantized shift")



        if not _MULTIPLIER_MIN <= self.multiplier < _MULTIPLIER_MAX:

            raise ValueError(

                f"invalid quantized multiplier: {self.multiplier} "

                f"(expected [{_MULTIPLIER_MIN}, {_MULTIPLIER_MAX}))"

            )



        if not _SHIFT_MIN <= self.shift <= _SHIFT_MAX:

            raise ValueError(

                f"invalid quantized shift: {self.shift} "

                f"(expected [{_SHIFT_MIN}, {_SHIFT_MAX}])"

            )





@dataclass(frozen=True, slots=True)

class TensorDesc:

    name: str

    shape: Sequence[int] = ()

    dtype: DType = DType.INT8

    quant: QuantParam | None = None



    def __post_init__(self) -> None:

        if not self.name:

            raise ValueError("tensor name must not be empty")



        shape = tuple(self.shape)



        for dim in shape:

            _require_plain_int(dim, f"tensor dimension of {self.name!r}")



        if any(dim <= 0 for dim in shape):

            raise ValueError(f"invalid tensor shape: {shape}")



        object.__setattr__(self, "shape", shape)



    @property

    def rank(self) -> int:

        return len(self.shape)



    @property

    def num_elements(self) -> int:

        return math.prod(self.shape)



    @property

    def size_bytes(self) -> int:

        return self.num_elements * self.dtype.size_bytes





class OpType(Enum):

    EMBEDDING = "embedding"

    CONV1D = "conv1d"

    MATMUL = "matmul"

    ADD = "add"

    LUT = "lut"

    MAXPOOL1D = "maxpool1d"

    ARGMAX = "argmax"





def _freeze(value: Any) -> Any:

    if isinstance(value, Mapping):

        return MappingProxyType({key: _freeze(item) for key, item in value.items()})



    if isinstance(value, (str, bytes)):

        return value



    if isinstance(value, bytearray):

        return bytes(value)



    if isinstance(value, AbstractSet):

        return frozenset(_freeze(item) for item in value)



    if isinstance(value, Sequence):

        return tuple(_freeze(item) for item in value)



    return value





@dataclass(frozen=True, slots=True)

class Node:

    op_type: OpType

    name: str

    inputs: Sequence[str]

    outputs: Sequence[str]

    attrs: Mapping[str, Any] = field(default_factory=dict)



    def __post_init__(self) -> None:

        if not self.name:

            raise ValueError("node name must not be empty")



        inputs = tuple(self.inputs)

        outputs = tuple(self.outputs)



        if not inputs:

            raise ValueError(f"node has no inputs: {self.name}")



        if not outputs:

            raise ValueError(f"node has no outputs: {self.name}")



        duplicated = _duplicates(outputs)



        if duplicated:

            raise ValueError(

                f"duplicate node output in {self.name}: {', '.join(duplicated)}"

            )



        object.__setattr__(self, "inputs", inputs)

        object.__setattr__(self, "outputs", outputs)

        object.__setattr__(self, "attrs", _freeze(self.attrs))



    def __hash__(self) -> int:

        return hash((self.op_type, self.name, self.inputs, self.outputs))





@dataclass(frozen=True, slots=True)

class _OpContext:

    node: Node

    inputs: tuple[TensorDesc, ...]

    outputs: tuple[TensorDesc, ...]



    @property

    def input(self) -> TensorDesc:

        return self.inputs[0]



    @property

    def output(self) -> TensorDesc:

        return self.outputs[0]



    def fail(self, reason: str, error: type[Exception] = ValueError) -> NoReturn:

        raise error(f"{self.node.op_type.value} node {self.node.name!r}: {reason}")



    def attr(self, name: str) -> Any:

        if name not in self.node.attrs:

            self.fail(f"missing required attribute {name!r}")



        return self.node.attrs[name]



    def int_attr(self, name: str, *, minimum: int, default: int | None = None) -> int:

        value = self.node.attrs.get(name, default)



        if value is None:

            self.fail(f"missing required attribute {name!r}")



        try:

            _require_plain_int(value, f"attribute {name!r}")

        except TypeError as error:

            self.fail(str(error), TypeError)



        if value < minimum:

            self.fail(f"attribute {name!r} must be >= {minimum}, got {value}")



        return value



    def require_dtype(self, tensor: TensorDesc, dtype: DType) -> None:

        if tensor.dtype is not dtype:

            self.fail(

                f"tensor {tensor.name!r} must be {dtype.value}, got {tensor.dtype.value}"

            )



    def require_rank(self, tensor: TensorDesc, *, minimum: int = 0, exact: int | None = None) -> None:

        if exact is not None and tensor.rank != exact:

            self.fail(f"tensor {tensor.name!r} must be rank-{exact}, got {tensor.shape}")



        if tensor.rank < minimum:

            self.fail(

                f"tensor {tensor.name!r} must be at least rank-{minimum}, got {tensor.shape}"

            )



    def require_shape(self, tensor: TensorDesc, expected: tuple[int, ...]) -> None:

        if tensor.shape != expected:

            self.fail(

                f"shape mismatch: {tensor.name!r} is {tensor.shape}, expected {expected}"

            )



    def require_output_quant(self) -> None:

        for tensor in self.outputs:

            if tensor.dtype is DType.INT8 and tensor.quant is None:

                self.fail(f"missing quantization parameters for {tensor.name!r}")





def _window_output_length(

    length: int,

    kernel: int,

    stride: int,

    padding: int,

) -> int | None:

    span = length + 2 * padding - kernel



    if span < 0:

        return None



    return span // stride + 1





def _validate_embedding(ctx: _OpContext) -> None:

    ctx.attr("weight")

    ctx.require_dtype(ctx.output, DType.INT8)





def _validate_conv1d(ctx: _OpContext) -> None:

    ctx.attr("weight")

    ctx.int_attr("stride", minimum=1, default=1)

    ctx.int_attr("padding", minimum=0, default=0)

    ctx.require_output_quant()

    ctx.require_rank(ctx.input, minimum=2)



    if ctx.output.rank != ctx.input.rank:

        ctx.fail(f"rank mismatch: input {ctx.input.shape}, output {ctx.output.shape}")





def _validate_matmul(ctx: _OpContext) -> None:

    ctx.attr("weight")

    ctx.require_output_quant()

    ctx.require_rank(ctx.input, exact=2)

    ctx.require_rank(ctx.output, exact=2)





def _validate_add(ctx: _OpContext) -> None:

    ctx.require_output_quant()



    left, right = ctx.inputs



    ctx.require_shape(right, left.shape)

    ctx.require_shape(ctx.output, left.shape)





def _validate_lut(ctx: _OpContext) -> None:

    table = ctx.attr("table")



    if isinstance(table, (str, bytes)) or not isinstance(table, Sequence):

        ctx.fail(

            f"attribute 'table' must be a sequence of int, got {type(table).__name__}",

            TypeError,

        )



    if len(table) != _LUT_SIZE:

        ctx.fail(f"table must contain exactly {_LUT_SIZE} entries, got {len(table)}")



    for index, value in enumerate(table):

        try:

            _require_plain_int(value, f"table entry {index}")

        except TypeError as error:

            ctx.fail(str(error), TypeError)



        if not _INT8_MIN <= value <= _INT8_MAX:

            ctx.fail(f"table entry {index} is outside int8 range: {value}")



    ctx.require_shape(ctx.output, ctx.input.shape)





def _validate_maxpool1d(ctx: _OpContext) -> None:

    ctx.require_rank(ctx.input, minimum=1)



    shape = ctx.input.shape

    axis = ctx.int_attr("axis", minimum=0, default=len(shape) - 1)

    if axis >= len(shape):

        ctx.fail(f"axis {axis} is outside the input rank {len(shape)}")



    length = shape[axis]

    kernel = ctx.int_attr("kernel_size", minimum=1, default=length)

    stride = ctx.int_attr("stride", minimum=1, default=1)

    padding = ctx.int_attr("padding", minimum=0, default=0)



    output_length = _window_output_length(length, kernel, stride, padding)

    if output_length is None:

        ctx.fail(f"kernel {kernel} exceeds padded input length {length + 2 * padding}")



    # ``axis`` defaults to the last axis, reproducing the original single-axis

    # pooling exactly; ``axis=0`` pools the leading (token) axis, which is what

    # a text/bag-shaped model needs (ROADMAP.md Phase 2).

    ctx.require_shape(ctx.output, shape[:axis] + (output_length,) + shape[axis + 1:])
def _validate_argmax(ctx: _OpContext) -> None:

    ctx.require_rank(ctx.input, minimum=1)

    ctx.require_dtype(ctx.output, DType.INT8)

    ctx.require_shape(ctx.output, ())

    # Total elements across all dimensions must fit in int8 non-negative range [0, 127]
    total_elements = ctx.input.num_elements

    if total_elements > _INT8_MAX + 1:

        ctx.fail(f"{total_elements} elements exceed the max index (127) representable by an int8 output")





class _OpSpec(NamedTuple):

    num_inputs: int

    num_outputs: int

    validate: Callable[[_OpContext], None]





_OP_SPECS: Final[Mapping[OpType, _OpSpec]] = MappingProxyType(

    {

        OpType.EMBEDDING: _OpSpec(1, 1, _validate_embedding),

        OpType.CONV1D: _OpSpec(1, 1, _validate_conv1d),

        OpType.MATMUL: _OpSpec(1, 1, _validate_matmul),

        OpType.ADD: _OpSpec(2, 1, _validate_add),

        OpType.LUT: _OpSpec(1, 1, _validate_lut),

        OpType.MAXPOOL1D: _OpSpec(1, 1, _validate_maxpool1d),

        OpType.ARGMAX: _OpSpec(1, 1, _validate_argmax),

    }

)





class LoweredGraph(NamedTuple):

    op_steps: tuple[OpStep, ...]

    tensors: Mapping[str, TensorInfo]

    graph_outputs: tuple[str, ...]





@dataclass(frozen=True, slots=True)

class Graph:

    inputs: Sequence[str]

    outputs: Sequence[str]

    nodes: Sequence[Node]

    tensors: Mapping[str, TensorDesc]



    def __post_init__(self) -> None:

        inputs = tuple(self.inputs)

        outputs = tuple(self.outputs)

        nodes = tuple(self.nodes)

        tensors = MappingProxyType(dict(self.tensors))



        _require_unique(inputs, "graph input")

        _require_unique(outputs, "graph output")



        if not tensors:

            raise ValueError("graph must contain tensors")



        for name, tensor in tensors.items():

            if name != tensor.name:

                raise ValueError(f"tensor key/name mismatch: {name} != {tensor.name}")



        object.__setattr__(self, "inputs", inputs)

        object.__setattr__(self, "outputs", outputs)

        object.__setattr__(self, "nodes", nodes)

        object.__setattr__(self, "tensors", tensors)



    def __hash__(self) -> int:

        return hash((self.inputs, self.outputs, self.nodes))



    def validate(self) -> None:

        self._validate_graph_io()

        self._validate_nodes()



    def lower(self) -> LoweredGraph:

        self.validate()



        return LoweredGraph(

            op_steps=tuple(self.to_op_steps()),

            tensors=MappingProxyType(self.to_tensor_infos()),

            graph_outputs=tuple(self.outputs),

        )



    def to_op_steps(self) -> list[OpStep]:

        return [

            OpStep(op_id=index, inputs=node.inputs, outputs=node.outputs)

            for index, node in enumerate(self.nodes)

        ]



    def to_tensor_infos(self) -> dict[str, TensorInfo]:

        return {

            name: TensorInfo(name=name, size_bytes=desc.size_bytes)

            for name, desc in self.tensors.items()

        }



    def _validate_graph_io(self) -> None:

        for label, names in (("input", self.inputs), ("output", self.outputs)):

            for name in names:

                if name not in self.tensors:

                    raise KeyError(f"unknown graph {label}: {name}")



    def _validate_nodes(self) -> None:

        defined = set(self.inputs)

        produced: set[str] = set()

        seen_names: set[str] = set()



        for node in self.nodes:

            if node.name in seen_names:

                raise ValueError(f"duplicate node name: {node.name}")



            seen_names.add(node.name)

            spec = _spec_for(node)



            self._validate_arity(node, spec)

            self._validate_node_inputs(node, defined)

            self._validate_node_outputs(node, defined)



            spec.validate(self._context(node))



            defined.update(node.outputs)

            produced.update(node.outputs)



        unproduced = tuple(name for name in self.outputs if name not in produced)



        if unproduced:

            raise ValueError(

                f"graph outputs are not produced by any node: {', '.join(unproduced)}"

            )



    def _validate_arity(self, node: Node, spec: _OpSpec) -> None:

        for label, actual, expected in (

            ("inputs", len(node.inputs), spec.num_inputs),

            ("outputs", len(node.outputs), spec.num_outputs),

        ):

            if actual != expected:

                raise ValueError(

                    f"{node.op_type.value} expects {expected} {label}, got {actual}"

                )



    def _validate_node_inputs(self, node: Node, defined: AbstractSet[str]) -> None:

        for name in node.inputs:

            if name not in self.tensors:

                raise KeyError(f"unknown tensor {name!r} in node {node.name!r}")



            if name not in defined:

                raise ValueError(f"forward or undefined reference: {node.name}.{name}")



    def _validate_node_outputs(self, node: Node, defined: AbstractSet[str]) -> None:

        for name in node.outputs:

            if name not in self.tensors:

                raise KeyError(f"unknown tensor {name!r} in node {node.name!r}")



            if name in defined:

                raise ValueError(f"multiple definitions of tensor: {name}")



    def _context(self, node: Node) -> _OpContext:

        return _OpContext(

            node=node,

            inputs=tuple(self.tensors[name] for name in node.inputs),

            outputs=tuple(self.tensors[name] for name in node.outputs),

        )





def _spec_for(node: Node) -> _OpSpec:

    try:

        return _OP_SPECS[node.op_type]

    except KeyError:

        raise NotImplementedError(

            f"no validation rules for op type: {node.op_type.value}"

        ) from None





def _duplicates(values: Iterable[str]) -> tuple[str, ...]:

    counts = Counter(values)

    return tuple(name for name, count in counts.items() if count > 1)





def _require_unique(values: Sequence[str], label: str) -> None:

    duplicated = _duplicates(values)



    if duplicated:

        raise ValueError(f"duplicate {label}: {', '.join(duplicated)}")