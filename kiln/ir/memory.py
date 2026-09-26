from __future__ import annotations

from collections import Counter

from typing import Final, Mapping

from .kir import Graph, OpType

from .plan import (
    DEFAULT_ALIGNMENT,
    MemoryPlan,
    allocate_arena,
    calculate_liveness,
)

__all__ = ("INPLACE_OPS", "plan_graph_memory", "plan_inplace_aliases")

#: Ops whose emitted C loop reads and writes one element at a time, in the same
#: order, so the output tensor may safely overwrite the input buffer.
INPLACE_OPS: Final = frozenset({OpType.LUT})


def plan_inplace_aliases(graph: Graph) -> Mapping[str, str]:
    """Returns ``{output: input}`` tensor pairs that may share one buffer.

    A pair is fused only when it is provably safe:

    * the op is elementwise and 1:1 (``INPLACE_OPS``),
    * both tensors have the same dtype and the same byte footprint,
    * the input is scratch memory (never a graph input, which is owned by the
      caller, and never a graph output),
    * nothing else consumes the input tensor.
    """
    consumers: Counter[str] = Counter()
    for node in graph.nodes:
        for name in node.inputs:
            consumers[name] += 1

    aliases: dict[str, str] = {}
    for node in graph.nodes:
        if node.op_type not in INPLACE_OPS:
            continue
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            continue

        source, target = node.inputs[0], node.outputs[0]
        if source in graph.inputs or source in graph.outputs:
            continue
        if target in graph.inputs or target in graph.outputs:
            continue
        if consumers[source] != 1:
            continue
        if source not in graph.tensors or target not in graph.tensors:
            continue

        source_desc, target_desc = graph.tensors[source], graph.tensors[target]
        if source_desc.dtype is not target_desc.dtype:
            continue
        if source_desc.size_bytes != target_desc.size_bytes:
            continue

        aliases[target] = source

    return aliases


def plan_graph_memory(
    graph: Graph,
    *,
    alignment: int = DEFAULT_ALIGNMENT,
) -> MemoryPlan:
    """Plans the arena exactly as ``emit_c`` lays it out.

    Two backend-wide reductions are applied on top of the First-Fit allocator:

    * elementwise ``LUT`` outputs reuse their input buffer (``aliases``),
    * graph inputs are read in place from the caller's ``input_bytes`` buffer
      (no node ever writes an input tensor), so they cost zero arena bytes.
    """
    aliases = plan_inplace_aliases(graph)
    liveness = calculate_liveness(graph.to_op_steps(), graph_outputs=graph.outputs)
    arena_tensors = {
        name: info
        for name, info in graph.to_tensor_infos().items()
        if name not in graph.inputs
    }
    arena_liveness = {
        name: bounds
        for name, bounds in liveness.items()
        if name not in graph.inputs
    }
    return allocate_arena(
        arena_tensors,
        arena_liveness,
        alignment=alignment,
        aliases=aliases,
    )
