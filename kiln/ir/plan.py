"""Live ranges and arena planning for one Kiln graph.



``calculate_liveness`` turns a scheduled graph into per-tensor live intervals;

``allocate_arena`` turns those intervals plus the tensor sizes into the arena

offsets the emitted C indexes.  Placement itself lives in ``kiln.ir.arena``

(historical First-Fit baseline plus a best-fit search that can only shrink the

arena), and ``kiln.ir.memory`` layers the two graph-wide reductions -- in-place

``LUT`` fusion and zero-copy graph inputs -- on top of it.

"""


from __future__ import annotations




from dataclasses import dataclass

from types import MappingProxyType

from typing import Final, Mapping, NamedTuple, Sequence
from . import arena




__all__ = (

    "TensorInfo",

    "OpStep",

    "Lifetime",

    "MemoryPlan",

    "calculate_liveness",

    "allocate_arena",

)



DEFAULT_ALIGNMENT: Final = 4





@dataclass(frozen=True, slots=True)

class TensorInfo:

    name: str

    size_bytes: int





@dataclass(frozen=True, slots=True)

class OpStep:

    op_id: int

    inputs: Sequence[str]

    outputs: Sequence[str]





class Lifetime(NamedTuple):

    first_op: int

    last_op: int





@dataclass(frozen=True, slots=True)

class MemoryPlan:

    total_arena_bytes: int

    offsets: Mapping[str, int]

    peak_concurrent_bytes: int
    #: Allocator strategy that produced ``offsets`` (see ``kiln.ir.arena``).

    strategy: str = "insertion-order/first-fit"






def calculate_liveness(
    ops: Sequence[OpStep],
    *,
    graph_outputs: Sequence[str] | None = None,
) -> dict[str, Lifetime]:
    if not ops:
        if graph_outputs:
            raise KeyError(f"unknown graph output: {graph_outputs[0]}")
        return {}

    seen_op_ids: set[int] = set()
    defined_outputs: set[str] = set()

    for step in ops:
        if step.op_id in seen_op_ids:
            raise ValueError(f"duplicate op_id: {step.op_id}")
        seen_op_ids.add(step.op_id)

        for out_name in step.outputs:
            if out_name in defined_outputs:
                raise ValueError(f"multiple definitions of output tensor: {out_name}")
            defined_outputs.add(out_name)

    schedule = sorted(ops, key=lambda step: step.op_id)
    first_op = schedule[0].op_id
    final_op = schedule[-1].op_id

    first: dict[str, int] = {}
    last: dict[str, int] = {}

    for step in schedule:
        for name in step.inputs:
            first.setdefault(name, first_op)
            last[name] = step.op_id

        for name in step.outputs:
            first.setdefault(name, step.op_id)
            last.setdefault(name, final_op)

    if graph_outputs is not None:
        for out in graph_outputs:
            if out not in first:
                raise KeyError(f"unknown graph output: {out}")
            last[out] = final_op

    return {
        name: Lifetime(first[name], last[name])
        for name in first
    }






def _align_up(value: int, alignment: int) -> int:

    return -(-value // alignment) * alignment





def _placement_order(

    tensors: Mapping[str, TensorInfo],

    lifetimes: Mapping[str, Lifetime | tuple[int, int]],

    alignment: int,

) -> list[tuple[str, Lifetime, int]]:

    requests: list[tuple[str, Lifetime, int]] = []



    for name, bounds in lifetimes.items():

        info = tensors.get(name)



        if info is None:

            raise KeyError(f"unknown tensor: {name}")



        lifetime = Lifetime(*bounds)



        if lifetime.first_op < 0:

            raise ValueError(f"invalid lifetime: {name}")



        if lifetime.last_op < lifetime.first_op:

            raise ValueError(f"invalid lifetime: {name}")



        size = _align_up(info.size_bytes, alignment)

        requests.append((name, lifetime, size))



    requests.sort(key=lambda item: (item[1].first_op, item[0]))



    return requests





def _resolve_aliases(

    aliases: Mapping[str, str] | None,

    lifetimes: Mapping[str, Lifetime | tuple[int, int]],

) -> dict[str, str]:

    """Maps every aliased tensor name to the root tensor owning its bytes."""

    if not aliases:

        return {}



    chains: dict[str, str] = {}

    for target, source in aliases.items():

        for name in (target, source):

            if name not in lifetimes:

                raise KeyError(f"unknown tensor: {name}")



        if target == source:

            raise ValueError(f"self alias: {target}")



        chains[target] = source



    roots: dict[str, str] = {}

    for target in chains:

        chain = [target]

        node = chains[target]

        while node in chains and node not in chain:

            chain.append(node)

            node = chains[node]



        if node in chain:

            raise ValueError(f"cyclic alias: {target}")



        for member in chain:

            roots[member] = node



    return roots





def _group_requests(

    requests: Sequence[tuple[str, Lifetime, int]],

    root_of: Mapping[str, str],

) -> list[tuple[str, Lifetime, int]]:

    """Merges aliased tensors into one request per physical buffer."""

    if not root_of:

        return list(requests)



    merged: dict[str, tuple[Lifetime, int]] = {}

    for name, lifetime, size in requests:

        root = root_of.get(name, name)

        current = merged.get(root)

        if current is None:

            merged[root] = (lifetime, size)

            continue



        previous, previous_size = current

        merged[root] = (

            Lifetime(

                min(previous.first_op, lifetime.first_op),

                max(previous.last_op, lifetime.last_op),

            ),

            max(previous_size, size),

        )



    return sorted(

        ((root, lifetime, size) for root, (lifetime, size) in merged.items()),

        key=lambda item: (item[1].first_op, item[0]),

    )





def allocate_arena(

    tensors: Mapping[str, TensorInfo],

    lifetimes: Mapping[str, Lifetime | tuple[int, int]],

    alignment: int = DEFAULT_ALIGNMENT,

    aliases: Mapping[str, str] | None = None,

) -> MemoryPlan:

    """Places every tensor inside one arena.



    ``aliases`` maps an output tensor to an input tensor whose bytes it may

    safely reuse *in place* (elementwise ops with identical sizes only).  An

    aliased pair shares one block sized by the largest member and spanning the

    union of their lifetimes, so an in-place ``LUT`` costs zero extra RAM.

    ``aliases=None`` reproduces the unaliased layout exactly.



    The offsets come from ``kiln.ir.arena.place``: the historical First-Fit

    layout is returned unless a size-descending / best-fit strategy strictly

    shrinks the arena, so this call can only ever match or beat the layout the

    Phase 1 certificate was taken from.  ``MemoryPlan.strategy`` names the

    winning layout.

    """

    if alignment <= 0:

        raise ValueError(f"invalid alignment: {alignment}")



    for tensor in tensors.values():

        if tensor.size_bytes < 0:

            raise ValueError(f"negative tensor size: {tensor.name}")



    root_of = _resolve_aliases(aliases, lifetimes)

    requests = _group_requests(

        _placement_order(tensors, lifetimes, alignment),

        root_of,

    )



    layout = arena.place(

        [

            (name, (lifetime.first_op, lifetime.last_op), size)

            for name, lifetime, size in requests

        ]

    )



    offsets = dict(layout.offsets)

    for member, root in root_of.items():

        offsets[member] = offsets[root]



    return MemoryPlan(

        total_arena_bytes=layout.arena_bytes,

        offsets=MappingProxyType(offsets),

        peak_concurrent_bytes=layout.peak_concurrent_bytes,

        strategy=layout.strategy,

    )
