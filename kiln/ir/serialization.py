"""KIR v1 JSON serialization.

The serialized form is intentionally boring: it is versioned, deterministic,
human-readable, and contains no Python class names.  MessagePack can be added
as a transport later without changing this schema.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .kir import DType, Graph, Node, OpType, QuantParam, TensorDesc

KIR_SCHEMA = "kiln.kir"
KIR_VERSION = 1
KIR_BUNDLE_SCHEMA = "kiln.kir.bundle"
KIR_BUNDLE_VERSION = 1


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"KIR attribute is not JSON serializable: {type(value).__name__}")


def graph_to_dict(graph: Graph) -> dict[str, Any]:
    """Return a KIR v1 schema object and validate it before returning."""
    graph.validate()
    return {
        "schema": KIR_SCHEMA,
        "version": KIR_VERSION,
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "tensors": [
            {
                "name": tensor.name,
                "shape": list(tensor.shape),
                "dtype": tensor.dtype.value,
                "quant": (
                    {"multiplier": tensor.quant.multiplier, "shift": tensor.quant.shift}
                    if tensor.quant is not None
                    else None
                ),
            }
            for tensor in graph.tensors.values()
        ],
        "nodes": [
            {
                "name": node.name,
                "op": node.op_type.value,
                "inputs": list(node.inputs),
                "outputs": list(node.outputs),
                "attrs": _json_value(node.attrs),
            }
            for node in graph.nodes
        ],
    }


def graph_from_dict(document: Mapping[str, Any]) -> Graph:
    """Construct and validate a Graph from a KIR v1 schema object."""
    if document.get("schema") != KIR_SCHEMA or document.get("version") != KIR_VERSION:
        raise ValueError(f"unsupported KIR schema: expected {KIR_SCHEMA!r} version {KIR_VERSION}")
    raw_tensors = document.get("tensors")
    raw_nodes = document.get("nodes")
    if not isinstance(raw_tensors, list) or not isinstance(raw_nodes, list):
        raise ValueError("KIR v1 requires list-valued 'tensors' and 'nodes'")

    tensors: dict[str, TensorDesc] = {}
    for raw in raw_tensors:
        if not isinstance(raw, Mapping):
            raise TypeError("KIR tensor entries must be objects")
        quant = raw.get("quant")
        qp = None
        if quant is not None:
            if not isinstance(quant, Mapping):
                raise TypeError("KIR tensor quant must be an object or null")
            qp = QuantParam(multiplier=int(quant["multiplier"]), shift=int(quant["shift"]))
        name = str(raw["name"])
        tensors[name] = TensorDesc(
            name=name,
            shape=tuple(int(dim) for dim in raw["shape"]),
            dtype=DType(str(raw["dtype"])),
            quant=qp,
        )

    nodes: list[Node] = []
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise TypeError("KIR node entries must be objects")
        attrs = raw.get("attrs", {})
        if not isinstance(attrs, Mapping):
            raise TypeError("KIR node attrs must be an object")
        nodes.append(
            Node(
                op_type=OpType(str(raw["op"])),
                name=str(raw["name"]),
                inputs=tuple(str(item) for item in raw["inputs"]),
                outputs=tuple(str(item) for item in raw["outputs"]),
                attrs=dict(attrs),
            )
        )
    graph = Graph(
        inputs=tuple(str(item) for item in document["inputs"]),
        outputs=tuple(str(item) for item in document["outputs"]),
        nodes=nodes,
        tensors=tensors,
    )
    graph.validate()
    return graph


def dumps_graph(graph: Graph, *, indent: int | None = 2) -> str:
    """Serialize a graph deterministically as UTF-8-compatible JSON text."""
    return json.dumps(graph_to_dict(graph), indent=indent, sort_keys=True, separators=(",", ": "))


def loads_graph(text: str) -> Graph:
    """Deserialize a graph from KIR v1 JSON text."""
    document = json.loads(text)
    if not isinstance(document, Mapping):
        raise ValueError("KIR document must be a JSON object")
    return graph_from_dict(document)


def bundle_to_dict(graph: Graph, weights: Mapping[str, Any]) -> dict[str, Any]:
    """Return a portable KIR graph plus quantized weights bundle."""
    return {
        "schema": KIR_BUNDLE_SCHEMA,
        "version": KIR_BUNDLE_VERSION,
        "graph": graph_to_dict(graph),
        "weights": _json_value(weights),
    }


def bundle_from_dict(document: Mapping[str, Any]) -> tuple[Graph, dict[str, Any]]:
    """Decode and validate a portable KIR bundle."""
    if document.get("schema") != KIR_BUNDLE_SCHEMA or document.get("version") != KIR_BUNDLE_VERSION:
        raise ValueError(
            f"unsupported KIR bundle: expected {KIR_BUNDLE_SCHEMA!r} version {KIR_BUNDLE_VERSION}"
        )
    graph_document = document.get("graph")
    raw_weights = document.get("weights")
    if not isinstance(graph_document, Mapping) or not isinstance(raw_weights, Mapping):
        raise ValueError("KIR bundle requires object-valued 'graph' and 'weights'")
    graph = graph_from_dict(graph_document)
    weights = {str(name): value for name, value in raw_weights.items()}
    return graph, weights


def dumps_bundle(graph: Graph, weights: Mapping[str, Any], *, indent: int | None = 2) -> str:
    """Serialize a complete graph and weights bundle deterministically."""
    return json.dumps(
        bundle_to_dict(graph, weights), indent=indent, sort_keys=True, separators=(",", ": ")
    )


def loads_bundle(text: str) -> tuple[Graph, dict[str, Any]]:
    """Deserialize a complete graph and weights bundle."""
    document = json.loads(text)
    if not isinstance(document, Mapping):
        raise ValueError("KIR bundle must be a JSON object")
    return bundle_from_dict(document)
