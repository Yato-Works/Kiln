from __future__ import annotations

from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc
from kiln.ir.serialization import dumps_graph, graph_from_dict, graph_to_dict, loads_graph


def _graph() -> Graph:
    tensors = {
        "tokens": TensorDesc("tokens", (4,), DType.UINT8),
        "emb": TensorDesc("emb", (4, 2), DType.INT8),
        "logits": TensorDesc("logits", (4, 2), DType.INT8, QuantParam(1 << 30, 1)),
        "label": TensorDesc("label", (), DType.INT8),
    }
    return Graph(
        inputs=["tokens"],
        outputs=["label"],
        tensors=tensors,
        nodes=[
            Node(OpType.EMBEDDING, "emb", ["tokens"], ["emb"], {"weight": "w"}),
            Node(OpType.MATMUL, "mm", ["emb"], ["logits"], {"weight": "w2"}),
            Node(OpType.ARGMAX, "argmax", ["logits"], ["label"]),
        ],
    )


def test_kir_v1_round_trip_is_deterministic() -> None:
    graph = _graph()
    document = graph_to_dict(graph)
    assert document["schema"] == "kiln.kir"
    assert document["version"] == 1
    assert graph_from_dict(document) == graph
    assert loads_graph(dumps_graph(graph)) == graph
    assert dumps_graph(graph) == dumps_graph(graph)


def test_kir_v1_rejects_unknown_version() -> None:
    document = graph_to_dict(_graph())
    document["version"] = 2
    try:
        graph_from_dict(document)
    except ValueError as error:
        assert "unsupported KIR schema" in str(error)
    else:
        raise AssertionError("unknown KIR version was accepted")
