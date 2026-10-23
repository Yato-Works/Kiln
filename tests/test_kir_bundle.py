from __future__ import annotations

from pathlib import Path

from kiln.cli import main
from kiln.ir.kir import DType, Graph, Node, OpType, TensorDesc
from kiln.ir.serialization import bundle_from_dict, bundle_to_dict, dumps_bundle, loads_bundle


def _bundle_graph() -> Graph:
    return Graph(
        inputs=["x"],
        outputs=["label"],
        tensors={
            "x": TensorDesc("x", (2,), DType.INT8),
            "scores": TensorDesc("scores", (2,), DType.INT8),
            "label": TensorDesc("label", (), DType.INT8),
        },
        nodes=[
            Node(OpType.LUT, "lut", ["x"], ["scores"], {"table": [0] * 256}),
            Node(OpType.ARGMAX, "argmax", ["scores"], ["label"]),
        ],
    )


def test_bundle_round_trip_preserves_graph_and_weights() -> None:
    graph = _bundle_graph()
    weights = {"unused": [[-1, 0, 1]]}
    document = bundle_to_dict(graph, weights)
    assert document["schema"] == "kiln.kir.bundle"
    assert bundle_from_dict(document) == (graph, weights)
    assert loads_bundle(dumps_bundle(graph, weights)) == (graph, weights)


def test_compile_kir_cli_consumes_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "model.kir.json"
    output = tmp_path / "model.c"
    bundle.write_text(dumps_bundle(_bundle_graph(), {}), encoding="utf-8")

    assert main(["compile-kir", str(bundle), "-o", str(output), "--runtime-header", "None"]) == 0
    assert output.is_file()
    assert "kiln_predict" in output.read_text(encoding="utf-8")


def test_validate_and_inspect_kir_cli(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "model.kir.json"
    bundle.write_text(dumps_bundle(_bundle_graph(), {}), encoding="utf-8")

    assert main(["validate", str(bundle)]) == 0
    assert "[VALID]" in capsys.readouterr().out

    assert main(["inspect", str(bundle), "--json"]) == 0
    output = capsys.readouterr().out
    assert '"arena_ram_bytes"' in output
    assert '"int32_overflow_free": true' in output
