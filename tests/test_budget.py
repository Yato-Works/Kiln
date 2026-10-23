from __future__ import annotations

import pytest
from kiln.budget.report import audit_graph, format_budget_certificate
from kiln.backend.c99.emit_c import emit_c
from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc


def test_budget_report_calculations() -> None:
    qp = QuantParam(multiplier=1073741824, shift=1)

    tensors = {
        "in": TensorDesc("in", shape=(2,), dtype=DType.UINT8),
        "emb": TensorDesc("emb", shape=(2, 8), dtype=DType.INT8),
        "conv": TensorDesc("conv", shape=(2, 8), dtype=DType.INT8, quant=qp),
        "lut": TensorDesc("lut", shape=(2, 8), dtype=DType.INT8),
        "sum": TensorDesc("sum", shape=(2, 8), dtype=DType.INT8, quant=qp),
        "fc": TensorDesc("fc", shape=(2, 4), dtype=DType.INT8, quant=qp),
        "pool": TensorDesc("pool", shape=(2, 1), dtype=DType.INT8),
        "out": TensorDesc("out", shape=(), dtype=DType.INT8),
    }

    w_emb = [[1 for _ in range(8)] for _ in range(256)]  # 256 * 8 = 2048 bytes
    w_conv = [[[1 for _ in range(3)] for _ in range(2)] for _ in range(2)]  # 2 * 2 * 3 = 12 bytes
    lut_table = [0 for _ in range(256)]  # 256 bytes
    w_fc = [[1 for _ in range(8)] for _ in range(4)]  # 4 * 8 = 32 bytes
    weights = {"w_emb": w_emb, "w_conv": w_conv, "w_fc": w_fc}

    nodes = [
        Node(OpType.EMBEDDING, "op_emb", inputs=["in"], outputs=["emb"], attrs={"weight": "w_emb"}),
        Node(
            OpType.CONV1D,
            "op_conv",
            inputs=["emb"],
            outputs=["conv"],
            attrs={"weight": "w_conv", "stride": 1, "padding": 1},
        ),
        Node(OpType.LUT, "op_lut", inputs=["conv"], outputs=["lut"], attrs={"table": lut_table}),
        Node(OpType.ADD, "op_add", inputs=["conv", "lut"], outputs=["sum"]),
        Node(OpType.MATMUL, "op_fc", inputs=["sum"], outputs=["fc"], attrs={"weight": "w_fc"}),
        Node(
            OpType.MAXPOOL1D,
            "op_pool",
            inputs=["fc"],
            outputs=["pool"],
            attrs={"kernel_size": 4, "stride": 4, "padding": 0},
        ),
        Node(OpType.ARGMAX, "op_argmax", inputs=["pool"], outputs=["out"]),
    ]

    graph = Graph(inputs=["in"], outputs=["out"], nodes=nodes, tensors=tensors)
    report = audit_graph(graph, weights, alignment=4)

    # Weights ROM: 2048 + 12 + 32 = 2092 bytes
    assert report.weights_rom_bytes == 2092
    # LUT ROM: 1 LUT = 256 bytes
    assert report.lut_rom_bytes == 256
    assert report.total_rom_bytes == 2092 + 256

    # MACs:
    # CONV1D: out_channels(2) * out_length(8) * in_channels(2) * kernel(3) = 96
    # MATMUL: rows(2) * out_features(4) * in_features(8) = 64
    # Total MACs = 96 + 64 = 160
    assert report.total_macs == 160

    # Total Adds:
    # ADD: 2 * 8 = 16
    assert report.total_adds == 16

    # Estimated cycles > 0
    assert report.estimated_worst_cycles > 0
    # Peak concurrent bytes > 0 and <= arena bytes
    assert report.peak_concurrent_bytes <= report.arena_ram_bytes

    # Formatted certificate
    cert = format_budget_certificate(report)
    assert "Kiln Model Static Budget Certificate" in cert
    assert "160 ops" in cert
    assert "strictly zero-malloc guarantee" in cert
    assert "Accumulator Safety" in cert

    # Interval Boundary Propagation audit is attached to the report
    assert report.interval_audit is not None
    assert report.interval_audit.int32_overflow_free is True
    assert report.interval_audit.total_accum_ops == 3  # CONV1D + ADD + MATMUL


def test_emit_c_contains_certificate() -> None:
    tensors = {
        "in": TensorDesc("in", shape=(2,), dtype=DType.UINT8),
        "emb": TensorDesc("emb", shape=(2, 4), dtype=DType.INT8),
        "out": TensorDesc("out", shape=(), dtype=DType.INT8),
    }
    w_emb = [[0 for _ in range(4)] for _ in range(256)]
    nodes = [
        Node(OpType.EMBEDDING, "op_emb", inputs=["in"], outputs=["emb"], attrs={"weight": "w_emb"}),
        Node(OpType.ARGMAX, "op_argmax", inputs=["emb"], outputs=["out"]),
    ]
    graph = Graph(inputs=["in"], outputs=["out"], nodes=nodes, tensors=tensors)
    c_code = emit_c(graph, {"w_emb": w_emb})
    assert "Kiln Model Static Budget Certificate" in c_code
