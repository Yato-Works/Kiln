from __future__ import annotations

import pytest
from kiln.analysis.interval import (
    Interval,
    IntervalAudit,
    NodeIntervalAudit,
    propagate_intervals,
)
from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc
from kiln.ir.sem import INT8_MAX, INT8_MIN, INT32_MAX, INT32_MIN
from kiln.spec import load_spec
from kiln.spec.loader import resolve_model_graph


def test_interval_basic_properties() -> None:
    inv = Interval(-10, 20)
    assert inv.lo == -10
    assert inv.hi == 20
    assert inv.contains(0)
    assert inv.contains(-10)
    assert inv.contains(20)
    assert not inv.contains(-11)
    assert not inv.contains(21)
    assert inv.fits_int8()
    assert inv.fits_int32()

    with pytest.raises(ValueError, match="invalid interval"):
        Interval(10, -5)


def test_interval_int8_and_int32_bounds() -> None:
    inv8 = Interval(-128, 127)
    assert inv8.fits_int8()
    assert inv8.fits_int32()

    inv_wide = Interval(-129, 127)
    assert not inv_wide.fits_int8()
    assert inv_wide.fits_int32()

    inv32 = Interval(INT32_MIN, INT32_MAX)
    assert not inv32.fits_int8()
    assert inv32.fits_int32()

    inv32_overflow = Interval(INT32_MIN - 1, INT32_MAX)
    assert not inv32_overflow.fits_int32()


def test_propagate_embedding_and_matmul() -> None:
    qp = QuantParam(multiplier=1073741824, shift=1)
    tensors = {
        "in": TensorDesc("in", shape=(2,), dtype=DType.UINT8),
        "emb": TensorDesc("emb", shape=(2, 4), dtype=DType.INT8),
        "logits": TensorDesc("logits", shape=(2, 2), dtype=DType.INT8, quant=qp),
        "out": TensorDesc("out", shape=(), dtype=DType.INT8),
    }

    # Weights: w_emb values range from -10 to +30
    w_emb = [[-10, 5, 20, 30] if i == 0 else [0, 0, 0, 0] for i in range(256)]
    # w_mm has shape (2, 4):
    # row 0: [2, -1, 3, -2]
    # row 1: [1, 1, 1, 1]
    w_mm = [
        [2, -1, 3, -2],
        [1, 1, 1, 1],
    ]
    weights = {"w_emb": w_emb, "w_mm": w_mm}

    nodes = [
        Node(OpType.EMBEDDING, "op_emb", inputs=["in"], outputs=["emb"], attrs={"weight": "w_emb"}),
        Node(OpType.MATMUL, "op_mm", inputs=["emb"], outputs=["logits"], attrs={"weight": "w_mm"}),
        Node(OpType.ARGMAX, "op_argmax", inputs=["logits"], outputs=["out"]),
    ]
    graph = Graph(inputs=["in"], outputs=["out"], nodes=nodes, tensors=tensors)

    audit = propagate_intervals(graph, weights)
    assert audit.int32_overflow_free is True
    assert audit.tensor_intervals["in"] == Interval(0, 255)
    assert audit.tensor_intervals["emb"] == Interval(-10, 30)

    # Calculation for row 0:
    # pos_sum = 2 + 3 = 5
    # neg_sum = -1 + -2 = -3
    # acc_lo = (-10 * 5) + (30 * -3) = -50 - 90 = -140
    # acc_hi = (30 * 5) + (-10 * -3) = 150 + 30 = 180
    # Calculation for row 1:
    # pos_sum = 4, neg_sum = 0
    # acc_lo = -10 * 4 = -40
    # acc_hi = 30 * 4 = 120
    # Overall matmul acc_lo = -140, acc_hi = 180
    na_mm = audit.node_audits["op_mm"]
    assert na_mm.accum_lo == -140
    assert na_mm.accum_hi == 180
    assert na_mm.accum_fits_int32 is True
    # Requantize with multiplier 1073741824, shift 1 (scale 0.25):
    # -140 * 0.25 -> -35, 180 * 0.25 -> 45
    assert audit.tensor_intervals["logits"] == Interval(-35, 45)
    assert na_mm.saturates is False

    # ARGMAX output for 2 classes is [0, 3] (num_elements = 4 for (2, 2) tensor)
    assert audit.tensor_intervals["out"] == Interval(0, 3)


def test_propagate_conv1d_lut_maxpool_and_add() -> None:
    qp = QuantParam(multiplier=1073741824, shift=1)
    # 256-element ReLU table: clamp negative to 0
    lut_relu = [max(0, v) for v in range(-128, 128)]

    tensors = {
        "x": TensorDesc("x", shape=(1, 4), dtype=DType.INT8),
        "conv": TensorDesc("conv", shape=(1, 4), dtype=DType.INT8, quant=qp),
        "act": TensorDesc("act", shape=(1, 4), dtype=DType.INT8),
        "pool": TensorDesc("pool", shape=(1, 2), dtype=DType.INT8),
        "res": TensorDesc("res", shape=(1, 2), dtype=DType.INT8, quant=qp),
    }

    # CONV1D weight: shape (1, 1, 2)
    # filter: [2, -3]
    w_conv = [[[2, -3]]]
    weights = {"w_conv": w_conv}

    nodes = [
        Node(OpType.CONV1D, "c1", inputs=["x"], outputs=["conv"], attrs={"weight": "w_conv", "stride": 1, "padding": 0}),
        Node(OpType.LUT, "l1", inputs=["conv"], outputs=["act"], attrs={"table": lut_relu}),
        Node(OpType.MAXPOOL1D, "p1", inputs=["act"], outputs=["pool"], attrs={"kernel_size": 2, "stride": 2, "padding": 0}),
        Node(OpType.ADD, "a1", inputs=["pool", "pool"], outputs=["res"]),
    ]
    graph = Graph(inputs=["x"], outputs=["res"], nodes=nodes, tensors=tensors)

    # Custom input bounds [-20, 50]
    audit = propagate_intervals(graph, weights, input_intervals={"x": Interval(-20, 50)})
    assert audit.int32_overflow_free is True

    # filter: pos_sum = 2, neg_sum = -3
    # acc_lo = (-20 * 2) + (50 * -3) = -40 - 150 = -190
    # acc_hi = (50 * 2) + (-20 * -3) = 100 + 60 = 160
    assert audit.node_audits["c1"].accum_lo == -190
    assert audit.node_audits["c1"].accum_hi == 160
    # Requantize * 0.25: -190 -> -48, 160 -> 40
    assert audit.tensor_intervals["conv"] == Interval(-48, 40)

    # LUT: ReLU maps [-48, 40] -> [0, 40]
    assert audit.tensor_intervals["act"] == Interval(0, 40)

    # MAXPOOL1D preserves bounds: [0, 40]
    assert audit.tensor_intervals["pool"] == Interval(0, 40)

    # ADD: [0, 40] + [0, 40] -> acc in [0, 80]
    # Requantize * 0.25: 80 -> 20
    assert audit.node_audits["a1"].accum_lo == 0
    assert audit.node_audits["a1"].accum_hi == 80
    assert audit.tensor_intervals["res"] == Interval(0, 20)


def test_propagate_detects_saturation() -> None:
    # Multiplier and shift that scale by 4x: multiplier=1073741824, shift=0 (actually 1073741824 is 0.5 when shift=0)
    # With multiplier 2147483647 (approx 1.0) and shift 0, scale is ~1.0
    qp = QuantParam(multiplier=2147483647, shift=0)
    tensors = {
        "x": TensorDesc("x", shape=(1, 4), dtype=DType.INT8),
        "y": TensorDesc("y", shape=(1, 4), dtype=DType.INT8, quant=qp),
    }
    # ADD [50, 100] + [50, 100] -> accum in [100, 200]
    # preclamp in [100, 200] -> exceeds 127, so it saturates!
    # postclamp is clamped to 127
    nodes = [Node(OpType.ADD, "add_sat", inputs=["x", "x"], outputs=["y"])]
    graph = Graph(inputs=["x"], outputs=["y"], nodes=nodes, tensors=tensors)

    audit = propagate_intervals(graph, {}, input_intervals={"x": Interval(50, 100)})
    assert audit.node_audits["add_sat"].saturates is True
    assert audit.saturating_ops == 1
    assert audit.tensor_intervals["y"].hi == 127


def test_propagate_real_models() -> None:
    # 1. Hazard model
    spec_hazard = load_spec("examples/hazard/kiln.yaml")
    loaded_hazard = resolve_model_graph(spec_hazard)
    audit_h = propagate_intervals(loaded_hazard.graph, loaded_hazard.weights)
    assert audit_h.int32_overflow_free is True
    assert audit_h.total_accum_ops == 1
    assert audit_h.min_accum == -67640
    assert audit_h.max_accum == 61845

    # 2. KWS model
    spec_kws = load_spec("examples/kws/kiln.yaml")
    loaded_kws = resolve_model_graph(spec_kws)
    audit_k = propagate_intervals(loaded_kws.graph, loaded_kws.weights)
    assert audit_k.int32_overflow_free is True
    assert audit_k.total_accum_ops == 5
    assert audit_k.min_accum == -385191
    assert audit_k.max_accum == 220602


def test_propagate_detects_int32_overflow() -> None:
    # Construct a synthetic node that forces 32-bit accumulator overflow
    qp = QuantParam(multiplier=1073741824, shift=1)
    tensors = {
        "x": TensorDesc("x", shape=(1, 100000), dtype=DType.INT8),
        "y": TensorDesc("y", shape=(1, 1), dtype=DType.INT8, quant=qp),
    }
    # 100,000 weights of 127 with input 127 gives accum = 100,000 * 127 * 127 = 1,612,900,000 (fits in int32)
    # But 200,000 weights of 127 with input 127 gives 3,225,800,000 > 2,147,483,647 (overflows int32!)
    count = 200000
    tensors = {
        "x": TensorDesc("x", shape=(1, count), dtype=DType.INT8),
        "y": TensorDesc("y", shape=(1, 1), dtype=DType.INT8, quant=qp),
    }
    w_fc = [[127 for _ in range(count)]]
    weights = {"w_fc": w_fc}
    nodes = [Node(OpType.MATMUL, "fc_overflow", inputs=["x"], outputs=["y"], attrs={"weight": "w_fc"})]
    graph = Graph(inputs=["x"], outputs=["y"], nodes=nodes, tensors=tensors)

    audit = propagate_intervals(graph, weights, input_intervals={"x": Interval(127, 127)})
    assert audit.int32_overflow_free is False
    assert audit.node_audits["fc_overflow"].accum_fits_int32 is False
    assert audit.max_accum == count * 127 * 127
