from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from kiln.backend.c99.emit_c import emit_c
from kiln.frontend.pytorch import export_to_kir, quantize_tensor, compute_multiplier_and_shift
from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc
from kiln.ref.ref import run_ref


def _has_gcc() -> bool:
    return shutil.which("gcc") is not None


class TinyClassifier(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Embedding(256, 8),
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.AdaptiveMaxPool1d(1),
        )


class GeluMlp(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Embedding(256, 6),
            nn.Linear(6, 8),
            nn.GELU(),
            nn.Linear(8, 3),
        )


def _compile_and_predict(c_code: str, test_inputs: list[list[int]]) -> list[int]:
    input_len = len(test_inputs[0])
    harness = (
        c_code
        + f"""
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, char **argv) {{
    if (argc < {input_len + 1}) return 1;
    uint8_t input[{input_len}];
    for (int i = 0; i < {input_len}; ++i) {{
        input[i] = (uint8_t)atoi(argv[i + 1]);
    }}
    int8_t pred = kiln_predict(input, {input_len});
    printf("%d\\n", (int)pred);
    return 0;
}}
"""
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "model.c"
        exe_path = Path(tmpdir) / "model.exe"
        src_path.write_text(harness, encoding="utf-8")

        subprocess.run(
            [
                "gcc",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                str(src_path),
                "-o",
                str(exe_path),
            ],
            check=True,
        )

        results = []
        for inp in test_inputs:
            cmd = [str(exe_path)] + [str(x) for x in inp]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            results.append(int(res.stdout.strip()))
        return results


def test_quantize_tensor_utility():
    t = torch.tensor([[-10.0, 0.0], [5.0, 10.0]], dtype=torch.float32)
    q_data, scale = quantize_tensor(t)
    assert scale == pytest.approx(10.0 / 127.0)
    assert q_data[0][0] == -127
    assert q_data[0][1] == 0
    assert q_data[1][1] == 127


def test_multiplier_shift_computation():
    mult, shift = compute_multiplier_and_shift(0.5)
    assert mult == 1073741824
    assert shift == 0

    mult2, shift2 = compute_multiplier_and_shift(0.125)
    assert mult2 == 1073741824
    assert shift2 == 2


@pytest.mark.skipif(not _has_gcc(), reason="gcc required for E2E PyTorch diff test")
def test_pytorch_conv_e2e_diff():
    torch.manual_seed(42)
    model = TinyClassifier()
    dummy_input = torch.tensor([10, 20], dtype=torch.uint8)

    graph, weights = export_to_kir(model, dummy_input)
    assert len(graph.nodes) == 6
    assert graph.nodes[-1].op_type.value == "argmax"

    c_code = emit_c(graph, weights, runtime_header=None)

    test_samples = [
        [10, 20],
        [0, 0],
        [255, 255],
        [128, 64],
        [42, 189],
    ]

    c_predictions = _compile_and_predict(c_code, test_samples)
    ref_predictions = [
        run_ref(graph, {"graph_in": sample}, weights)["graph_out"][0] for sample in test_samples
    ]

    assert c_predictions == ref_predictions


@pytest.mark.skipif(not _has_gcc(), reason="gcc required for E2E PyTorch diff test")
def test_pytorch_gelu_mlp_e2e_diff():
    torch.manual_seed(123)
    model = GeluMlp()
    dummy_input = torch.tensor([5, 15, 25], dtype=torch.uint8)

    graph, weights = export_to_kir(model, dummy_input)
    c_code = emit_c(graph, weights, runtime_header=None)

    test_samples = [
        [5, 15, 25],
        [0, 1, 2],
        [255, 128, 0],
        [77, 88, 99],
    ]

    c_predictions = _compile_and_predict(c_code, test_samples)
    ref_predictions = [
        run_ref(graph, {"graph_in": sample}, weights)["graph_out"][0] for sample in test_samples
    ]

    assert c_predictions == ref_predictions


def test_compute_multiplier_and_shift_rejects_invalid_scales():
    # 1. Non-finite values
    with pytest.raises(ValueError, match="must be finite"):
        compute_multiplier_and_shift(float("nan"))
    with pytest.raises(ValueError, match="must be finite"):
        compute_multiplier_and_shift(float("inf"))

    # 2. Non-positive values
    with pytest.raises(ValueError, match="strictly positive"):
        compute_multiplier_and_shift(0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_multiplier_and_shift(-0.5)

    # 3. Scale >= 1.0 (unsupported left shift / negative shift)
    with pytest.raises(ValueError, match=">= 1.0 is unsupported"):
        compute_multiplier_and_shift(1.0)
    with pytest.raises(ValueError, match=">= 1.0 is unsupported"):
        compute_multiplier_and_shift(1.5)
    with pytest.raises(ValueError, match=">= 1.0 is unsupported"):
        compute_multiplier_and_shift(2.0)


def test_pytorch_float_vs_quantized_numerical_accuracy():
    """
    Cross-model validation: Compares PyTorch native FP32 forward pass
    against Kiln quantized execution to ensure the quantization scale
    does not saturate or diverge from true float predictions.
    """
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Embedding(256, 16),
        nn.Linear(16, 4, bias=False),
    )
    dummy = torch.tensor([10, 20], dtype=torch.uint8)

    graph, weights = export_to_kir(model, dummy)

    test_inputs = [
        [10, 20],
        [42, 189],
        [77, 88],
        [1, 254],
        [100, 200],
    ]

    for sample in test_inputs:
        with torch.no_grad():
            x = torch.tensor(sample, dtype=torch.long)
            emb = model[0](x)
            out_float = model[1](emb)
            torch_pred = torch.argmax(out_float.flatten()).item()

        kiln_pred = run_ref(graph, {"graph_in": sample}, weights)["graph_out"][0]
        assert torch_pred == kiln_pred, (
            f"Float vs Quantized divergence: Float={torch_pred}, Kiln={kiln_pred}"
        )


def test_argmax_rejects_exceeding_128_elements():
    # 1. 1D shape with 129 elements
    tensors1 = {
        "in": TensorDesc("in", shape=(129,), dtype=DType.INT8),
        "out": TensorDesc("out", shape=(), dtype=DType.INT8),
    }
    graph1 = Graph(
        inputs=["in"],
        outputs=["out"],
        nodes=[Node(OpType.ARGMAX, "argmax", inputs=["in"], outputs=["out"])],
        tensors=tensors1,
    )
    with pytest.raises(ValueError, match="exceed the max index"):
        graph1.validate()

    # 2. Multi-dimensional shape where total elements exceed 128 (e.g. 2 x 65 = 130)
    tensors2 = {
        "in": TensorDesc("in", shape=(2, 65), dtype=DType.INT8),
        "out": TensorDesc("out", shape=(), dtype=DType.INT8),
    }
    graph2 = Graph(
        inputs=["in"],
        outputs=["out"],
        nodes=[Node(OpType.ARGMAX, "argmax", inputs=["in"], outputs=["out"])],
        tensors=tensors2,
    )
    with pytest.raises(ValueError, match="exceed the max index"):
        graph2.validate()

    # 3. Exactly 128 elements is valid (0 to 127)
    tensors3 = {
        "in": TensorDesc("in", shape=(128,), dtype=DType.INT8),
        "out": TensorDesc("out", shape=(), dtype=DType.INT8),
    }
    graph3 = Graph(
        inputs=["in"],
        outputs=["out"],
        nodes=[Node(OpType.ARGMAX, "argmax", inputs=["in"], outputs=["out"])],
        tensors=tensors3,
    )
    graph3.validate()  # should not raise!


@pytest.mark.skipif(not _has_gcc(), reason="gcc required for tie-break test")
def test_argmax_tie_break_earliest_index_c_and_ref():
    """
    Verifies that when multiple classes have identical maximum scores,
    both Python ref.py and emitted C99 strictly return the EARLIEST index.
    """
    # Input has 5 classes: [10, 50, 50, 20, 50]
    # Maximum is 50, appearing at index 1, 2, 4. Earliest index MUST be 1.
    tensors = {
        "in": TensorDesc("in", shape=(1,), dtype=DType.UINT8),
        "emb": TensorDesc("emb", shape=(1, 5), dtype=DType.INT8),
        "out": TensorDesc("out", shape=(), dtype=DType.INT8),
    }
    w_emb = [[10, 50, 50, 20, 50] for _ in range(256)]
    nodes = [
        Node(OpType.EMBEDDING, "op_emb", inputs=["in"], outputs=["emb"], attrs={"weight": "w_emb"}),
        Node(OpType.ARGMAX, "op_argmax", inputs=["emb"], outputs=["out"]),
    ]
    graph = Graph(inputs=["in"], outputs=["out"], nodes=nodes, tensors=tensors)
    c_code = emit_c(graph, {"w_emb": w_emb}, runtime_header=None)

    # 1. Check Python reference
    ref_val = run_ref(graph, {"in": [0]}, {"w_emb": w_emb})["out"][0]
    assert ref_val == 1, f"Expected earliest index 1, got {ref_val}"

    # 2. Check GCC compiled C output
    c_preds = _compile_and_predict(c_code, [[0]])
    assert c_preds[0] == 1, f"Expected C earliest index 1, got {c_preds[0]}"
    assert c_preds[0] == ref_val


@pytest.mark.skipif(not _has_gcc(), reason="gcc required for row-major layout test")
def test_conv1d_direct_int8_row_major_layout_verification():
    """
    Rigorous verification of KIR Specification Section 2.2:
    - Input tensor shape: (C_in=2, L=4), DType: INT8 (no EMBEDDING).
    - Flattened row-major layout: offset(c, t) = c * L + t.
    - Each position has an intentionally unique value:
        channel 0: [10, 20, 30, 40]
        channel 1: [50, 60, 70, 80]
      Row-major flat buffer: [10, 20, 30, 40, 50, 60, 70, 80].
    - Conv1D filter: out_channels=1, in_channels=2, kernel_size=2, stride=1, padding=0.
      Weight:
        channel 0 filter: [1, 0]  (picks only t)
        channel 1 filter: [0, 1]  (picks t+1 from channel 1)
      So for output position op:
        expected_acc = 1 * input[0, op] + 0 * input[0, op+1]
                     + 0 * input[1, op] + 1 * input[1, op+1]
                     = input[0, op] + input[1, op+1]
      Out length = 4 - 2 + 1 = 3.
      op=0: input[0, 0] + input[1, 1] = 10 + 60 = 70.
      op=1: input[0, 1] + input[1, 2] = 20 + 70 = 90.
      op=2: input[0, 2] + input[1, 3] = 30 + 80 = 110.
    - With scale=0.5 (mult=1073741824, shift=0):
      op=0 -> 70, op=1 -> 90, op=2 -> 110.
    - Argmax over [70, 90, 110] MUST return index 2!
      (If axes were transposed, op addresses would be completely scrambled and produce wrong index).
    """
    qp = QuantParam(multiplier=1073741824, shift=0)

    tensors = {
        "raw_in": TensorDesc("raw_in", shape=(2, 4), dtype=DType.INT8),
        "conv_out": TensorDesc("conv_out", shape=(1, 3), dtype=DType.INT8, quant=qp),
        "pred": TensorDesc("pred", shape=(), dtype=DType.INT8),
    }

    w_conv = [
        [
            [1, 0],  # ch 0
            [0, 1],  # ch 1
        ]
    ]
    weights = {"w_conv": w_conv}

    nodes = [
        Node(
            OpType.CONV1D,
            "node_conv",
            inputs=["raw_in"],
            outputs=["conv_out"],
            attrs={"weight": "w_conv", "stride": 1, "padding": 0},
        ),
        Node(OpType.ARGMAX, "node_argmax", inputs=["conv_out"], outputs=["pred"]),
    ]

    graph = Graph(inputs=["raw_in"], outputs=["pred"], nodes=nodes, tensors=tensors)
    c_code = emit_c(graph, weights, runtime_header=None)

    flat_input = [10, 20, 30, 40, 50, 60, 70, 80]

    # Python reference
    ref_out = run_ref(graph, {"raw_in": flat_input}, weights)["pred"][0]
    assert ref_out == 2, f"Expected argmax index 2 (op=2 with val 110), got {ref_out}"

    # C executable verification
    c_preds = _compile_and_predict(c_code, [flat_input])
    assert c_preds[0] == 2, f"Expected C argmax index 2, got {c_preds[0]}"
    assert c_preds[0] == ref_out


class ResidualToy(nn.Module):
    """Residual network with Skip Connection: y = AdaptiveMaxPool1d(relu(conv1(x)) + conv2(x))"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(2, 2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        res = x
        h = self.relu(self.conv1(x))
        h = self.conv2(h)
        h = h + res  # Residual Add!
        return self.pool(h)


@pytest.mark.skipif(not _has_gcc(), reason="gcc required for E2E PyTorch diff test")
def test_pytorch_residual_add_e2e_diff():
    torch.manual_seed(42)
    model = ResidualToy().eval()
    dummy_input = torch.randn(2, 6)

    graph, weights = export_to_kir(model, dummy_input)
    # Check that ADD op is present in the lowered graph
    node_types = [n.op_type for n in graph.nodes]
    assert OpType.ADD in node_types, f"Expected OpType.ADD in {node_types}"
    assert graph.nodes[-1].op_type == OpType.ARGMAX

    c_code = emit_c(graph, weights, runtime_header=None)

    # Test with 8 distinct test vectors
    test_samples = [[((trial * 29 + i * 17) % 256) for i in range(12)] for trial in range(8)]

    c_predictions = _compile_and_predict(c_code, test_samples)
    for sample, c_pred in zip(test_samples, c_predictions):
        int8_vals = [(b if b < 128 else b - 256) for b in sample]
        ref_out = run_ref(graph, {"graph_in": int8_vals}, weights)["graph_out"]
        ref_pred = ref_out[0] if isinstance(ref_out, list) else ref_out
        assert c_pred == ref_pred, f"Bit-exact mismatch: c={c_pred} vs ref={ref_pred}"
