from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from kiln.backend.c99.emit_c import emit_c
from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc
from kiln.ref.ref import run_ref


def _has_gcc() -> bool:
    return shutil.which("gcc") is not None


@pytest.mark.skipif(not _has_gcc(), reason="gcc compiler is required for C99 E2E diff tests")
class TestE2EDiff:
    """
    Ultimate End-to-End bit-exact differential testing suite:
    Validates that the emitted C99 code compiled with `gcc -O3 -Wall -Werror`
    matches the Python reference implementation bit-for-bit across various network architectures.
    """

    def _compile_and_run(self, c_code: str, test_inputs: list[list[int]]) -> list[int]:
        """Compiles C code with -O3 -Wall -Werror and executes the test inputs."""
        input_len = len(test_inputs[0])

        main_c = c_code + f"""
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
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / "model.c"
            exe_path = Path(tmpdir) / "model.exe"
            src_path.write_text(main_c, encoding="utf-8")

            compile_cmd = ["gcc", "-O3", "-Wall", "-Werror", str(src_path), "-o", str(exe_path)]
            build_res = subprocess.run(compile_cmd, capture_output=True, text=True)
            if build_res.returncode != 0:
                pytest.fail(f"C compilation failed:\nSTDOUT:\n{build_res.stdout}\nSTDERR:\n{build_res.stderr}")

            predictions: list[int] = []
            for inp in test_inputs:
                run_cmd = [str(exe_path)] + [str(x) for x in inp]
                run_res = subprocess.run(run_cmd, capture_output=True, text=True, check=True)
                predictions.append(int(run_res.stdout.strip()))

            return predictions

    def test_all_seven_ops_pipeline(self) -> None:
        """
        Tests a full end-to-end network featuring all 7 Kiln IR operations:
        EMBEDDING -> CONV1D -> LUT -> ADD -> MATMUL -> MAXPOOL1D -> ARGMAX
        """
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

        w_emb = [[(i * 3 + j * 5) % 30 - 15 for j in range(8)] for i in range(256)]
        w_conv = [[[(oc * 7 + ic * 3 + k) % 15 - 7 for k in range(3)] for ic in range(2)] for oc in range(2)]
        lut_table = [((i - 128) // 3) for i in range(256)]
        w_fc = [[(o * 11 + f * 7) % 20 - 10 for f in range(8)] for o in range(4)]

        weights = {"w_emb": w_emb, "w_conv": w_conv, "w_fc": w_fc}

        nodes = [
            Node(OpType.EMBEDDING, "op_emb", inputs=["in"], outputs=["emb"], attrs={"weight": "w_emb"}),
            Node(OpType.CONV1D, "op_conv", inputs=["emb"], outputs=["conv"], attrs={"weight": "w_conv", "stride": 1, "padding": 1}),
            Node(OpType.LUT, "op_lut", inputs=["conv"], outputs=["lut"], attrs={"table": lut_table}),
            Node(OpType.ADD, "op_add", inputs=["conv", "lut"], outputs=["sum"]),
            Node(OpType.MATMUL, "op_fc", inputs=["sum"], outputs=["fc"], attrs={"weight": "w_fc"}),
            Node(OpType.MAXPOOL1D, "op_pool", inputs=["fc"], outputs=["pool"], attrs={"kernel_size": 4, "stride": 4, "padding": 0}),
            Node(OpType.ARGMAX, "op_argmax", inputs=["pool"], outputs=["out"]),
        ]

        graph = Graph(inputs=["in"], outputs=["out"], nodes=nodes, tensors=tensors)
        c_code = emit_c(graph, weights, runtime_header=None)

        rng = random.Random(42)
        test_inputs = [[rng.randint(0, 255), rng.randint(0, 255)] for _ in range(15)]

        c_preds = self._compile_and_run(c_code, test_inputs)
        ref_preds = [run_ref(graph, {"in": inp}, weights)["out"][0] for inp in test_inputs]

        assert c_preds == ref_preds
        assert set(c_preds) == {0, 1}

    def test_mlp_classifier(self) -> None:
        """
        Tests an MLP classifier topology:
        EMBEDDING -> MATMUL -> LUT (activation) -> MATMUL -> ARGMAX
        """
        qp1 = QuantParam(multiplier=1200000000, shift=2)
        qp2 = QuantParam(multiplier=1073741824, shift=1)

        tensors = {
            "tokens": TensorDesc("tokens", shape=(3,), dtype=DType.UINT8),
            "emb": TensorDesc("emb", shape=(3, 6), dtype=DType.INT8),
            "hidden": TensorDesc("hidden", shape=(3, 8), dtype=DType.INT8, quant=qp1),
            "act": TensorDesc("act", shape=(3, 8), dtype=DType.INT8),
            "logits": TensorDesc("logits", shape=(3, 5), dtype=DType.INT8, quant=qp2),
            "pred": TensorDesc("pred", shape=(), dtype=DType.INT8),
        }

        rng = random.Random(999)
        w_emb = [[rng.randint(-20, 20) for _ in range(6)] for _ in range(256)]
        w_fc1 = [[rng.randint(-15, 15) for _ in range(6)] for _ in range(8)]
        lut_relu = [max(0, i - 128) for i in range(256)]
        w_fc2 = [[rng.randint(-15, 15) for _ in range(8)] for _ in range(5)]

        weights = {"w_emb": w_emb, "w_fc1": w_fc1, "w_fc2": w_fc2}

        nodes = [
            Node(OpType.EMBEDDING, "n_emb", inputs=["tokens"], outputs=["emb"], attrs={"weight": "w_emb"}),
            Node(OpType.MATMUL, "n_fc1", inputs=["emb"], outputs=["hidden"], attrs={"weight": "w_fc1"}),
            Node(OpType.LUT, "n_lut", inputs=["hidden"], outputs=["act"], attrs={"table": lut_relu}),
            Node(OpType.MATMUL, "n_fc2", inputs=["act"], outputs=["logits"], attrs={"weight": "w_fc2"}),
            Node(OpType.ARGMAX, "n_argmax", inputs=["logits"], outputs=["pred"]),
        ]

        graph = Graph(inputs=["tokens"], outputs=["pred"], nodes=nodes, tensors=tensors)
        c_code = emit_c(graph, weights, runtime_header=None)

        test_inputs = [[rng.randint(0, 255) for _ in range(3)] for _ in range(10)]
        c_preds = self._compile_and_run(c_code, test_inputs)
        ref_preds = [run_ref(graph, {"tokens": inp}, weights)["pred"][0] for inp in test_inputs]

        assert c_preds == ref_preds

    def test_cnn_strided_maxpool(self) -> None:
        """
        Tests a 1D-CNN with non-trivial padding, stride, and maxpool:
        EMBEDDING -> CONV1D(stride=2, padding=1) -> MAXPOOL1D(stride=2) -> ARGMAX
        """
        qp = QuantParam(multiplier=1150000000, shift=2)

        tensors = {
            "in": TensorDesc("in", shape=(1,), dtype=DType.UINT8),
            "emb": TensorDesc("emb", shape=(1, 10), dtype=DType.INT8),
            "conv": TensorDesc("conv", shape=(3, 5), dtype=DType.INT8, quant=qp),
            "pool": TensorDesc("pool", shape=(3, 2), dtype=DType.INT8),
            "out": TensorDesc("out", shape=(), dtype=DType.INT8),
        }

        rng = random.Random(777)
        w_emb = [[rng.randint(-30, 30) for _ in range(10)] for _ in range(256)]
        w_conv = [[[rng.randint(-20, 20) for _ in range(3)] for _ in range(1)] for _ in range(3)]
        weights = {"w_emb": w_emb, "w_conv": w_conv}

        nodes = [
            Node(OpType.EMBEDDING, "emb_node", inputs=["in"], outputs=["emb"], attrs={"weight": "w_emb"}),
            Node(OpType.CONV1D, "conv_node", inputs=["emb"], outputs=["conv"], attrs={"weight": "w_conv", "stride": 2, "padding": 1}),
            Node(OpType.MAXPOOL1D, "pool_node", inputs=["conv"], outputs=["pool"], attrs={"kernel_size": 3, "stride": 2, "padding": 0}),
            Node(OpType.ARGMAX, "argmax_node", inputs=["pool"], outputs=["out"]),
        ]

        graph = Graph(inputs=["in"], outputs=["out"], nodes=nodes, tensors=tensors)
        c_code = emit_c(graph, weights, runtime_header=None)

        test_inputs = [[i] for i in range(0, 256, 17)]
        c_preds = self._compile_and_run(c_code, test_inputs)
        ref_preds = [run_ref(graph, {"in": inp}, weights)["out"][0] for inp in test_inputs]

        assert c_preds == ref_preds