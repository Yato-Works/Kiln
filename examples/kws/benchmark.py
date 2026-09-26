"""Shared Phase 1a benchmark plumbing (dataset, emitted-C execution, accuracy).

``evaluate.py``, ``ptq_test.py``, ``qat_eval.py`` and ``tests/test_kws.py`` all
need the same three things: the frozen train/val/test split, a way to run the
*emitted* C artifact over int8 samples, and a top-1 accuracy helper.  Keeping
that here means the compiler output is exercised through exactly one code path,
so a regression in ``emit_c`` cannot hide behind a stale copy in a script.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from examples.kws.kws_graph import split_indices
from examples.kws.model import quantize_logmel_to_int8
from kiln.backend.c99.emit_c import emit_c
from kiln.ir.kir import Graph

__all__ = (
    "DATA_PATH",
    "INPUT_LENGTH",
    "int8_samples",
    "load_splits",
    "predict_c",
    "predict_pytorch",
    "top1_accuracy",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "kws_processed_data.pt")

#: 10 mel bins x 49 frames, i.e. the graph input shape.
INPUT_LENGTH = 10 * 49

#: Harness appended to the emitted translation unit: reads a length-prefixed
#: stream of int8 frames and prints one prediction per line.
_HARNESS = """
#include <stdio.h>
#include <stdint.h>

int main(int argc, char **argv) {{
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    int32_t count = 0;
    if (fread(&count, sizeof(int32_t), 1, f) != 1) return 3;
    for (int32_t iter = 0; iter < count; ++iter) {{
        int8_t input[{length}];
        if (fread(input, sizeof(int8_t), {length}, f) != {length}) return 4;
        int8_t prediction = kiln_predict((const uint8_t *)input, {length});
        printf("%d\\n", (int)prediction);
    }}
    fclose(f);
    return 0;
}}
"""

#: gcc flags used by every benchmark: the emitted C must be warning-clean C99.
GCC_FLAGS = ("-O3", "-Wall", "-Wextra", "-Werror", "-pedantic")


def load_splits() -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Loads the precomputed corpus and the frozen (train, val, test) split."""
    data = torch.load(DATA_PATH, weights_only=True)
    specs, labels = data["specs"], data["labels"]
    return specs, labels, split_indices(len(labels))


def int8_samples(specs: torch.Tensor) -> list[list[int]]:
    """Quantizes log-mel dB spectrograms exactly as the C frontend will."""
    return [quantize_logmel_to_int8(spec).flatten().tolist() for spec in specs]


def predict_pytorch(model: nn.Module, specs: torch.Tensor) -> list[int]:
    """Reference FP32 predictions on the same int8-quantized inputs."""
    quantized = quantize_logmel_to_int8(specs).float() / 127.0
    with torch.no_grad():
        logits = model(quantized).squeeze(-1)
    return torch.argmax(logits, dim=-1).tolist()


def predict_c(
    graph: Graph,
    weights: Mapping[str, Any],
    samples: Sequence[Sequence[int]],
) -> list[int]:
    """Compiles the emitted C with gcc and returns one prediction per sample."""
    harness = emit_c(graph, weights, runtime_header=None) + _HARNESS.format(length=INPUT_LENGTH)

    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "kiln_benchmark.c"
        executable = Path(tmpdir) / "kiln_benchmark.exe"
        source.write_text(harness, encoding="utf-8")
        compile_result = subprocess.run(
            ["gcc", *GCC_FLAGS, str(source), "-o", str(executable)],
            capture_output=True,
            text=True,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(f"emitted C failed to compile:\n{compile_result.stderr}")

        payload = Path(tmpdir) / "inputs.bin"
        with open(payload, "wb") as handle:
            handle.write(struct.pack("<i", len(samples)))
            for sample in samples:
                handle.write(struct.pack(f"{len(sample)}b", *sample))

        run = subprocess.run([str(executable), str(payload)], capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"C benchmark exited with {run.returncode}: {run.stderr}")
        stdout = run.stdout

    return [int(line) for line in stdout.split()]


def top1_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    """Percentage of matching predictions (``0.0`` for an empty set)."""
    if not len(labels):
        return 0.0
    hits = sum(int(p) == int(y) for p, y in zip(predictions, labels))
    return hits / len(labels) * 100.0
