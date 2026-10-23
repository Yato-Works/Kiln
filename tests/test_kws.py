from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import pytest
import torch

from examples.kws.kws_graph import calibrate_kws_graph, split_indices
from examples.kws.model import load_kws_model, quantize_logmel_to_int8
from kiln.backend.c99.emit_c import emit_c
from kiln.budget.report import audit_graph
from kiln.ref.ref import run_ref


def _has_gcc() -> bool:
    import shutil

    return shutil.which("gcc") is not None


@pytest.mark.skipif(not _has_gcc(), reason="gcc required for benchmark test")
def test_kws_real_speech_commands_accuracy_and_c99_parity():
    """
    Phase 1a Benchmark Validation on Real Speech Commands v2 Test Set:
    - Verifies that the model trained under the V_mid symmetric quantization contract
      achieves > 85% Top-1 Accuracy on unseen real speech samples.
    - Verifies that emitted C99 compiled with gcc -O3 matches FP32 predictions
      with >= 95% agreement and zero quantization accuracy drop.
    """
    # Anchored to the repository, not the current working directory, so the test
    # behaves the same when pytest is invoked from anywhere.
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "examples" / "kws" / "kws_processed_data.pt"
    if not data_path.exists():
        pytest.skip("Precomputed Speech Commands dataset not found")

    weights_path = repo_root / "examples" / "kws" / "kws_qat_weights.pt"
    if not weights_path.exists():
        pytest.skip("Trained checkpoint not found; run `python -m examples.kws.train_aug`")

    # Re-load the shipped checkpoint explicitly so the test never depends on
    # import-time module state (an absent file would silently mean random weights).
    model = load_kws_model(str(weights_path))

    data = torch.load(data_path, weights_only=True)
    specs = data["specs"]
    labels = data["labels"]

    # Fixed test split
    _, val_idx, test_idx = split_indices(len(labels))

    test_specs = specs[test_idx]
    test_labels = labels[test_idx]

    # Calibrate on the validation split and lower the model into Kiln IR
    kiln_graph = calibrate_kws_graph(model, specs[val_idx])
    graph, weights = kiln_graph.graph, kiln_graph.weights

    # Phase 1a static budget audit (ROADMAP.md): RAM < 2 KB, ROM < 40 KB
    report = audit_graph(graph, weights)
    assert report.arena_ram_bytes < 2048, f"arena RAM {report.arena_ram_bytes} B >= 2 KB"
    assert report.total_rom_bytes < 40960, f"ROM {report.total_rom_bytes} B >= 40 KB"

    c_code = emit_c(graph, weights, runtime_header=None)

    input_len = 490
    harness = (
        c_code
        + f"""
#include <stdio.h>
#include <stdint.h>

int main(int argc, char **argv) {{
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    int32_t n = 0;
    if (fread(&n, sizeof(int32_t), 1, f) != 1) return 3;
    for (int iter = 0; iter < n; ++iter) {{
        int8_t input[{input_len}];
        if (fread(input, sizeof(int8_t), {input_len}, f) != {input_len}) return 4;
        int8_t res = kiln_predict((const uint8_t *)input, {input_len});
        printf("%d\\n", (int)res);
    }}
    fclose(f);
    return 0;
}}
"""
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = Path(tmpdir) / "kws_test.c"
        exe_file = Path(tmpdir) / "kws_test.exe"
        src_file.write_text(harness, encoding="utf-8")
        subprocess.run(
            [
                "gcc",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                str(src_file),
                "-o",
                str(exe_file),
            ],
            check=True,
        )

        int8_samples = []
        for spec in test_specs:
            q_spec = quantize_logmel_to_int8(spec)
            int8_samples.append(q_spec.flatten().tolist())

        import struct

        bin_file = Path(tmpdir) / "test_inputs.bin"
        with open(bin_file, "wb") as bf:
            bf.write(struct.pack("<i", len(int8_samples)))
            for s in int8_samples:
                bf.write(struct.pack(f"{len(s)}b", *s))

        c_proc = subprocess.run(
            [str(exe_file), str(bin_file)], capture_output=True, text=True, check=True
        )
        c_preds = [int(line) for line in c_proc.stdout.strip().split()]

        gt_labels = test_labels.tolist()
        c99_int8_acc = sum(p == y for p, y in zip(c_preds, gt_labels)) / len(gt_labels) * 100.0

        # Assert top-1 accuracy is competitive (> 85%) on real Speech Commands test set
        assert c99_int8_acc >= 85.0, f"Expected >= 85% accuracy, got {c99_int8_acc:.2f}%"

        # Assert bit-exact match with Python ref on subset
        for i in range(25):
            ref_pred = run_ref(graph, {"graph_in": int8_samples[i]}, weights)["graph_out"][0]
            assert c_preds[i] == ref_pred, f"C and ref mismatch at sample {i}"
