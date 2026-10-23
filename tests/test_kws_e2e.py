"""Phase 1b end-to-end parity: raw PCM -> fixed-point C frontend -> emitted Kiln graph.

``tests/test_kws_frontend.py`` proves that ``kws_frontend.c`` reproduces the
training-time log-mel reference and that ``kws_predict_raw_pcm`` forwards those
features, but it links a *stub* ``kiln_predict``.  This module links the real
artifact emitted by ``kiln.backend.c99.emit_c`` for the shipped checkpoint, so
the complete device-side path

    16-bit PCM -> kws_frontend.c -> int8 10x49 log-mel -> generated C99 -> class index

is exercised against the PyTorch pipeline on real Speech Commands utterances.

Two independent claims are asserted:

* the generated graph is bit-exact on the features the C frontend actually
  produces (``kiln.ref.run_ref`` vs the compiled graph), and
* the full PCM-to-label path agrees with the PyTorch pipeline.  The frontend may
  differ from the FP32 reference by one int8 step per cell (see
  ``test_kws_frontend.py``), so a disagreement is a near-tie decision flipped by
  that one-step delta; the check is therefore an agreement rate, not equality.
"""

from __future__ import annotations

import array
import io
import shutil
import struct
import subprocess
import tarfile
import tempfile
import wave
from pathlib import Path

import pytest
import torch

from examples.kws.benchmark import DATA_PATH, load_splits
from examples.kws.kws_graph import calibrate_kws_graph
from examples.kws.model import (
    KWS_CLASSES,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    load_kws_model,
    quantize_logmel_to_int8,
)
from kiln.backend.c99.emit_c import emit_c
from kiln.ref.ref import run_ref

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "examples" / "kws" / "frontend"
FRONTEND_SRC = FRONTEND_DIR / "kws_frontend.c"
PIPELINE_SRC = FRONTEND_DIR / "kws_pipeline.c"
CHECKPOINT = REPO_ROOT / "examples" / "kws" / "kws_qat_weights.pt"
SPEECH_COMMANDS_ARCHIVE = REPO_ROOT / "speech_commands_test_set_v0.02.tar.gz"

N_SPEC_VALUES = N_MELS * N_FRAMES
GCC_FLAGS = ("-std=c99", "-O3", "-Wall", "-Wextra", "-Werror", "-pedantic")

#: Clips taken from each of the 12 classes, and the tolerance the PCM path is
#: held to (the frontend may be one int8 step off per cell, so a % of near-tie
#: decisions can flip; see the module docstring).  Both thresholds are measured
#: values with margin, see ``ROADMAP.md`` Phase 1b.
CLIPS_PER_CLASS = 2
MIN_E2E_ACCURACY = 0.75
MIN_AGREEMENT = 0.90

HARNESS = r"""
#include <stdio.h>
#include <stdint.h>
#include "kws_frontend.h"
#include "kws_pipeline.h"

/* Reads: int32 case_count, then per case int32 sample_count + int16 samples.
 * Prints, per case, the predicted class followed by the 10 lines of 49 int8
 * features the frontend produced, so the classifier can be checked separately
 * from the fixed-point frontend. */
int main(int argc, char **argv) {
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    int32_t case_count = 0;
    if (fread(&case_count, sizeof(int32_t), 1, f) != 1) return 3;
    static int16_t pcm[2 * KWS_SAMPLE_RATE];
    for (int32_t c = 0; c < case_count; ++c) {
        int32_t sample_count = 0;
        if (fread(&sample_count, sizeof(int32_t), 1, f) != 1) return 4;
        if (sample_count < 0 || sample_count > 2 * KWS_SAMPLE_RATE) return 5;
        if (sample_count > 0 &&
            fread(pcm, sizeof(int16_t), (size_t)sample_count, f) != (size_t)sample_count) {
            return 6;
        }
        int8_t features[KWS_N_MELS * KWS_N_FRAMES];
        kws_frontend_process(pcm, (size_t)sample_count, features);
        printf("%d\n", (int)kws_predict_raw_pcm(pcm, (size_t)sample_count));
        for (int m = 0; m < KWS_N_MELS; ++m) {
            for (int frame = 0; frame < KWS_N_FRAMES; ++frame) {
                printf("%d%c", (int)features[m * KWS_N_FRAMES + frame],
                       frame + 1 == KWS_N_FRAMES ? '\n' : ' ');
            }
        }
    }
    fclose(f);
    return 0;
}
"""


def _read_wav_pcm(payload: bytes) -> torch.Tensor:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE, "unexpected sample rate in archive"
        assert wav.getsampwidth() == 2, "expected 16-bit PCM in archive"
        assert wav.getnchannels() == 1, "expected mono PCM in archive"
        samples = array.array("h", wav.readframes(wav.getnframes()))
    return torch.tensor(samples, dtype=torch.int16)


def _real_utterances() -> list[tuple[str, int, torch.Tensor]]:
    """Two clips from each of the 12 classes, so the check spans every label."""
    if not SPEECH_COMMANDS_ARCHIVE.exists():
        pytest.skip("Speech Commands archive not found")

    label_index = {name: index for index, name in enumerate(KWS_CLASSES)}
    by_class: dict[str, list[tarfile.TarInfo]] = {}
    with tarfile.open(SPEECH_COMMANDS_ARCHIVE, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".wav"):
                continue
            # Archive directories are "./yes/..." and "./_silence_/...", while
            # KWS_CLASSES uses "yes" and "silence" (see precompute.py's map).
            raw_label = member.name.strip("./").split("/")[0]
            by_class.setdefault(raw_label.strip("_"), []).append(member)
        assert set(by_class) == set(label_index), "unexpected archive labels"

        cases: list[tuple[str, int, torch.Tensor]] = []
        for label, index in sorted(label_index.items(), key=lambda item: item[1]):
            members = sorted(by_class[label], key=lambda member: member.name)
            stride = max(1, len(members) // CLIPS_PER_CLASS)
            for member in members[::stride][:CLIPS_PER_CLASS]:
                handle = tar.extractfile(member)
                assert handle is not None
                cases.append((member.name, index, _read_wav_pcm(handle.read())))
    assert len(cases) == len(label_index) * CLIPS_PER_CLASS, "unexpected archive layout"
    return cases


def _shipped_kiln_graph():
    if not CHECKPOINT.exists():
        pytest.skip("Trained checkpoint not found; run `python -m examples.kws.train_aug`")
    if not Path(DATA_PATH).exists():
        pytest.skip("Precomputed Speech Commands dataset not found")
    specs, _, (_, val_idx, _) = load_splits()
    lowered = calibrate_kws_graph(load_kws_model(str(CHECKPOINT)), specs[val_idx])
    return lowered.graph, lowered.weights


@pytest.fixture(scope="module")
def e2e_exe() -> Path:
    """Builds the fixed-point frontend + emitted classifier + glue into one binary."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc required to build the end-to-end KWS pipeline")

    graph, weights = _shipped_kiln_graph()
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        (workdir / "kiln_model.c").write_text(
            emit_c(graph, weights, runtime_header=None), encoding="utf-8"
        )
        (workdir / "main.c").write_text(HARNESS, encoding="utf-8")
        executable = workdir / "kws_e2e.exe"
        result = subprocess.run(
            [
                "gcc",
                *GCC_FLAGS,
                "-I",
                str(FRONTEND_DIR),
                str(workdir / "main.c"),
                str(workdir / "kiln_model.c"),
                str(FRONTEND_SRC),
                str(PIPELINE_SRC),
                "-o",
                str(executable),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"end-to-end build failed:\n{result.stderr}"
        yield executable


def _run_e2e(executable: Path, cases: list[torch.Tensor]) -> list[tuple[int, torch.Tensor]]:
    payload = bytearray(struct.pack("<i", len(cases)))
    for pcm in cases:
        samples = pcm.to(torch.int16).flatten().tolist()
        payload += struct.pack("<i", len(samples))
        payload += array.array("h", samples).tobytes()

    with tempfile.TemporaryDirectory() as tmpdir:
        blob = Path(tmpdir) / "pcm_cases.bin"
        blob.write_bytes(bytes(payload))
        result = subprocess.run(
            [str(executable), str(blob)], capture_output=True, text=True, check=True
        )

    lines = result.stdout.split("\n")
    outputs: list[tuple[int, torch.Tensor]] = []
    cursor = 0
    for _ in cases:
        prediction = int(lines[cursor])
        cursor += 1
        rows: list[int] = []
        for _ in range(N_MELS):
            rows.extend(int(value) for value in lines[cursor].split())
            cursor += 1
        features = torch.tensor(rows, dtype=torch.int16)
        assert features.numel() == N_SPEC_VALUES, "harness printed the wrong feature count"
        outputs.append((prediction, features.view(N_MELS, N_FRAMES)))
    return outputs


def _reference_logmel(pcm: torch.Tensor) -> torch.Tensor:
    from examples.kws.dataset import extract_logmel_spectrogram

    # The reference normalises by the mel peak, so the int16 -> float scale does
    # not matter; 1/32768 mirrors the usual torchaudio WAV decode.
    return extract_logmel_spectrogram(pcm.to(torch.float32) / 32768.0)


def _pytorch_pipeline_prediction(model, pcm: torch.Tensor) -> int:
    """Reference device path: PCM -> log-mel -> int8 -> FP32 classifier -> argmax."""
    quantized = quantize_logmel_to_int8(_reference_logmel(pcm))
    with torch.no_grad():
        logits = model(quantized.float().unsqueeze(0) / 127.0).squeeze(-1)
    return int(torch.argmax(logits, dim=-1).item())


def test_kws_end_to_end_c_pipeline_matches_pytorch(e2e_exe: Path) -> None:
    cases = _real_utterances()
    model = load_kws_model(str(CHECKPOINT))
    graph, weights = _shipped_kiln_graph()

    outputs = _run_e2e(e2e_exe, [pcm for _, _, pcm in cases])

    correct = 0
    pipeline_agreement = 0
    bit_exact = 0
    for (name, truth, pcm), (prediction, features) in zip(cases, outputs):
        # 1. The generated graph must be bit-exact on the C frontend's features.
        reference_label = run_ref(graph, {"graph_in": features.flatten().tolist()}, weights)[
            "graph_out"
        ][0]
        bit_exact += int(prediction == reference_label)
        assert prediction == reference_label, (
            f"{name}: compiled C graph disagrees with kiln.ref.run_ref"
        )

        # 2. The full PCM-to-label path must classify real utterances correctly.
        correct += int(prediction == truth)

        # 3. ...and must track the PyTorch pipeline on the same audio.
        if prediction == _pytorch_pipeline_prediction(model, pcm):
            pipeline_agreement += 1

    accuracy = correct / len(cases)
    agreement = pipeline_agreement / len(cases)
    assert accuracy >= MIN_E2E_ACCURACY, (
        f"end-to-end accuracy {accuracy * 100:.1f}% < {MIN_E2E_ACCURACY * 100:.0f}% "
        f"({correct}/{len(cases)})"
    )
    assert agreement >= MIN_AGREEMENT, (
        f"end-to-end agreement {agreement * 100:.1f}% < {MIN_AGREEMENT * 100:.0f}% "
        f"({pipeline_agreement}/{len(cases)})"
    )
    # Printed so the Phase 1b certificate line in ROADMAP.md section 5 (and the
    # README table) can be reproduced with `pytest tests/test_kws_e2e.py -s`.
    print(
        f"PCM -> label via the generated C: {bit_exact}/{len(cases)} bit-exact vs kiln.ref, "
        f"{correct}/{len(cases)} correct ({accuracy * 100:.1f}%), "
        f"{pipeline_agreement}/{len(cases)} identical to the PyTorch pipeline"
    )
