"""Regression tests for the fixed-point KWS frontend.

``examples/kws/frontend/kws_frontend.c`` must reproduce the training-time
reference (``extract_logmel_spectrogram`` + ``quantize_logmel_to_int8``) from raw
16-bit PCM.  These tests compile the real C source with the project's strict
warning flags, run it over deterministic and real utterances, and compare the
emitted 10x49 int8 log-mel frames against the PyTorch reference.

Tolerance contract: one int8 step is 40/127 dB ~= 0.315 dB, and the fixed-point
pipeline is expected to stay within a single step of the FP32 reference (it is
usually bit exact).
"""

from __future__ import annotations

import array
import importlib.util
import io
import math
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile
import wave
from pathlib import Path

import pytest
import torch

from examples.kws.dataset import extract_logmel_spectrogram
from examples.kws.model import (
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    quantize_logmel_to_int8,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "examples" / "kws" / "frontend"
FRONTEND_SRC = FRONTEND_DIR / "kws_frontend.c"
TABLES_HEADER = FRONTEND_DIR / "kws_tables.h"
SPEECH_COMMANDS_ARCHIVE = REPO_ROOT / "speech_commands_test_set_v0.02.tar.gz"

N_SPEC_VALUES = N_MELS * N_FRAMES
MAX_ABS_DELTA = 1
MAX_MAE = 0.12

FRONTEND_HARNESS = r"""
#include <stdio.h>
#include <stdint.h>
#include "kws_frontend.h"

/*
 * Reads: int32 case_count, then per case int32 sample_count followed by that
 * many int16 samples.  Prints one block of KWS_N_MELS lines of KWS_N_FRAMES
 * int8 values per case.
 */
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
        int8_t spec[KWS_N_MELS * KWS_N_FRAMES];
        kws_frontend_process(pcm, (size_t)sample_count, spec);
        for (int m = 0; m < KWS_N_MELS; ++m) {
            for (int frame = 0; frame < KWS_N_FRAMES; ++frame) {
                printf("%d%c", (int)spec[m * KWS_N_FRAMES + frame],
                       frame + 1 == KWS_N_FRAMES ? '\n' : ' ');
            }
        }
    }
    fclose(f);
    return 0;
}
"""


PIPELINE_HARNESS = r"""
#include <stdio.h>
#include <stdint.h>
/* kws_pipeline.h pulls in the frontend shape macros, so no extra include. */
#include "kws_pipeline.h"

/* Stand-in for the model the Kiln C99 backend generates: records the int8
 * features it is called with and returns a recognisable class index. */
static int8_t captured[KWS_N_MELS * KWS_N_FRAMES];

int8_t kiln_predict(const uint8_t *input_bytes, int len) {
    if (len != KWS_N_MELS * KWS_N_FRAMES) return -1;
    for (int i = 0; i < len; ++i) {
        captured[i] = (int8_t)input_bytes[i];
    }
    return 42;
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    int32_t sample_count = 0;
    if (fread(&sample_count, sizeof(int32_t), 1, f) != 1) return 3;
    if (sample_count < 0 || sample_count > 2 * KWS_SAMPLE_RATE) return 4;
    static int16_t pcm[2 * KWS_SAMPLE_RATE];
    if (sample_count > 0 &&
        fread(pcm, sizeof(int16_t), (size_t)sample_count, f) != (size_t)sample_count) {
        return 5;
    }
    fclose(f);
    printf("%d\n", (int)kws_predict_raw_pcm(pcm, (size_t)sample_count));
    for (int i = 0; i < KWS_N_MELS * KWS_N_FRAMES; ++i) {
        printf("%d%c", (int)captured[i], (i + 1) % KWS_N_FRAMES == 0 ? '\n' : ' ');
    }
    return 0;
}
"""


def _compile_pipeline(workdir: Path) -> Path:
    harness_src = workdir / "kws_pipeline_harness.c"
    exe = workdir / "kws_pipeline_harness.exe"
    harness_src.write_text(PIPELINE_HARNESS, encoding="utf-8")
    subprocess.run(
        [
            "gcc",
            "-std=c99",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(FRONTEND_DIR),
            str(harness_src),
            str(FRONTEND_SRC),
            str(FRONTEND_DIR / "kws_pipeline.c"),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return exe


def _compile_frontend(workdir: Path) -> Path:
    harness_src = workdir / "kws_frontend_harness.c"
    exe = workdir / "kws_frontend_harness.exe"
    harness_src.write_text(FRONTEND_HARNESS, encoding="utf-8")
    subprocess.run(
        [
            "gcc",
            "-std=c99",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(FRONTEND_DIR),
            str(harness_src),
            str(FRONTEND_SRC),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return exe


def _run_frontend(exe: Path, pcm_cases: list[torch.Tensor]) -> torch.Tensor:
    """Run the C frontend on int16 PCM cases; returns a (cases, 10, 49) int16 tensor."""
    payload = bytearray(struct.pack("<i", len(pcm_cases)))
    for pcm in pcm_cases:
        samples = pcm.to(torch.int16).flatten().tolist()
        payload += struct.pack("<i", len(samples))
        payload += array.array("h", samples).tobytes()

    with tempfile.TemporaryDirectory() as tmpdir:
        blob = Path(tmpdir) / "pcm_cases.bin"
        blob.write_bytes(bytes(payload))
        result = subprocess.run([str(exe), str(blob)], capture_output=True, text=True, check=True)
    values = [int(v) for v in result.stdout.split()]
    assert len(values) == len(pcm_cases) * N_SPEC_VALUES, (
        f"frontend printed {len(values)} values, expected {len(pcm_cases) * N_SPEC_VALUES}"
    )
    return torch.tensor(values, dtype=torch.int16).view(len(pcm_cases), N_MELS, N_FRAMES)


@pytest.fixture(scope="module")
def pipeline_exe() -> Path:
    if shutil.which("gcc") is None:
        pytest.skip("gcc required to build the C KWS frontend")
    with tempfile.TemporaryDirectory() as tmpdir:
        yield _compile_pipeline(Path(tmpdir))


def _run_pipeline(exe: Path, pcm: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Run `kws_predict_raw_pcm`; returns (class index, features handed to the model)."""
    samples = pcm.to(torch.int16).flatten().tolist()
    payload = struct.pack("<i", len(samples)) + array.array("h", samples).tobytes()
    with tempfile.TemporaryDirectory() as tmpdir:
        blob = Path(tmpdir) / "pcm.bin"
        blob.write_bytes(payload)
        result = subprocess.run([str(exe), str(blob)], capture_output=True, text=True, check=True)
    lines = result.stdout.split()
    prediction = int(lines[0])
    features = torch.tensor([int(v) for v in lines[1:]], dtype=torch.int16)
    assert features.numel() == N_SPEC_VALUES
    return prediction, features.view(N_MELS, N_FRAMES)


def _reference_int8(pcm: torch.Tensor) -> torch.Tensor:
    """Python reference for the same 16-bit PCM.

    The reference normalises by the mel peak, so the int16 -> float scale does
    not matter; 1/32768 mirrors the usual torchaudio WAV decode.
    """
    return quantize_logmel_to_int8(extract_logmel_spectrogram(pcm.to(torch.float32) / 32768.0))


def _assert_parity(name: str, c_frames: torch.Tensor, reference: torch.Tensor) -> None:
    delta = (c_frames.to(torch.int32) - reference.to(torch.int32)).abs()
    max_delta = int(delta.max())
    mae = float(delta.double().mean())
    assert max_delta <= MAX_ABS_DELTA, f"{name}: max |delta| = {max_delta} int8 steps"
    assert mae <= MAX_MAE, f"{name}: MAE = {mae:.4f} int8 steps"


def _synthetic_cases() -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1337)
    t = torch.arange(SAMPLE_RATE, dtype=torch.float32)

    voiced = (
        0.35 * torch.sin(2 * torch.pi * 220.0 * t / SAMPLE_RATE)
        + 0.15 * torch.sin(2 * torch.pi * 660.0 * t / SAMPLE_RATE)
    ) * (1.0 + torch.sin(2 * torch.pi * 3.0 * t / SAMPLE_RATE))
    noise = 0.25 * torch.randn(SAMPLE_RATE, generator=generator)
    speech_like = (voiced + noise).clamp(-1.0, 1.0)
    speech_like[12000:] = 0.0  # trailing digital silence, as in clipped utterances

    short_utterance = (0.6 * voiced[:4000] + 0.05 * noise[:4000]).clamp(-1.0, 1.0)

    # A quiet utterance followed by a loud burst in the last 200 samples: the
    # loudest mel frame then falls in the two STFT frames the reference crops
    # away (49/50), yet it still normalises by them, so the frontend has to
    # analyse 51 frames.  The quiet head keeps the retained frames well above
    # the -80 dB clamp.
    tail_heavy = (0.15 * voiced + 0.2 * noise).clamp(-1.0, 1.0)
    tail_heavy[15800:] = 0.9 * torch.sin(2 * torch.pi * 1000.0 * t[15800:] / SAMPLE_RATE)

    # Longer than one second: the reference trims to SAMPLE_RATE samples and the
    # frontend must cap the same way, ignoring the loud tail.
    over_long = torch.cat([speech_like, 0.8 * torch.randn(4000, generator=generator)])

    def to_pcm(signal: torch.Tensor) -> torch.Tensor:
        return (signal.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)

    return [
        ("speech-like", to_pcm(speech_like)),
        ("noise", to_pcm(noise)),
        ("tone", to_pcm(0.4 * torch.sin(2 * torch.pi * 440.0 * t / SAMPLE_RATE))),
        ("short-utterance", to_pcm(short_utterance)),
        ("over-long", to_pcm(over_long)),
        ("tail-heavy", to_pcm(tail_heavy)),
        ("empty", torch.zeros(0, dtype=torch.int16)),
        ("silence", torch.zeros(SAMPLE_RATE, dtype=torch.int16)),
    ]


@pytest.fixture(scope="module")
def frontend_exe() -> Path:
    if shutil.which("gcc") is None:
        pytest.skip("gcc required to build the C KWS frontend")
    with tempfile.TemporaryDirectory() as tmpdir:
        yield _compile_frontend(Path(tmpdir))


def test_kws_frontend_matches_python_reference_on_synthetic_pcm(frontend_exe: Path) -> None:
    cases = _synthetic_cases()
    c_frames = _run_frontend(frontend_exe, [pcm for _, pcm in cases])

    for (name, pcm), c_spec in zip(cases, c_frames):
        reference = _reference_int8(pcm)
        if pcm.numel() == 0 or not bool(pcm.any()):
            # Digital silence: the reference clamps every cell to -80 dB.
            assert torch.equal(reference, torch.full_like(reference, -127)), name
        elif name == "tail-heavy":
            # The peak frame is cropped away, so nothing reaches 0 dB and
            # nothing sits on the floor either.
            assert 0 < int(reference.max()) < 127, name
        else:
            assert int(reference.max()) == 127, f"{name}: reference is not peak normalised"
        _assert_parity(name, c_spec, reference)


def test_kws_frontend_tail_heavy_clip_uses_51_frame_peak(frontend_exe: Path) -> None:
    """The reference crops frames 49/50 *after* normalising by the mel peak.

    A clip whose loudest mel frame is one of those two frames would drift by the
    full peak difference if the frontend only analysed 49 frames.
    """
    pcm = dict(_synthetic_cases())["tail-heavy"]
    reference = _reference_int8(pcm)
    assert int(reference.max()) < 127, "case no longer peaks inside the cropped frames"
    assert int(reference.min()) > -127, "case degenerates into the dB floor"
    c_spec = _run_frontend(frontend_exe, [pcm])[0]
    _assert_parity("tail-heavy", c_spec, reference)


def test_kws_frontend_digital_silence_maps_to_floor(frontend_exe: Path) -> None:
    silence_cases = [
        torch.zeros(0, dtype=torch.int16),
        torch.zeros(SAMPLE_RATE, dtype=torch.int16),
    ]
    c_frames = _run_frontend(frontend_exe, silence_cases)
    for c_spec in c_frames:
        assert torch.equal(c_spec, torch.full_like(c_spec, -127))


def _read_wav_pcm(payload: bytes) -> torch.Tensor:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE, "unexpected sample rate in archive"
        assert wav.getsampwidth() == 2, "expected 16-bit PCM in archive"
        assert wav.getnchannels() == 1, "expected mono PCM in archive"
        samples = array.array("h", wav.readframes(wav.getnframes()))
    return torch.tensor(samples, dtype=torch.int16)


def test_kws_frontend_matches_python_reference_on_real_speech(frontend_exe: Path) -> None:
    if not SPEECH_COMMANDS_ARCHIVE.exists():
        pytest.skip("Speech Commands archive not found")

    with tarfile.open(SPEECH_COMMANDS_ARCHIVE, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".wav")]
        assert len(members) > 400, "unexpected archive layout"
        selected = members[::7][:57]
        pcm_cases = []
        for member in selected:
            member_file = tar.extractfile(member)
            assert member_file is not None
            pcm_cases.append(_read_wav_pcm(member_file.read()))

    c_frames = _run_frontend(frontend_exe, pcm_cases)

    exact = 0
    total_delta = 0
    for member, pcm, c_spec in zip(selected, pcm_cases, c_frames):
        reference = _reference_int8(pcm)
        delta = (c_spec.to(torch.int32) - reference.to(torch.int32)).abs()
        exact += int((delta == 0).sum())
        total_delta += int(delta.sum())
        _assert_parity(member.name, c_spec, reference)

    cells = len(pcm_cases) * N_SPEC_VALUES
    assert exact / cells >= 0.90, f"only {100 * exact / cells:.1f}% of cells are bit exact"
    assert total_delta / cells <= MAX_MAE


def _parse_int_table(source: str, name: str) -> list[int]:
    match = re.search(rf"{name}\[[^\]]*\] = \{{(.*?)\}};", source, re.S)
    assert match is not None, f"{name} not found in kws_tables.h"
    return [int(v) for v in re.findall(r"-?\d+", match.group(1))]


def test_kws_pipeline_forwards_frontend_features(frontend_exe: Path, pipeline_exe: Path) -> None:
    """`kws_predict_raw_pcm` must hand the frontend's int8 frames to the model."""
    pcm = dict(_synthetic_cases())["speech-like"]
    direct = _run_frontend(frontend_exe, [pcm])[0]
    prediction, forwarded = _run_pipeline(pipeline_exe, pcm)
    assert prediction == 42, "pipeline did not return the model's class index"
    assert torch.equal(forwarded, direct), "pipeline modified the frontend features"


def test_kws_frontend_tables_match_generator() -> None:
    """``kws_tables.h`` must stay byte-identical to ``gen_tables.py`` output."""
    spec = importlib.util.spec_from_file_location("kws_gen_tables", FRONTEND_DIR / "gen_tables.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.generate() == TABLES_HEADER.read_text(encoding="utf-8")


def test_kws_hann_window_table_matches_torch() -> None:
    """The Q15 table must be the periodic torch.hann_window(640) window.

    The generator rounds float64 values while torch evaluates the window in
    float32, so a handful of taps may differ by one LSB; a symmetric (non
    periodic) window or the wrong length would differ by orders of magnitude.
    """
    values = _parse_int_table(TABLES_HEADER.read_text(encoding="utf-8"), "kws_hann_window")
    assert len(values) == 640
    expected = torch.round(torch.hann_window(640) * 32767.0).to(torch.int32)
    assert int((torch.tensor(values, dtype=torch.int32) - expected).abs().max()) <= 1


def test_kws_log2_lut_matches_q16_reference() -> None:
    """The 64-entry log2 LUT must be round(log2(1 + i/64) * 2^16)."""
    values = _parse_int_table(TABLES_HEADER.read_text(encoding="utf-8"), "kws_log2_lut")
    expected = [int(round(math.log2(1.0 + i / 64.0) * 65536)) for i in range(64)]
    assert values == expected
