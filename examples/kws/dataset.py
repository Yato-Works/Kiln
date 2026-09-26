from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .model import (
    KWS_CLASSES,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    quantize_logmel_to_int8,
)

# 12-class index map
CLASS_TO_IDX: dict[str, int] = {cls_name: i for i, cls_name in enumerate(KWS_CLASSES)}


def extract_logmel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_frames: int = N_FRAMES,
    win_length_ms: float = 40.0,
    hop_length_ms: float = 20.0,
) -> torch.Tensor:
    """
    Extracts log-mel spectrogram matching standard KWS preprocessor:
    - Sample rate: 16000 Hz (1 sec -> 16000 samples)
    - Window length: 40 ms -> 640 samples
    - Hop length: 20 ms -> 320 samples
    - Target frames: 49
    - Mel bins: 10
    Output shape: (n_mels=10, n_frames=49), row-major: [channel, frame]
    """
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)  # mono

    # Pad or trim to exactly 1 second (16000 samples)
    target_samples = sample_rate
    if waveform.shape[-1] < target_samples:
        pad_amt = target_samples - waveform.shape[-1]
        waveform = F.pad(waveform, (0, pad_amt))
    elif waveform.shape[-1] > target_samples:
        waveform = waveform[:target_samples]

    n_fft = 1024
    win_length = int(sample_rate * (win_length_ms / 1000.0))  # 640
    hop_length = int(sample_rate * (hop_length_ms / 1000.0))  # 320

    window = torch.hann_window(win_length)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    # Power spectrum: (n_fft // 2 + 1, frames)
    power_spec = torch.abs(stft) ** 2

    # Mel filterbank creation
    # Frequency range: 20 Hz to 4000 Hz
    f_min = 20.0
    f_max = 4000.0

    def hz_to_mel(f: float) -> float:
        return 2595.0 * torch.log10(torch.tensor(1.0 + f / 700.0)).item()

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = [mel_to_hz(m.item()) for m in mel_points]

    bin_points = [int(round((h * n_fft) / sample_rate)) for h in hz_points]

    weights = torch.zeros((n_mels, n_fft // 2 + 1), dtype=torch.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                weights[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                weights[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    # Filtered mel spectrogram: (n_mels, frames)
    mel_spec = torch.matmul(weights, power_spec)

    # Convert to log-mel dB (clamped between -80.0 dB and 0.0 dB)
    ref_val = torch.max(mel_spec).clamp(min=1e-6)
    log_mel = 10.0 * torch.log10((mel_spec / ref_val).clamp(min=1e-8))
    log_mel = torch.clamp(log_mel, min=-80.0, max=0.0)

    # Crop or pad frames to exactly n_frames
    if log_mel.shape[-1] < n_frames:
        log_mel = F.pad(log_mel, (0, n_frames - log_mel.shape[-1]), value=-80.0)
    elif log_mel.shape[-1] > n_frames:
        log_mel = log_mel[:, :n_frames]

    return log_mel


class SpeechCommandsSubsetDataset(Dataset):
    """
    PyTorch Dataset for Speech Commands v2 subsets:
    Loads WAV, computes log-mel, and provides both FP32 and int8 quantized tensors.
    """

    def __init__(self, samples: Sequence[tuple[torch.Tensor, int]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        waveform, label = self.samples[idx]
        logmel_db = extract_logmel_spectrogram(waveform)
        int8_feat = quantize_logmel_to_int8(logmel_db)
        return {
            "logmel_db": logmel_db,
            "int8_feat": int8_feat,
            "label": torch.tensor(label, dtype=torch.long),
        }
