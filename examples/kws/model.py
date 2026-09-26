from __future__ import annotations

import os
import torch
import torch.nn as nn

KWS_CLASSES: tuple[str, ...] = (
    "silence",
    "unknown",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)

SAMPLE_RATE: int = 16000
N_MELS: int = 10
N_FRAMES: int = 49

V_MIN_DB: float = -80.0
V_MAX_DB: float = 0.0
V_MID_DB: float = -40.0
HALF_RANGE_DB: float = 40.0
SCALE_DB: float = 40.0 / 127.0


#: Stage layout as ``(out_channels, kernel_size, padding, pool_size)``.
#:
#: The last stage is the classifier head: it maps to ``num_classes`` logits,
#: applies no ReLU and is never pooled.  ``pool_size <= 1`` means "no pooling",
#: and is only legal on the head (a non-final stage must pool, otherwise the
#: next stage would pool with a stale stage index).
#:
#: Sizing rules (Phase 1a budget: ROM < 40 KB, RAM < 2 KB).  With the planner's
#: in-place LUT fusion + zero-copy graph inputs + ``kiln.ir.arena`` placement, a
#: stage whose convolution output is ``c_out x L_out`` bytes costs:
#:
#:   * ``c_out * L_out`` bytes in the first stage (the 10 x 49 input lives in
#:     the caller's buffer, so it is free), and
#:   * the pooled tensor stays live while the *next* convolution writes its own
#:     output, so the arena peak is the largest of
#:     ``c_out * (frames_in + frames_pooled)`` and
#:     ``frames_pooled * (c_out + c_next)`` -- the two sets of tensors that are
#:     live at the same op, and
#:   * everything else is packed into the gaps left by dead tensors, so the
#:     arena stays at that peak instead of growing by whole blocks: the
#:     allocator, not the weights, decides which tensor reuses which gap.
#:
#: With ``(27, 5, 2, 2)`` the first stage costs 27 * 73 = 1,971 B and the second
#: convolution needs 55 * 24 = 1,320 B <= 27 * 49 = 1,323 B, so it lands inside
#: the recycled block and the total stays at 1,972 B in the emitted C.  Every
#: later stage is far smaller (<= 672 B), because the time axis is already short.
#:
#: Measured certificate for this layout (734-sample test split, split seed 1337,
#: from ``python -m examples.kws.evaluate``): 38,173 parameters (~38.6K, the MLPerf
#: DS-CNN footprint), 39,197 B ROM, 1,972 B arena, 310,382 MACs and 89.10% int8
#: C top-1 accuracy (validation 91.27% FP32 / 91.00% C99).  Four training runs of
#: the shipped recipe span 88.69-90.05% FP32 test accuracy (two of them measured
#: 89.10% and 90.05% through the emitted C), i.e. within the +/-2.2 point
#: confidence interval of a 734-clip test set; the shipped checkpoint is the
#: highest-validation run (see ROADMAP.md Phase 1, findings 4 and 5).
KWS_ARCHITECTURE: tuple[tuple[int, int, int, int], ...] = (
    (27, 5, 2, 2),
    (55, 3, 1, 4),
    (112, 3, 1, 3),
    (112, 1, 0, 2),
    (12, 1, 0, 0),
)


def quantize_logmel_to_int8(logmel_db: torch.Tensor) -> torch.Tensor:
    centered = logmel_db - V_MID_DB
    scaled = centered / SCALE_DB
    return torch.clamp(torch.round(scaled), -127, 127).to(torch.int8)


class KilnKWSModel(nn.Sequential):
    """1-D CNN expressible in Kiln IR (CONV1D + LUT/ReLU + MAXPOOL1D + ARGMAX)."""

    def __init__(
        self,
        num_classes: int = len(KWS_CLASSES),
        architecture: tuple[tuple[int, int, int, int], ...] = KWS_ARCHITECTURE,
    ) -> None:
        layers: list[nn.Module] = []
        in_channels = N_MELS
        last_stage = len(architecture) - 1

        for index, (width, kernel, padding, pool) in enumerate(architecture):
            out_channels = num_classes if index == last_stage else width
            if index != last_stage and pool <= 1:
                raise ValueError(
                    "every non-final stage must pool: stage "
                    f"{index} of {architecture} has pool_size={pool}"
                )
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    padding=padding,
                    bias=False,
                )
            )
            in_channels = out_channels

            if index == last_stage:
                break

            layers.append(nn.ReLU())
            if pool > 1:
                layers.append(nn.MaxPool1d(kernel_size=pool, stride=pool))

        super().__init__(*layers)


def load_kws_model(
    weights_path: str | None = None,
    *,
    num_classes: int = len(KWS_CLASSES),
    architecture: tuple[tuple[int, int, int, int], ...] = KWS_ARCHITECTURE,
) -> KilnKWSModel:
    """Builds the KWS model and loads the shipped int8-trained weights if present."""
    model = KilnKWSModel(num_classes=num_classes, architecture=architecture)
    path = weights_path or os.path.join(os.path.dirname(__file__), "kws_qat_weights.pt")
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, weights_only=True))
        except RuntimeError as error:
            raise RuntimeError(
                f"{path} does not match KWS_ARCHITECTURE; retrain it with "
                f"`python -m examples.kws.train_aug`"
            ) from error
    model.eval()
    return model


model = load_kws_model()

dummy_input = torch.zeros((N_MELS, N_FRAMES), dtype=torch.int8)

