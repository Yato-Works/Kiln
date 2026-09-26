"""Shipped Phase 1a training recipe for the Kiln KWS classifier.

Trains exactly the contract the C backend executes:

* inputs are the int8 log-mel tensors the fixed-point frontend produces,
  interpreted as ``q / 127.0`` (``V_mid = -40 dB`` symmetric quantization),
* the network only uses ops Kiln can lower (see ``examples/kws/model.py``),
* optional augmentation happens *in the quantized domain* (time roll, gain
  offset and SpecAugment masks), so training can never drift from the deployed
  contract.

The defaults reproduce the shipped checkpoint: 300 epochs, no augmentation, the
``shipped`` architecture, writing ``kws_qat_weights.pt``.

Usage::

    python -m examples.kws.train_aug                          # shipped artifact
    python -m examples.kws.train_aug --augment                # + augmentation preset
    python -m examples.kws.train_aug --architecture stage3 --epochs 120
    python -m examples.kws.train_aug --seed 777 --out %TEMP%/kws_s777.pt

The script prints one ``RESULT {json}`` line at the end so sweeps can be
compared mechanically (val accuracy drives model selection, test accuracy is
reported for the Phase 1a benchmark table).

CUDA caveat: ``--device cuda`` runs the convolutions through TF32 with
non-deterministic cuDNN kernels, so runs of the same seed are only close, not
bit-identical: three runs of the identical command measured FP32 val
91.13 / 91.27 / 91.00 and FP32 test 89.78 / 88.69 / 89.51 (the highest-validation
run is the one shipped, see ROADMAP.md finding 5).  If you need identical weights
across runs, train with deterministic settings (e.g. ``--device cpu``); the
emitted C, the budget audit and the benchmark are deterministic either way.

Legacy recipes are kept for the PTQ-vs-QAT comparison: ``train.py`` fits the same
architecture in the raw dB domain, and ``ptq_test.py`` shows what that costs once
it is lowered (50.54% vs 89.10% int8 C, see ROADMAP.md).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from examples.kws.kws_graph import split_indices
from examples.kws.model import KWS_ARCHITECTURE, KilnKWSModel, quantize_logmel_to_int8

__all__ = ("ARCHITECTURES", "AUGMENTATION", "main")

DATA_PATH = os.path.join(os.path.dirname(__file__), "kws_processed_data.pt")
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "kws_qat_weights.pt")

#: Standard int8-domain augmentation preset, applied by ``--augment``.
#:
#: Measured Phase 1a finding: on this 4,890-clip corpus (3,423 training clips) the
#: preset *hurts* (85.69% vs 88.42% test at 120 epochs, same architecture and
#: seed), so the shipped checkpoint is trained with augmentation disabled and the
#: flags default to 0.  The switches stay available for larger corpora.
AUGMENTATION: dict[str, int] = {
    "time_shift": 6,
    "gain_steps": 12,
    "time_masks": 2,
    "freq_masks": 1,
}

#: Architectural presets.
#:
#: ``shipped`` mirrors ``model.KWS_ARCHITECTURE`` (the artifact the Phase 1a
#: certificate is taken from).  The others are kept as the measured search space:
#: every entry was trained on the frozen split, and the arena size is the one
#: ``kiln.budget.report.audit_graph`` reports for the emitted C.
#:
#: RAM sizing rule of thumb (``kiln.ir.memory`` + ``kiln.ir.arena``): a fused
#: ``CONV1D -> LUT`` stage keeps its output and its pooled output alive at the
#: same time, and the pooled tensor is still live while the *next* convolution
#: writes its output.  For a first stage with ``c0`` channels and ``L0`` pooled
#: frames the arena therefore has to hold
#: ``max(c0 * (49 + L0), L0 * (c0 + c1))`` bytes, and the allocator reaches that
#: bound because everything else is packed into gaps left by dead tensors.  With
#: 24 pooled frames that is ``73 * c0 <= 2048`` and ``24 * (c0 + c1) <= 2048``
#: before alignment, i.e. ``c0 <= 28`` and ``c0 + c1 <= 85`` -- the First-Fit
#: planner this file was first measured with needed ``c1 * L0 <= c0 * 49``
#: instead, which capped ``c1`` at 55 for ``c0 = 27``.
ARCHITECTURES: dict[str, tuple[tuple[int, int, int, int], ...]] = {
    # The certificate layout (ROADMAP.md section 1): 1,992 B arena / 39,614 B ROM
    # / 91.01% int8 C test (90.74% FP32, val 91.54% FP32 / 91.68% C99).  The best
    # layout measured on this corpus, and inside both budgets because
    # ``kiln.ir.arena`` reuses the first stage's freed 1,324 B block; the
    # First-Fit-only planner this file was first measured with needed 3,316 B.
    "shipped": KWS_ARCHITECTURE,
    # The previous shipped layout (27/55), kept as the certificate-v1 artifact:
    # 1,972 B arena / 39,197 B ROM / 89.10% int8 C test (88.69% FP32, val 91.27%
    # FP32 / 91.00% C99).  `python -m examples.kws.qat_eval --architecture
    # previous --weights examples/kws/kws_qat_weights_27_55.pt` reprints it.
    "previous": ((27, 5, 2, 2), (55, 3, 1, 4), (112, 3, 1, 3), (112, 1, 0, 2), (12, 1, 0, 0)),
    # Original Phase 1a baseline: 2,336 B arena (over budget), 90.87% test.
    "small": ((32, 3, 1, 2), (48, 3, 1, 24), (12, 1, 0, 0)),
    # 16 pooled frames: 2,016 B arena, 88.69-89.65% test.
    "stage3": ((31, 5, 2, 3), (64, 3, 1, 4), (96, 3, 1, 2), (96, 1, 0, 2), (12, 1, 0, 0)),
    # 12 pooled frames: 2,016 B arena, 87.74% test.
    "pool4": ((33, 5, 2, 4), (128, 3, 1, 4), (128, 1, 0, 3), (12, 1, 0, 0)),
    # 9 pooled frames: 2,032 B arena, 89.24% test (300 ep; 89.65% at 120 ep).
    "pool5": ((35, 5, 2, 5), (128, 3, 1, 3), (128, 1, 0, 3), (12, 1, 0, 0)),
    # The unconstrained layout: 24 pooled frames with a 64-channel second stage.
    # ``c0 + c1 = 90 > 85``, so the second convolution (64 * 24 = 1,536 B) still
    # does not fit next to the 624 B pooled tensor: 2,160 B with the improved
    # allocator (3,436 B with First-Fit).  Best accuracy measured (90.60% test).
    "wide_second": ((26, 5, 2, 2), (64, 3, 1, 4), (96, 3, 1, 3), (128, 1, 0, 2), (12, 1, 0, 0)),
}


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="shipped")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="enable the standard int8-domain augmentation preset (see AUGMENTATION)",
    )
    parser.add_argument("--time-shift", type=int, default=0, help="max |frame| roll")
    parser.add_argument("--gain-steps", type=int, default=0, help="max |int8 step| gain")
    parser.add_argument("--time-masks", type=int, default=0, help="SpecAugment time masks")
    parser.add_argument("--time-mask-width", type=int, default=8)
    parser.add_argument("--freq-masks", type=int, default=0, help="SpecAugment mel masks")
    parser.add_argument("--freq-mask-width", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20240925)
    parser.add_argument("--split-seed", type=int, default=1337, help="split fixed for Phase 1a")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--out", default=DEFAULT_WEIGHTS)
    arguments = parser.parse_args(argv)

    # ``--augment`` fills in every augmentation knob that the caller left at its
    # (disabled) default, so augmentation can only ever be switched on.
    if arguments.augment:
        for name, value in AUGMENTATION.items():
            if getattr(arguments, name) == 0:
                setattr(arguments, name, value)
    return arguments



def _time_roll(inputs: torch.Tensor, max_shift: int, generator: torch.Generator) -> torch.Tensor:
    """Circularly shifts each sample along the time axis (speaker timing jitter)."""
    length = inputs.shape[-1]
    if max_shift <= 0 or length < 2:
        return inputs

    offsets = torch.randint(-max_shift, max_shift + 1, (inputs.shape[0],), generator=generator)
    index = (torch.arange(length) - offsets[:, None]) % length
    return inputs.gather(2, index[:, None, :].expand(-1, inputs.shape[1], -1))


def _gain_offset(inputs: torch.Tensor, max_steps: int, generator: torch.Generator) -> torch.Tensor:
    """Shifts every mel bin by a random number of int8 steps (microphone gain)."""
    if max_steps <= 0:
        return inputs

    steps = torch.randint(-max_steps, max_steps + 1, (inputs.shape[0], 1, 1), generator=generator)
    return torch.clamp(inputs + steps.to(inputs.dtype), -127, 127)


def _mask(
    inputs: torch.Tensor,
    axis: int,
    count: int,
    width: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Silences ``count`` random bands of up to ``width`` slots along ``axis``."""
    if count <= 0 or width <= 0:
        return inputs

    size = inputs.shape[axis]
    positions = torch.arange(size)[None, :]
    output = inputs
    for _ in range(count):
        span = torch.randint(1, min(width, size) + 1, (inputs.shape[0],), generator=generator)
        start = torch.randint(0, size, (inputs.shape[0],), generator=generator)
        band = ((positions - start[:, None]) % size) < span[:, None]
        if axis == 1:
            band = band[:, :, None]
        else:
            band = band[:, None, :]
        output = torch.where(band, torch.full_like(output, -127), output)
    return output


def _augment(
    inputs: torch.Tensor,
    arguments: argparse.Namespace,
    generator: torch.Generator,
) -> torch.Tensor:
    """Augments int8 steps (floats in ``[-127, 127]``), returning the same scale."""
    augmented = _time_roll(inputs, arguments.time_shift, generator)
    augmented = _gain_offset(augmented, arguments.gain_steps, generator)
    augmented = _mask(augmented, 2, arguments.time_masks, arguments.time_mask_width, generator)
    return _mask(augmented, 1, arguments.freq_masks, arguments.freq_mask_width, generator)


def _accuracy(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, batch_size: int) -> float:
    correct = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            logits = model(inputs[start : start + batch_size]).squeeze(-1)
            correct += (logits.argmax(dim=-1) == labels[start : start + batch_size]).sum().item()
    return correct / len(labels)


def main(argv: list[str] | None = None) -> None:
    arguments = _parse_arguments(argv)
    torch.manual_seed(arguments.seed)
    device = torch.device(arguments.device)

    data = torch.load(DATA_PATH, weights_only=True)
    labels = data["labels"]
    train_idx, val_idx, test_idx = split_indices(len(labels), seed=arguments.split_seed)

    # The int8 contract: the exact tensors the C frontend hands to kiln_predict.
    # Augmentation runs in int8-step space (floats in [-127, 127]) so the
    # deployed quantization is what gets trained and perturbed.
    quantized = quantize_logmel_to_int8(data["specs"]).float()
    train_x, train_y = quantized[train_idx], labels[train_idx]
    val_x, val_y = quantized[val_idx] / 127.0, labels[val_idx]
    test_x, test_y = quantized[test_idx] / 127.0, labels[test_idx]

    model = KilnKWSModel(architecture=ARCHITECTURES[arguments.architecture]).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"architecture={arguments.architecture} params={parameters:,} "
        f"train={len(train_y)} val={len(val_y)} test={len(test_y)} device={device}",
        flush=True,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=arguments.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.lr,
        weight_decay=arguments.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arguments.epochs)
    generator = torch.Generator().manual_seed(arguments.seed)

    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.time()

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        order = torch.randperm(len(train_y), generator=generator)
        running_loss = 0.0
        correct = 0

        for start in range(0, len(order), arguments.batch_size):
            batch = order[start : start + arguments.batch_size]
            x_batch = (_augment(train_x[batch], arguments, generator) / 127.0).to(device)
            y_batch = train_y[batch].to(device)

            optimizer.zero_grad()
            logits = model(x_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(dim=-1) == y_batch).sum().item()

        scheduler.step()

        validation = _accuracy(model, val_x.to(device), val_y.to(device), arguments.batch_size)
        if validation > best_accuracy:
            best_accuracy = validation
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        if epoch % arguments.report_every == 0 or epoch == arguments.epochs:
            print(
                f"epoch {epoch:4d}/{arguments.epochs} | loss {running_loss / len(train_y):.4f} | "
                f"train {correct / len(train_y) * 100:5.2f}% | val {validation * 100:5.2f}% | "
                f"best {best_accuracy * 100:5.2f}%",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval().cpu()
    test_accuracy = _accuracy(model, test_x, test_y, arguments.batch_size)

    torch.save(best_state, arguments.out)
    print(
        "RESULT "
        + json.dumps(
            {
                "architecture": arguments.architecture,
                "parameters": parameters,
                "epochs": arguments.epochs,
                "val_accuracy": round(best_accuracy * 100, 2),
                "test_accuracy": round(test_accuracy * 100, 2),
                "test_samples": int(len(test_y)),
                "seconds": round(time.time() - started, 1),
                "output": arguments.out,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
