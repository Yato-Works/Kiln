# Kiln V2.0.0 — Foundation Release

Kiln V2 is the first public generation of Kiln: a board-free, reproducible
compiler foundation for deterministic TinyML inference.

## Included

- Frozen KIR v1 JSON graphs and graph-plus-weights bundles.
- Deterministic KIR VM with traces, replay, and resource metrics.
- PyTorch `torch.fx` frontend for the documented operator subset.
- Standalone C99 backend with zero dynamic allocation in generated inference code.
- Static arena, ROM, MAC, and int32 accumulator safety reporting.
- Independent integer reference and differential testing.
- Host linking and experimental Cortex-M/QEMU target paths.
- Official Hazard and KWS bundles.
- Clean-checkout tutorials, contributor workflow, security policy, and package CI.

## Audience

Kiln V2 is intended for embedded engineers, students, and researchers. A board
is not required to learn or verify the core workflow.

## Honest boundaries

The int32 result is a conservative static interval-analysis certificate, not a
machine-checked theorem. VM cycle counts are virtual-target estimates, not
silicon latency. ONNX/TFLite importers, turnkey RP2040 workflows, and broad
hardware validation remain future work.

See the [repository README](../README.md), [V2 overview](V2.md),
[CHANGELOG](../CHANGELOG.md), and [target validation contract](../TARGET_VALIDATION.md)
for the complete contract.
