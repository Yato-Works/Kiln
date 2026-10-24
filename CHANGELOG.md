# Changelog

All notable changes to Kiln are recorded here.

## [Unreleased]

- Continue expanding verified KIR bundles and target backends.

## [2.0.1] - 2026-10-24

- Make the shipped KWS bundle reproducible across supported PyTorch versions by
  freezing its calibration scales as part of the artifact contract.
- Generate the Hazard certificate fixture before the RP2040 adapter test in CI.
- Correct the V2 GitHub Release notes formatting.

## [2.0.0] - 2026-10-24

- Kiln V2 foundation release: a board-free, reproducible TinyML compiler core.
- Frozen KIR v1 JSON graph and bundle schemas with compatibility policy.
- Deterministic KIR VM with traces, replay, metrics, and fail-closed validation.
- C99 backend with static arena/ROM/MAC certificates and zero-malloc evidence.
- Host, Cortex-M cross-compile, and Cortex-M3 QEMU replay paths.
- Official portable Hazard and KWS KIR bundles with reproducible exporters.
- Board-free self-test, capability doctor, clean-checkout tutorials, and release tooling.
