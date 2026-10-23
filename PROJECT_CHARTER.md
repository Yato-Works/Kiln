# Kiln Project Charter

**Status:** adopted 2026-10-23  
**Current milestone:** v0.2 foundation

## Mission

Kiln is an open compiler foundation that turns quantized models into tiny,
deterministic, zero-malloc inference programs whose resource usage and integer
safety can be inspected and independently reproduced.

The primary audiences are embedded engineers, students, and researchers.

## Product promise

> Certified TinyML C compilers: readable generated C, explicit budgets, and
> fail-closed safety checks.

Kiln is not required to support every model or every device immediately. It is
required to be honest about its supported subset and rigorous about the
guarantees it claims inside that subset.

## Non-negotiable principles

1. Zero dynamic allocation in generated inference code.
2. Deterministic output for identical KIR, weights, and toolchain inputs.
3. An independent integer reference for every supported semantic operation.
4. Static RAM/ROM/MAC accounting before deployment.
5. Fail-closed behavior when overflow safety cannot be established.
6. Readable, inspectable generated C and versioned KIR.
7. Reproducible tests and documented evidence rather than marketing claims.
8. A learning path that exposes compiler, quantization, and verification ideas.

## Milestones

### v0.2 — Foundation

- quality gates: formatter, linter, type checker, clean CI
- KIR v1 schema and JSON round trip
- strict frontend diagnostics
- explicit quantization policy and calibration API
- CLI environment diagnostics
- reference models and evidence remain reproducible

### v0.5 — Usable preview

- PyTorch frontend documented as a stable supported subset
- at least one external model format, prioritized by user demand
- CMake and one embedded SDK integration
- hardware-backed validation on representative Cortex-M boards
- beginner, embedded, and research tutorials
- contribution, security, release, and compatibility policies

### v1.0 — Trustworthy release

- stable KIR v1 compatibility policy
- reproducible release artifacts
- verified target profiles and measured device benchmarks
- formal or bounded checking for the highest-risk generated kernels
- clear support matrix and migration policy
- external users can reproduce the reference certificates from a clean checkout

## Scope decisions

Kiln will expand its supported operators and frontends incrementally. It will
not weaken integer semantics, safety gates, or evidence standards to increase
the headline compatibility number. GUI products, cloud training, and a general
purpose runtime are outside the core project unless they directly improve the
compiler, its verification, or its learning experience.

## Decision rule

When a proposed feature conflicts with a guarantee, preserve the guarantee and
make the feature experimental or reject it with a precise diagnostic. When a
feature benefits one audience but harms the others, prefer a layered design:
simple defaults, inspectable internals, and extension points for researchers.

## Success measures

- a new user reaches a successful generated-C build in ten minutes
- a student can explain one KIR operation and its certificate
- an embedded engineer can reproduce RAM/ROM measurements on target hardware
- a researcher can serialize, modify, and re-run KIR without importing PyTorch
- every release has a clean-checkout reproduction command and a published
  evidence report
