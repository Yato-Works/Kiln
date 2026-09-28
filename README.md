# Kiln

**Kiln is the "QEMU of TinyML": a compiler framework that turns a PyTorch model into
deterministic, zero-malloc, bit-exact C99 for microcontrollers.**

The emitted translation unit is a single `kiln_predict(const uint8_t *input, int32_t length)`
function plus static tables — no runtime, no interpreter, no `malloc`. Arena and ROM
footprints are known before you flash, and the generated C is bit-identical to an
independent Python reference (`kiln.ref`) and to the PyTorch model it came from.

* **Integer-only semantics** — int8 activations, int32 accumulators, Q31 multiplier +
  shift requantisation, fixed tie-breaking, no floating point, no data-dependent control flow.
* **Evidence over adjectives** — every guarantee is a test: differential fuzzing,
  adversarial edge vectors, and bit-exact comparison of gcc-built C against the reference.
* **Budget certificate** — a static arena/ROM/MAC audit is printed and written next to the C.

## Status

| Phase | Scope | State |
| :---: | :--- | :--- |
| 0 | Compiler core invariants (KIR, differential / edge-case / tie-break fuzzing) | **DONE** |
| 1 | Backend validation: MLPerf-Tiny KWS — int8, raw PCM → label, all in C | **DONE** |
| 2 | Synthetic-data distillation pipeline (teacher LLM → int8 student) | **DONE** (M2.0–M2.4) |
| 3 | Declarative `kiln.yaml` → `kiln build` → compile-time architecture search | **DONE** (hazard only; KWS search not yet available) |
| 4 | Certified trust: `kiln link` (strict compile + link + image measurement + replay) | **IN PROGRESS** |
| 5-6 | Moonshots, ecosystem | planned |

[ROADMAP.md](ROADMAP.md) holds the phase goals, the Phase 1 close-out and the known gaps;
[KIR.md](KIR.md) is the frozen intermediate-representation contract.

## Quick start

```bash
cd /path/to/kiln
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

Three commands — no dataset download needed:

```bash
python -m kiln ops                            # 1. the 7 IR operations (the whole instruction set)
python -m kiln audit examples/kws/model.py    # 2. static footprint certificate (add --json for machines)
python -m pytest tests/ -q                    # 3. every guarantee (all tests, ~2 min)
```

The Phase 4 link tests compile and run C, so they skip themselves when the host
has no C compiler; nothing else in the suite needs one.

Emit the C yourself and build it with any C99 compiler:

```bash
python -m kiln compile examples/kws/model.py -o kws_model.c --runtime-header None
gcc -std=c99 -O2 -Wall -Wextra -Werror -pedantic -c kws_model.c -o kws_model.o
```

`--runtime-header None` emits a self-contained translation unit; the default
(`kiln_runtime.h`) is for projects that ship their own shim. `kiln verify <model.py>`
adds gcc-based differential fuzzing against the Python reference, and `kiln audit --json`
returns the same certificate as structured data. `kiln audit --intervals` prints the
Interval Boundary Propagation report — per-tensor value bounds and per-node
accumulator/saturation ranges — that also lands in the certificate's
`Accumulator Safety` line as a static proof of int32 non-overflow.

### Declarative builds (Phase 3)

A task is a YAML spec, and the budget gate is part of the build:

```bash
python -m kiln build examples/hazard/kiln.yaml            # kiln.yaml -> model.c + budget_certificate.txt
python -m kiln search examples/hazard/kiln-search.yaml    # RAM/ROM/capacity Pareto front (hazard only)
python -m kiln link examples/hazard/kiln.yaml --run       # strict compile + link + image measurement + replay
```

`kiln build` resolves `model.entry_point`, audits the resolved graph against the
spec's `budget:` and exits non-zero if a limit is exceeded (the artifacts are still
written, so the failure can be diagnosed). The one failure that writes *nothing* is
an int32 accumulator overflow: the certificate's `Accumulator Safety` proof is a
hard gate in `build`, `compile` and `link`, so a model that cannot be accumulated
safely never produces a `model.c` to flash by accident. `kiln search` enumerates a grid of
factory parameters and audits **every candidate at compile time** — a footprint
depends on shapes and weight extents, never on weight values, so a search needs no
training run — then writes `search_report.md` and `search_results.csv` next to the
spec. Both reference specs reproduce the Phase 1 and Phase 2 certificates exactly:
1,992 B / 39,614 B / 314,342 MACs for KWS and 1,344 B / 2,096 B for the hazard
student.

> **Note:** `kiln search` currently only has a reference spec for the hazard
> example. KWS search is not yet available.

`kiln link` closes that loop on a **real binary** (Phase 4): it compiles `model.c`,
the declared frontend and a generated replay harness with
`-std=c99 -O3 -Wall -Wextra -Werror -pedantic`, links an executable, and reports
what the image actually costs -- `.bss` against the audited arena, read-only data
against the audited weights, and the undefined symbols of the *model object*,
which is where the zero-malloc guarantee becomes an observation
(`no allocation entry point referenced`) instead of a claim. With `--run` the
harness replays five fixed byte vectors and the tests compare the labels a linked
program prints against `kiln.ref`, for the int8 KWS model and the uint8 hazard
student alike.

## Phase 1 certificate (keyword spotting, shipped model)

`python -m examples.kws.evaluate` reprints these numbers on any machine with `gcc`:

| Metric | Value | Contract |
| :--- | :--- | :--- |
| int8 C99 test accuracy | **91.01 %** (734 held-out clips) | >= 85 % (asserted) |
| FP32 test accuracy | 90.74 % | — |
| Quantisation delta | +0.27 % | no accuracy loss |
| Emitted C vs FP32 predictions | 98.50 % agreement | >= 95 % (asserted) |
| Arena RAM / total ROM | **1,992 B** / 39,614 B | < 2 KB / < 40 KB (asserted) |
| Raw PCM → label (C frontend + emitted graph) | 24/24 bit-exact vs reference, 23/24 correct, 24/24 identical to the PyTorch pipeline | spans all 12 classes |

Two caveats, both documented in `ROADMAP.md`:

* the **compiler** is exactly reproducible, the **training run** is not — CUDA kernels are
  not bit-stable, so re-running `train_aug` lands within ~1 point of the shipped checkpoint
  (selection therefore uses the validation split, never the test split);
* 92.2 % (the MLPerf reference) is not reached because the Phase 1 corpus is 4,890 clips
  instead of ~105 K; training accuracy saturates at 99.85 % while validation plateaus near 91.5 %.

## Datasets and artifacts

| Path | Size | Tracked | Notes |
| :--- | :--- | :---: | :--- |
| `examples/kws/kws_processed_data.pt` | 9.2 MB | yes | frozen 4,890-clip log-mel corpus + split — the certificate input |
| `examples/kws/kws_qat_weights.pt` | 0.1 MB | yes | the shipped int8 checkpoint (27/56 layout, seed 777) |
| `examples/kws/kws_qat_weights_27_55.pt` | 0.1 MB | yes | the previous 27/55 checkpoint; `python -m examples.kws.qat_eval --architecture previous --weights examples/kws/kws_qat_weights_27_55.pt` reprints the v1 certificate (89.10 % int8 C, 1,972 B arena) |
| `examples/kws/kws_trained_weights.pt` | 0.1 MB | yes | dB-domain checkpoint written by `examples/kws/train.py` and consumed by `ptq_test.py` (`ROADMAP.md` §4, finding 3) |
| `speech_commands_test_set_v0.02.tar.gz` | 107 MB | no | raw WAVs for the end-to-end audio tests; without it those tests report `skipped` |

The raw archive is a separate download (Speech Commands v0.02 test set) and is expected
next to the repository root; it is never vendored.

## Repository layout

```
kiln/ir/         KIR: ops, dtypes, quantisation params, validation, best-fit arena allocator
kiln/frontend/   PyTorch -> KIR lowering
kiln/analysis/   interval boundary propagation: int32 accumulator + saturation audit
kiln/budget/     static arena/ROM/MAC audit and certificate formatting
kiln/backend/    C99 emitter (one static arena, zero-malloc)
kiln/ref/        independent integer reference interpreter (the oracle)
kiln/spec/       declarative kiln.yaml schema + model/graph resolution
kiln/search/     compile-time footprint search: grid enumeration, Pareto front, report
kiln/link/       strict compile + link, image measurement (sections/symbols) and replay harness
kiln/cli.py      `compile`, `build`, `search`, `link`, `audit`, `verify`, `ops`
examples/kws/    Phase 1: model, training recipes, evaluation, fixed-point C frontend
examples/hazard/ Phase 2: M2.0 fixed-point feature contract + token-bag graph
tests/           the guarantees: diff, fuzz, edge cases, budget, CLI, KWS, PCM→C E2E
```

## What is guaranteed, and what is not

* Guaranteed: identical C output for identical inputs, zero dynamic allocation,
  bit-exact agreement between the emitted C and `kiln.ref`, and a static footprint bound.
  Placement is interval-aware (`kiln/ir/arena.py`): the allocator keeps the historical
  First-Fit layout unless a best-fit search *strictly* shrinks the arena, so re-emitting
  an existing model cannot move its offsets and a 20 B gap shortfall no longer costs a
  whole 1.3 KB block (`ROADMAP.md` §3). Since Phase 4 those guarantees are also
  *observed* on a linked image (`kiln link`): the model object references no
  allocation entry point, the linked `.bss` reserves the audited arena, and the
  labels a linked program prints match the reference oracle.
* Not guaranteed (yet): execution on real silicon (the latency figure is an
  upper-bound cycle estimate, and `kiln link` targets the host until a cross
  toolchain plus a target profile exist — see Phase 4), and bit-identical
  *training* runs.

## Evidence

Every number in this README is reproduced by a single script — nothing is
hand-typed:

```bash
python scripts/reproduce_evidence.py    # re-runs every command, writes EVIDENCE.md
```

`EVIDENCE.md` quotes the verbatim stdout of each run (the test suite, `kiln build`,
`gcc -Werror -pedantic` + `nm -u`, the safety refusal, the KWS and hazard
evaluations, the training repro and the data-provenance manifest). The showcase
GIFs are rendered from the same real runs by `scripts/render_demo_gif.py`:

* [scripts/demo_showcase1.gif](scripts/demo_showcase1.gif) — the "2KB brain":
  one C file, a clean `gcc -std=c99 -O2 -Wall -Wextra -Werror -pedantic` compile,
  and `nm -u` showing `abort` as the only unresolved symbol.
* [scripts/demo_showcase2.gif](scripts/demo_showcase2.gif) — the refusal: a model
  whose int32 accumulator genuinely overflows (140,000 MACs at the 127·127
  maximum product) is rejected with exit 1 and *no* C file is written.

## License

MIT — see [LICENSE](LICENSE).
