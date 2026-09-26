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
| 2 | Synthetic-data distillation pipeline (teacher LLM → int8 student) | next |
| 3-6 | Declarative `kiln.yaml`, certified trust, moonshots, ecosystem | planned |

[ROADMAP.md](ROADMAP.md) holds the phase goals, the Phase 1 close-out and the known gaps;
[KIR.md](KIR.md) is the frozen intermediate-representation contract.

## Quick start

```bash
git clone <this-repository> kiln && cd kiln
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

Three commands — no dataset download, no compiler needed:

```bash
python -m kiln ops                            # 1. the 7 IR operations (the whole instruction set)
python -m kiln audit examples/kws/model.py    # 2. static footprint certificate (add --json for machines)
python -m pytest tests/ -q                    # 3. every guarantee (75 tests, ~1 min)
```

Emit the C yourself and build it with any C99 compiler:

```bash
python -m kiln compile examples/kws/model.py -o kws_model.c --runtime-header None
gcc -std=c99 -O2 -Wall -Wextra -Werror -pedantic -c kws_model.c -o kws_model.o
```

`--runtime-header None` emits a self-contained translation unit; the default
(`kiln_runtime.h`) is for projects that ship their own shim. `kiln verify <model.py>`
adds gcc-based differential fuzzing against the Python reference, and `kiln audit --json`
returns the same certificate as structured data.

## Phase 1 certificate (keyword spotting, shipped model)

`python -m examples.kws.evaluate` reprints these numbers on any machine with `gcc`:

| Metric | Value | Contract |
| :--- | :--- | :--- |
| int8 C99 test accuracy | **89.10 %** (734 held-out clips) | >= 85 % (asserted) |
| FP32 test accuracy | 88.69 % | — |
| Quantisation delta | +0.41 % | no accuracy loss |
| Emitted C vs FP32 predictions | 99.18 % agreement | >= 95 % (asserted) |
| Arena RAM / total ROM | **1,972 B** / 39,197 B | < 2 KB / < 40 KB (asserted) |
| Raw PCM → label (C frontend + emitted graph) | 24/24 bit-exact vs reference, 23/24 correct | spans all 12 classes |

Two caveats, both documented in `ROADMAP.md`:

* the **compiler** is exactly reproducible, the **training run** is not — CUDA kernels are
  not bit-stable, so re-running `train_aug` lands within ~1 point of the shipped checkpoint
  (selection therefore uses the validation split, never the test split);
* 92.2 % (the MLPerf reference) is not reached because the Phase 1 corpus is 4,890 clips
  instead of ~105 K; training accuracy saturates at 99.85 % while validation plateaus near 91 %.

## Datasets and artifacts

| Path | Size | Tracked | Notes |
| :--- | :--- | :---: | :--- |
| `examples/kws/kws_processed_data.pt` | 9.2 MB | yes | frozen 4,890-clip log-mel corpus + split — the certificate input |
| `examples/kws/kws_qat_weights.pt` | 0.1 MB | yes | the shipped int8 checkpoint |
| `examples/kws/kws_trained_weights.pt` | 0.1 MB | yes | dB-domain checkpoint used by the deployability study in `ROADMAP.md` §4.3 |
| `speech_commands_test_set_v0.02.tar.gz` | 107 MB | no | raw WAVs for the end-to-end audio tests; without it those tests report `skipped` |

The raw archive is a separate download (Speech Commands v0.02 test set) and is expected
next to the repository root; it is never vendored.

## Repository layout

```
kiln/ir/         KIR: ops, dtypes, quantisation params, validation, tie-break rules
kiln/frontend/   PyTorch -> KIR lowering
kiln/budget/     static arena/ROM/MAC audit and certificate formatting
kiln/backend/    C99 emitter (in-place LIFO arena, zero-malloc)
kiln/ref/        independent integer reference interpreter (the oracle)
kiln/cli.py      `compile`, `audit`, `verify`, `ops`
examples/kws/    Phase 1: model, training recipes, evaluation, fixed-point C frontend
tests/           the guarantees: diff, fuzz, edge cases, budget, CLI, KWS, PCM→C E2E
```

## What is guaranteed, and what is not

* Guaranteed: identical C output for identical inputs, zero dynamic allocation,
  bit-exact agreement between the emitted C and `kiln.ref`, and a static footprint bound.
* Not guaranteed (yet): execution on real silicon (the latency figure is an upper-bound
  cycle estimate, see Phase 4), and bit-identical *training* runs.

## License

MIT — see [LICENSE](LICENSE).
