# Kiln VM

Kiln VM is the first-class host target for KIR v1. It exists so that students,
researchers, and compiler contributors can execute a model without owning an
MCU board. It consumes the frozen `kiln.kir.bundle` JSON format directly; it
does not import PyTorch and it does not silently lower unsupported operations.

## Run a bundle

```bash
python -m kiln export-kir examples/kws/model.py -o model.kir.json
python -m kiln validate model.kir.json
python -m kiln vm model.kir.json --json --trace
```

Traces are reviewable artifacts and can be replayed:

```bash
python -m kiln vm model.kir.json --trace --trace-output run.trace.json
python -m kiln vm model.kir.json --replay-trace run.trace.json --json
```

For negative testing, `--max-cycles N` enforces a virtual execution budget and
`--fault-node NODE` deliberately fails before a named KIR node. These controls
are intended for CI, tutorials, and fault-injection experiments.

For a single-input graph, `--input` may point to a JSON array. For multiple
inputs, provide an object keyed by tensor name. `--input -` reads the same JSON
document from standard input. If omitted, the VM uses deterministic zero-valued
inputs, which is useful for smoke tests.

```json
[-128, 0, 127]
```

The `--json` output contains `outputs`, `metrics`, and, with `--trace`, one event
for each KIR node. Each event records the output element count, range, and a
SHA-256 digest instead of dumping the entire tensor. This keeps traces useful
for large models while making accidental changes obvious.

The repository's board-free smoke test is runnable from a clean checkout:

```bash
python -m kiln vm examples/vm/hello.kir.json \
  --input examples/vm/input.json --json --trace
```

The same frozen bundle can be taken all the way to a strict linked host image:

```bash
python -m kiln validate examples/vm/hello.kir.json
python -m kiln compile-kir examples/vm/hello.kir.json \
  -o build/tutorial_model.c --runtime-header None
python -m kiln link examples/vm/hello.kir.json --run \
  --output build/tutorial-link
```

Direct bundle linking does not import PyTorch. It generates the same deterministic
replay harness used by declarative specs, measures the linked image, and checks
the five fixed predictions against the KIR reference path.

## Execution contract

The VM supports the complete KIR v1 instruction set:

| Instruction | VM behavior |
| --- | --- |
| `embedding` | uint8 vocabulary lookup into an int8 table |
| `conv1d` | signed int8 convolution, int32 accumulation, KIR requantization |
| `matmul` | signed int8 matrix multiplication with int32 accumulation |
| `add` | elementwise signed int8 addition with saturating requantization |
| `lut` | 256-entry signed int8 lookup |
| `maxpool1d` | deterministic windowed maximum |
| `argmax` | first-index-wins reduction |

The VM validates the graph, weights, shapes, dtypes, quantization parameters,
and static int32 safety before executing. A malformed, unsupported, or
unverified bundle fails closed with a diagnostic.

## Metrics

The reported `cycles` value is a deterministic virtual-instruction cost model.
It is useful for comparing KIR graphs and validating regressions, but it is not
a clock-cycle measurement for an RP2040, QEMU machine, or any other CPU.

- `arena_bytes`: the same statically planned tensor arena used by the C99 backend.
- `peak_concurrent_bytes`: the liveness lower bound from the planner.
- `rom_bytes`: referenced weights plus LUT tables.
- `allocation_bytes`: guest-side dynamic allocation contract; currently always
  zero for one invocation after VM construction. Python host allocations are
  intentionally not represented by this guest metric.
- `static_cycle_upper_bound`: the existing conservative budget estimate.
- `trace`: the ordered KIR node execution, including per-node virtual cycles.

The VM is intentionally a host virtual target, not a substitute for silicon
evidence. A future backend may implement the same KIR bundle through QEMU,
Renode, or a board adapter and must report its own measured evidence separately.

## Contributing a target backend

Target adapters should preserve the KIR v1 bundle boundary and provide a
reproducible runner plus evidence. A brave contributor with an RP2040, RISC-V
board, or another supported microcontroller would be very welcome: please use
the [hardware runner contract](../TARGET_VALIDATION.md), record the exact board,
toolchain, SDK, clock, and commands, and submit the captured certificate. No
board is required to use or develop Kiln VM.
