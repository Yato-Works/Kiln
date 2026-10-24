# Getting started from a clean checkout

This is the shortest reproducible path for a student, researcher, or embedded
engineer who has just cloned Kiln. It intentionally starts with the torch-free
compiler core before introducing model export.

## 1. Install the core

Linux/macOS:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The core install does not require PyTorch. To export a PyTorch model, add the
optional frontend and test dependencies:

```bash
python -m pip install -e ".[frontend,test]"
```

## 2. Confirm the installation

```bash
python -m kiln doctor
python -m kiln selftest
python -m kiln ops
```

`kiln selftest` is deliberately board-free. It exercises KIR serialization,
static int32 safety, VM execution, and parity with the independent reference
interpreter without importing PyTorch or requiring a C compiler. If this passes,
the core installation is usable. `kiln doctor --json` is available for scripts
that need to inspect optional capabilities.

`kiln ops` is deliberately small: the seven operations shown are the complete
KIR v1 instruction set. Unsupported graph constructs fail with a structured
diagnostic; they are not silently approximated.

## 3. Compile a portable KIR bundle

With the PyTorch frontend installed:

```bash
python -m kiln export-kir examples/kws/model.py -o /tmp/kws.kir.json
python -m kiln validate /tmp/kws.kir.json
python -m kiln inspect /tmp/kws.kir.json --json
python -m kiln compile-kir /tmp/kws.kir.json -o /tmp/kws_model.c --runtime-header None
python -m kiln vm /tmp/kws.kir.json --json --trace
```

The last command does not import PyTorch. This is the recommended boundary for
language-independent tooling: frontend authors produce KIR, while compiler and
verification tooling consumes KIR.

The VM command also consumes only the bundle. It is the recommended path when
you do not have a board; its cycle count is a deterministic virtual-target model,
not a physical latency measurement. See [`docs/VM.md`](VM.md).

## 4. Run the deterministic checks

```bash
python -m pip install -e ".[test]"
python -m pytest tests/ -q
```

For a linked host image, use:

```bash
python -m kiln link examples/hazard/kiln.yaml --run
```

The report distinguishes static estimates from observed execution. A host run
does not count as evidence for a microcontroller target.

The shortest no-board tutorial is therefore:

```bash
python -m kiln selftest
python -m kiln validate examples/vm/hello.kir.json
python -m kiln vm examples/vm/hello.kir.json \
  --input examples/vm/input.json --json --trace
python -m kiln compile-kir examples/vm/hello.kir.json \
  -o build/tutorial_model.c --runtime-header None
python -m kiln link examples/vm/hello.kir.json --run \
  --output build/tutorial-link
```

This path is intentionally the first success criterion for students and
researchers. PyTorch export and native linking are separate capabilities, not
requirements for learning the KIR contract or running a model. The final two
commands add the standalone C99 artifact and a strict linked-image replay; the
link command consumes KIR directly and does not need the PyTorch frontend.

For a practical shipped model, repeat the same path with the Hazard student:

```bash
python -m kiln validate examples/hazard/hazard.kir.json
python -m kiln vm examples/hazard/hazard.kir.json \
  --input examples/hazard/data/example_tokens.json --json
python -m kiln link examples/hazard/hazard.kir.json --run \
  --output build/hazard-bundle-link
```

The bundle is already tokenized input-to-output KIR. The original tokenizer and
the C frontend remain available for end-to-end text integration, while this
path lets a user study the compiler and target behavior independently.

To classify raw text with the same tokenizer contract:

```bash
python -m examples.hazard.predict \
  --text "print(eval(user_input))" --json
```

The command is a convenience wrapper around the shipped KIR bundle and VM. It
reports resource metrics, but the Hazard student is explicitly a compact
certification example rather than a production security oracle.

## 5. Choose the next path

- Read [`KIR_COMPATIBILITY.md`](../KIR_COMPATIBILITY.md) before generating KIR
  from another language.
- Read [`TARGET_VALIDATION.md`](../TARGET_VALIDATION.md) before interpreting
  QEMU or hardware results.
- Use [`docs/SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md) to check whether an operator,
  frontend, or target is stable.
- Hardware integration starts from [`examples/hardware/README.md`](../examples/hardware/README.md).

## Troubleshooting

- `torch` missing: install `.[frontend]`; the core compiler is intentionally
  usable without it.
- Cross compiler missing: host builds and KIR validation remain available;
  `kiln doctor` reports the missing target toolchain explicitly.
- `--run` rejected for a bare-metal profile: configure a truthful system
  emulator or hardware runner. Do not substitute a host process and call it
  MCU evidence.
