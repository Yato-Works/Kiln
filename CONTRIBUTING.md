# Contributing to Kiln

Thank you for helping build a reproducible TinyML compiler. Kiln is deliberately
small and strict: a useful contribution explains its contract, includes a test,
and records what is measured versus what is only estimated.

## Development setup

The core workflow does not require PyTorch:

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test,dev]"
python -m kiln doctor --strict
python -m kiln selftest
```

Before opening a pull request, run the same gates as CI:

```bash
python -m ruff format --check kiln tests scripts/capture_serial.py
python -m ruff check kiln tests scripts/capture_serial.py
python -m mypy kiln
python -m pytest tests/ -q -rs
python -m build --sdist --wheel --outdir dist
```

## Change boundaries

- KIR schema or semantics changes require an explicit compatibility discussion,
  schema tests, and an update to [`KIR_COMPATIBILITY.md`](KIR_COMPATIBILITY.md).
- New frontends belong under `kiln/frontend/<name>/` and must fail with a useful
  diagnostic for unsupported graphs.
- New target backends must preserve the KIR bundle boundary and state whether
  evidence is host, emulator, or physical hardware evidence. See
  [`TARGET_VALIDATION.md`](TARGET_VALIDATION.md).
- New guarantees need a test or captured evidence. Do not describe a virtual
  cycle estimate as silicon latency.
- Keep generated model assets reproducible. Add or update the exporter next to
  the artifact and test that regeneration is byte-for-byte stable.

## Pull requests

Keep a pull request focused on one logical change. Describe the user-visible
outcome, the evidence used to validate it, and any known limitations. Small
documentation and test-only pull requests are welcome.

## Commit style

Use a short imperative subject with a conventional prefix when practical:

```text
feat: add a KIR bundle importer
fix: reject malformed quantization metadata
docs: clarify QEMU evidence boundaries
test: cover maxpool tie-breaking
```

Please do not commit build directories, local virtual environments, downloaded
datasets, or hardware secrets.
