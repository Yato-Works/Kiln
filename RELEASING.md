# Releasing Kiln

This is the release checklist for maintainers. A release is not ready merely
because the package builds: the wheel, source distribution, KIR contract, and
evidence must agree.

## Local release candidate

From a clean checkout:

```bash
python -m pip install -e ".[test,dev]"
python -m kiln doctor --strict
python -m kiln selftest --json
python -m pytest tests/ -q -rs
python -m ruff format --check kiln tests scripts/capture_serial.py
python -m ruff check kiln tests scripts/capture_serial.py
python -m mypy kiln
python -m build --sdist --wheel --outdir dist
```

Inspect the artifacts, then test the wheel in a fresh environment without the
editable checkout on `PYTHONPATH`:

```bash
python -m venv /tmp/kiln-release-venv
/tmp/kiln-release-venv/bin/python -m pip install dist/kiln-*.whl
/tmp/kiln-release-venv/bin/python -m kiln selftest --json
/tmp/kiln-release-venv/bin/python -m kiln ops
```

On Windows, use a dedicated `.venv-release\Scripts\python.exe` instead of the
POSIX path above.

## Artifact boundary

The wheel contains the `kiln` Python package and CLI only. Official model
bundles, datasets, example frontends, and evidence remain repository artifacts;
they are not silently copied into the PyPI wheel. Verify both distributions:

```bash
python -m pip install --no-deps --force-reinstall dist/kiln-*.whl
python -c "import kiln; print(kiln.__version__)"
```

## Publish and tag

1. Update `kiln/_version.py`, `CHANGELOG.md`, and the release date.
2. Run the complete candidate checklist and attach the CI run to the release.
3. Publish the sdist and wheel through the trusted project publishing workflow.
4. Create the matching `v<version>` Git tag and GitHub release notes.
5. Verify installation from the published index in a fresh environment.

Never overwrite an existing version. If an artifact or certificate changes,
increment the version or document the compatibility decision explicitly.
