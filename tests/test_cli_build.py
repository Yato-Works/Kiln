"""End-to-end tests for the declarative ``kiln build`` pipeline (Phase 3 Step 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kiln.cli import main
from kiln.spec import load_spec, resolve_model_graph
from kiln.spec.pipeline import SafetyViolationError, emit_spec_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_emits_c_certificate_and_budget_gate(
    toy_spec_factory, capsys: pytest.CaptureFixture[str]
) -> None:
    spec_path = toy_spec_factory()

    ret = main(["build", str(spec_path)])

    assert ret == 0
    out_dir = spec_path.parent / "build"
    model_c = out_dir / "model.c"
    assert model_c.exists()
    content = model_c.read_text(encoding="utf-8")
    assert "kiln_predict" in content
    assert "Kiln Model Static Budget Certificate" in content
    assert (out_dir / "budget_certificate.txt").is_file()

    captured = capsys.readouterr()
    assert "[SUCCESS]" in captured.out
    assert "[BUDGET] arena RAM" in captured.out
    assert "[BUDGET] total ROM" in captured.out
    assert "OK" in captured.out


def test_build_explicit_output_dir_and_frontend_copy(
    toy_spec_factory, temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (temp_dir / "stub_frontend.c").write_text("/* stub */\n", encoding="utf-8")
    spec_path = toy_spec_factory(
        frontend={
            "type": "custom",
            "source_files": ["stub_frontend.c"],
            "header_files": ["missing.h"],
        }
    )
    out_dir = temp_dir / "dist"

    ret = main(["build", str(spec_path), "-o", str(out_dir)])

    assert ret == 0
    assert (out_dir / "model.c").is_file()
    assert (out_dir / "stub_frontend.c").read_text(encoding="utf-8") == "/* stub */\n"
    assert not (out_dir / "missing.h").exists()
    captured = capsys.readouterr()
    assert "frontend file not found" in captured.err


def test_build_budget_violation_returns_failure(
    toy_spec_factory, capsys: pytest.CaptureFixture[str]
) -> None:
    """Over-budget specs fail the build, but artifacts stay for diagnosis."""
    spec_path = toy_spec_factory(
        budget={
            "ram_arena_max_bytes": 8,
            "rom_total_max_bytes": 8,
            "macs_max": 1,
        }
    )

    ret = main(["build", str(spec_path)])

    assert ret == 1
    captured = capsys.readouterr()
    assert "[BUDGET EXCEEDED]" in captured.err
    assert "MACs" in captured.err
    assert captured.out.count("EXCEEDED") == 3
    assert (spec_path.parent / "build" / "model.c").exists()


def test_build_input_shape_mismatch_warns(
    toy_spec_factory, capsys: pytest.CaptureFixture[str]
) -> None:
    """The spec's declared input is checked softly against the resolved graph."""
    spec_path = toy_spec_factory(input_shape=(7,))

    ret = main(["build", str(spec_path)])

    assert ret == 0
    assert "does not match" in capsys.readouterr().err


def test_build_missing_spec_returns_error(
    temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ret = main(["build", str(temp_dir / "nope.yaml")])

    assert ret == 1
    assert "[ERROR] Build failed" in capsys.readouterr().err


def test_build_hazard_reference_spec(temp_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`kiln build` on the shipped hazard reference spec (full pipeline)."""
    spec_path = REPO_ROOT / "examples" / "hazard" / "kiln.yaml"
    assert spec_path.is_file(), "reference spec must be checked in"
    out_dir = temp_dir / "hazard"

    ret = main(["build", str(spec_path), "-o", str(out_dir)])

    assert ret == 0
    assert (out_dir / "model.c").is_file()
    assert (out_dir / "hazard_tokenizer.c").is_file()
    assert (out_dir / "hazard_pipeline.h").is_file()
    assert (out_dir / "budget_certificate.txt").is_file()
    captured = capsys.readouterr()
    assert "[SUCCESS] Built hazard-sniffer" in captured.out
    assert "[BUDGET] arena RAM" in captured.out


def test_build_int32_overflow_aborts_and_fails(
    overflow_spec_factory, capsys: pytest.CaptureFixture[str]
) -> None:
    """An overflowing model fails the build *and* leaves no artifacts to flash."""
    spec_path = overflow_spec_factory()
    out_dir = spec_path.parent / "build"

    ret = main(["build", str(spec_path), "-o", str(out_dir)])

    assert ret == 1
    captured = capsys.readouterr()
    assert "[SAFETY VIOLATION]" in captured.err
    assert "int32 accumulator overflow detected" in captured.err
    assert "no artifacts written" in captured.err
    # The certificate still explains the refusal...
    assert "WARNING: Potential int32 overflow detected" in captured.out
    # ...but neither artifact survived.
    assert not (out_dir / "model.c").exists()
    assert not (out_dir / "budget_certificate.txt").exists()


def test_emit_spec_artifacts_refuses_overflow_before_writing(
    overflow_spec_factory, temp_dir: Path
) -> None:
    """The gate lives in the emitter, so ``kiln link`` inherits it too."""
    spec = load_spec(overflow_spec_factory())
    loaded = resolve_model_graph(spec)
    out_dir = temp_dir / "emitted"

    with pytest.raises(SafetyViolationError, match="int32 accumulator overflow detected") as exc:
        emit_spec_artifacts(spec, loaded, out_dir)

    assert exc.value.report.int32_overflow_free is False
    assert not out_dir.exists()  # not even an empty directory
