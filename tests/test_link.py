"""Tests for Phase 4's link step (``kiln link``).

What is under test: the emitted translation unit is not merely *emitted*, it is
compiled under the strict flags, linked with the spec's own C frontend, measured
(section sizes, undefined symbols) and executed -- and the labels the linked
binary prints are compared against ``kiln.ref``, i.e. bit-exactness is checked on
a real program rather than on a string.

The suite skips itself when the host has no C compiler; the zero-malloc check
additionally needs binutils (``nm``/``objdump``), which the report marks as
unavailable instead of guessing.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from kiln.cli import main
from kiln.ir.kir import DType
from kiln.link import (
    STRICT_FLAGS,
    ToolchainError,
    detect_toolchain,
    harness_vectors,
    link_spec,
    parse_predictions,
    render_harness,
    to_input_domain,
)
from kiln.ref.ref import run_ref
from kiln.spec import load_spec, resolve_model_graph

REPO_ROOT = Path(__file__).resolve().parents[1]
HAZARD_SPEC = REPO_ROOT / "examples" / "hazard" / "kiln.yaml"
KWS_SPEC = REPO_ROOT / "examples" / "kws" / "kiln.yaml"


def _has_compiler() -> bool:
    try:
        detect_toolchain()
    except ToolchainError:
        return False
    return True


requires_compiler = pytest.mark.skipif(
    not _has_compiler(), reason="no C compiler on PATH (kiln link needs one)"
)


def _reference_labels(loaded, vectors) -> tuple[int, ...]:
    """The oracle's labels for the harness vectors, in the input tensor's domain."""
    desc = loaded.graph.tensors[loaded.graph.inputs[0]]
    return tuple(
        run_ref(
            loaded.graph,
            {loaded.graph.inputs[0]: to_input_domain(desc.dtype, vector)},
            loaded.weights,
        )[loaded.graph.outputs[0]][0]
        for vector in vectors
    )


# ------------------------------------------------------------ the harness


def test_harness_vectors_are_pinned() -> None:
    """The replay set is a reviewable constant, not a random draw."""
    assert harness_vectors(4) == (
        (0x00, 0x00, 0x00, 0x00),
        (0x00, 0xFF, 0x00, 0xFF),
        (0xFF, 0xFF, 0xFF, 0xFF),
        (0x80, 0x80, 0x80, 0x80),
        (0x01, 0x08, 0x0F, 0x16),
    )
    # Same length, same vectors -- the harness inherits Kiln's determinism.
    assert harness_vectors(96) == harness_vectors(96)
    assert len(harness_vectors(96)) == 5
    assert all(len(vector) == 96 for vector in harness_vectors(96))


def test_harness_vectors_reject_an_empty_input() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        harness_vectors(0)


def test_render_harness_is_a_pure_function_of_the_vectors() -> None:
    source = render_harness(4)

    assert source == render_harness(4)
    assert "static const uint8_t k_vectors[5][4]" in source
    assert "0x01, 0x08, 0x0F, 0x16" in source  # the ramp row, verbatim
    assert "kiln_predict(k_vectors[vector], 4)" in source
    assert 'printf("%d %d\\n"' in source
    # Explicit vectors are honoured (the harness can be replayed on demand).
    custom = render_harness(2, ((1, 2),))
    assert "k_vectors[1][2]" in custom and "0x01, 0x02" in custom


def test_render_harness_rejects_a_mismatched_vector() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        render_harness(4, ((1, 2),))


def test_parse_predictions_restores_index_order() -> None:
    stdout = "2 7\n0 3\n1 5\n"

    assert parse_predictions(stdout, expected=3) == (3, 5, 7)


def test_parse_predictions_reports_a_missing_vector() -> None:
    with pytest.raises(ValueError, match=r"missing vector\(s\) \[2\]"):
        parse_predictions("0 1\n1 2\n", expected=3)


def test_to_input_domain_follows_the_graph_dtype() -> None:
    """The C entry point takes bytes; an int8 graph input reads them signed."""
    assert to_input_domain(DType.UINT8, (0x00, 0x7F, 0x80, 0xFF)) == [0, 127, 128, 255]
    assert to_input_domain(DType.INT8, (0x00, 0x7F, 0x80, 0xFF)) == [0, 127, -128, -1]


# ------------------------------------------------------------ the toolchain


def test_strict_flags_match_the_example_suites() -> None:
    assert STRICT_FLAGS == ("-std=c99", "-O3", "-Wall", "-Wextra", "-Werror", "-pedantic")


@requires_compiler
def test_detect_toolchain_prefers_the_explicit_path() -> None:
    explicit = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")

    toolchain = detect_toolchain(explicit)

    assert toolchain.cc == str(Path(explicit))
    assert toolchain.name in ("gcc", "cc", "clang")


@requires_compiler
def test_detect_toolchain_honours_kiln_cc(monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    monkeypatch.setenv("KILN_CC", str(explicit))

    assert detect_toolchain().cc == str(Path(explicit))


def test_detect_toolchain_rejects_a_missing_compiler() -> None:
    with pytest.raises(ToolchainError, match="C compiler not found"):
        detect_toolchain("kiln-no-such-compiler")


# --------------------------------------------------------- the linked build


@requires_compiler
def test_link_toy_spec_builds_and_measures(temp_dir: Path, toy_spec_factory) -> None:
    spec = load_spec(toy_spec_factory())

    outcome = link_spec(spec, out_dir=temp_dir / "link", run=True)

    assert outcome.executable.is_file()
    assert [path.name for path in outcome.sources] == ["model.c", "harness.c"]
    assert outcome.symbols_inspected
    assert outcome.malloc_symbols == frozenset()
    assert outcome.text_bytes is not None and outcome.text_bytes > 0
    assert outcome.arena_matches is True          # the linked .bss reserves the arena
    assert outcome.rom_covers_weights is True
    assert len(outcome.vectors) == len(outcome.predictions) == 5
    assert all(0 <= label < 3 for label in outcome.predictions)


@requires_compiler
def test_link_hazard_binary_is_bit_exact_against_the_reference(temp_dir: Path) -> None:
    """The link step's strongest claim: a linked *program* agrees with kiln.ref."""
    spec = load_spec(HAZARD_SPEC)
    loaded = resolve_model_graph(spec)

    outcome = link_spec(spec, out_dir=temp_dir / "hazard", run=True)

    assert outcome.predictions == _reference_labels(loaded, outcome.vectors)
    assert len(set(outcome.predictions)) > 1     # the bag discriminates, it does not collapse

    # The spec's own C frontend is part of the verified image...
    assert [path.name for path in outcome.sources] == [
        "model.c",
        "hazard_tokenizer.c",
        "hazard_pipeline.c",
        "harness.c",
    ]
    # ...and the linked image still respects the audited envelope.
    assert outcome.report.arena_ram_bytes == 1344
    assert outcome.arena_matches is True
    assert outcome.rom_covers_weights is True
    assert outcome.malloc_symbols == frozenset()


@requires_compiler
def test_link_kws_int8_binary_is_bit_exact_against_the_reference(temp_dir: Path) -> None:
    """The int8 path: the C side reads the harness bytes two's-complement."""
    spec = load_spec(KWS_SPEC)
    loaded = resolve_model_graph(spec)
    desc = loaded.graph.tensors[loaded.graph.inputs[0]]
    assert desc.dtype is DType.INT8

    outcome = link_spec(spec, out_dir=temp_dir / "kws", run=True)

    assert outcome.predictions == _reference_labels(loaded, outcome.vectors)
    # The Phase 1 certificate, re-measured on the linked image.
    assert outcome.report.arena_ram_bytes == 1992
    assert outcome.report.total_rom_bytes == 39614
    assert outcome.report.total_macs == 314342
    assert "kws_frontend.c" in {path.name for path in outcome.sources}
    assert outcome.arena_matches is True
    assert outcome.malloc_symbols == frozenset()


@requires_compiler
def test_link_without_running_reports_no_predictions(temp_dir: Path, toy_spec_factory) -> None:
    spec = load_spec(toy_spec_factory())

    outcome = link_spec(spec, out_dir=temp_dir / "link")

    assert outcome.predictions == ()
    assert outcome.executable.is_file()


@requires_compiler
def test_link_skips_a_missing_frontend_file(temp_dir: Path, toy_spec_factory) -> None:
    spec_path = toy_spec_factory(frontend={"type": "custom", "source_files": ["absent.c"]})

    outcome = link_spec(load_spec(spec_path), out_dir=temp_dir / "link")

    assert [path.name for path in outcome.sources] == ["model.c", "harness.c"]
    assert outcome.executable.is_file()


@requires_compiler
def test_link_surfaces_a_frontend_compile_error(temp_dir: Path, toy_spec_factory) -> None:
    (temp_dir / "broken.c").write_text("int broken(void) { return ; }\n", encoding="utf-8")
    spec_path = toy_spec_factory(frontend={"type": "custom", "source_files": ["broken.c"]})

    with pytest.raises(ToolchainError) as error:
        link_spec(load_spec(spec_path), out_dir=temp_dir / "link")

    message = str(error.value)
    assert "compiling broken.c failed" in message
    assert "-std=c99" in message  # the command line is part of the diagnosis


@requires_compiler
def test_strict_flags_make_a_warning_fatal(temp_dir: Path, toy_spec_factory) -> None:
    """`-Werror` is load-bearing: a sloppy frontend fails the link, not the firmware."""
    (temp_dir / "noisy.c").write_text(
        "int noisy(void) { int unused; return 0; }\n", encoding="utf-8"
    )
    spec_path = toy_spec_factory(frontend={"type": "custom", "source_files": ["noisy.c"]})

    with pytest.raises(ToolchainError, match="Werror"):
        link_spec(load_spec(spec_path), out_dir=temp_dir / "link")


# ------------------------------------------------------------------- the CLI


@requires_compiler
def test_cli_link_prints_and_writes_the_report(
    temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = temp_dir / "link"

    ret = main(["link", str(HAZARD_SPEC), "-o", str(out_dir), "--run"])

    assert ret == 0
    captured = capsys.readouterr()
    assert "Kiln Link Report: hazard-sniffer" in captured.out
    assert "[LINKED]" in captured.out
    assert "zero-malloc observed" in captured.out
    assert "audited arena +" in captured.out      # the frontend brings its own .bss
    assert "Harness replay (5 vectors)" in captured.out
    assert "zeros=" in captured.out

    report = (out_dir / "link_report.txt").read_text(encoding="utf-8")
    assert report.splitlines()[1].strip() == "Kiln Link Report: hazard-sniffer"
    assert report.endswith("=" * 73 + "\n")


@requires_compiler
def test_cli_link_without_a_compiler_errors(
    temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ret = main(["link", str(HAZARD_SPEC), "-o", str(temp_dir / "link"), "--cc", "kiln-no-such-cc"])

    assert ret == 1
    captured = capsys.readouterr()
    assert "[ERROR] Link failed" in captured.err
    assert "C compiler not found: kiln-no-such-cc" in captured.err


@requires_compiler
def test_cli_link_budget_violation_returns_failure(
    temp_dir: Path, toy_spec_factory, capsys: pytest.CaptureFixture[str]
) -> None:
    spec_path = toy_spec_factory(budget={"ram_arena_max_bytes": 4, "rom_total_max_bytes": 8})
    out_dir = temp_dir / "link"

    ret = main(["link", str(spec_path), "-o", str(out_dir)])

    assert ret == 1
    captured = capsys.readouterr()
    assert "EXCEEDED" in captured.out
    assert "[BUDGET EXCEEDED]" in captured.err
    # The artifacts stay on disk so the violation can be diagnosed.
    assert (out_dir / "model.c").is_file()
    assert (out_dir / "link_report.txt").is_file()


