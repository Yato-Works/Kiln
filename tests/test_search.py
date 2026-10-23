"""Tests for the compile-time architecture search (``kiln search``, Phase 3 M3.2).

The claim under test: a Kiln footprint is a *compile-time* quantity, so the
whole declared grid can be enumerated, audited and compared without a training
run -- exactly, deterministically, and with the failures visible instead of
fatal.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
import yaml

from examples.hazard.graph import HAZARD_CLASSES, build_bag_candidate
from examples.hazard.tokenizer import VOCAB_SIZE
from kiln.cli import main
from kiln.search import (
    capacity_classes,
    enumerate_grid,
    format_search_report,
    pareto_front,
    run_search,
    search_results_csv,
)
from kiln.search.pareto import dominates
from kiln.spec import load_spec, load_spec_from_dict, resolve_model_graph

REPO_ROOT = Path(__file__).resolve().parents[1]
HAZARD_SEARCH_SPEC = REPO_ROOT / "examples" / "hazard" / "kiln-search.yaml"

#: Shape-parameterized factory fixture: the same bag shape as the Phase 2 graph,
#: but for any ``(tokens, dim, labels)`` -- and with a limit that makes one grid
#: point illegal, which is how the error-row path gets exercised.
TOY_SEARCH_MODEL_SOURCE = '''"""Toy searchable kiln spec fixture: a parametric bag classifier."""

from __future__ import annotations

from kiln.frontend.pytorch import compute_multiplier_and_shift
from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc

VOCAB = 256


def build(tokens: int = 4, dim: int = 4, labels: int = 3) -> tuple[Graph, dict]:
    """Graph + placeholder weights for one grid point (no training data needed)."""
    tensors = {
        "tokens": TensorDesc("tokens", shape=(tokens,), dtype=DType.UINT8),
        "emb": TensorDesc("emb", shape=(tokens, dim), dtype=DType.INT8),
        "logits": TensorDesc(
            "logits",
            shape=(tokens, labels),
            dtype=DType.INT8,
            quant=QuantParam(*compute_multiplier_and_shift(127.0 / 37772)),
        ),
        "pooled": TensorDesc("pooled", shape=(1, labels), dtype=DType.INT8),
        "label": TensorDesc("label", shape=(), dtype=DType.INT8),
    }
    nodes = [
        Node(OpType.EMBEDDING, "op_emb", inputs=["tokens"], outputs=["emb"], attrs={"weight": "w_emb"}),
        Node(OpType.MATMUL, "op_mm", inputs=["emb"], outputs=["logits"], attrs={"weight": "w_mm"}),
        Node(
            OpType.MAXPOOL1D,
            "op_pool",
            inputs=["logits"],
            outputs=["pooled"],
            attrs={"axis": 0, "kernel_size": tokens, "stride": tokens, "padding": 0},
        ),
        Node(OpType.ARGMAX, "op_argmax", inputs=["pooled"], outputs=["label"]),
    ]
    weights = {
        "w_emb": [[(row + col) % 5 - 2 for col in range(dim)] for row in range(VOCAB)],
        "w_mm": [[1] * dim for _ in range(labels)],
    }
    return Graph(inputs=["tokens"], outputs=["label"], nodes=nodes, tensors=tensors), weights


def build_with_limit(tokens: int = 4, dim: int = 4) -> tuple[Graph, dict]:
    """Same shape, but rejects windows wider than 16: one illegal grid point."""
    if tokens > 16:
        raise ValueError(f"window {tokens} is not supported by this toy frontend")
    return build(tokens=tokens, dim=dim)
'''

#: Minimal nn.Module class (the loader's module-export path).
NN_FIXTURE_SOURCE = '''"""Tiny nn.Module fixture for the spec loader's module-export path."""

from __future__ import annotations

import torch.nn as nn


class TinyNet(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(nn.Embedding(256, 4), nn.Linear(4, 2))
'''


def _spec_dict(**overrides) -> dict:
    data = {
        "name": "toy-search",
        "version": 1,
        "task": "classification",
        "domain": "general",
        "input": {"name": "tokens", "shape": [4], "dtype": "uint8"},
        "output": {"name": "label", "classes": ["a", "b", "c"]},
        "model": {"entry_point": "toy_search_fixture.py:build"},
        "budget": {"ram_arena_max_bytes": 2048, "rom_total_max_bytes": 40960},
    }
    data.update(overrides)
    return data


@pytest.fixture
def search_project(temp_dir: Path) -> Path:
    """A spec project holding the parametric toy factory."""
    (temp_dir / "toy_search_fixture.py").write_text(TOY_SEARCH_MODEL_SOURCE, encoding="utf-8")
    return temp_dir


def _write_spec(project: Path, data: dict) -> Path:
    path = project / "kiln.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- the schema


def test_search_block_parses(search_project: Path) -> None:
    spec = load_spec_from_dict(
        _spec_dict(
            search={
                "params": {"tokens": [8, 4], "dim": [2]},
                "entry_point": "toy_search_fixture.py:build_with_limit",
                "max_candidates": 32,
            }
        ),
        base_dir=search_project,
    )

    assert spec.search is not None
    assert spec.search.params == {"tokens": (8, 4), "dim": (2,)}
    assert spec.search.entry_point == "toy_search_fixture.py:build_with_limit"
    assert spec.search.max_candidates == 32


def test_search_defaults_when_optional_keys_are_omitted(search_project: Path) -> None:
    spec = load_spec_from_dict(
        _spec_dict(search={"params": {"tokens": [4]}}), base_dir=search_project
    )

    assert spec.search is not None
    assert spec.search.entry_point is None
    assert spec.search.max_candidates == 2048


def test_specs_without_a_search_block_have_no_search(search_project: Path) -> None:
    assert load_spec_from_dict(_spec_dict(), base_dir=search_project).search is None
    # An empty block declares nothing to explore rather than an empty grid.
    assert load_spec_from_dict(_spec_dict(search={}), base_dir=search_project).search is None


@pytest.mark.parametrize(
    "search, match",
    [
        ({"params": {}}, "non-empty 'params'"),
        ({"params": {"tokens": 8}}, "list of integer values"),
        ({"params": {"tokens": []}}, "at least one value"),
        ({"params": {"tokens": [0]}}, "positive integers"),
        ({"params": {"tokens": [-4]}}, "positive integers"),
        ({"params": {"tokens": ["wide"]}}, "invalid literal"),
        ({"params": {"a-b": [1]}}, "identifier"),
        ({"params": {"tokens": [1, 2, 3]}, "max_candidates": 2}, "exceeding max_candidates"),
        ({"params": {"tokens": [1]}, "max_candidates": 0}, "must be positive"),
    ],
)
def test_invalid_search_blocks_raise(search_project: Path, search: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_spec_from_dict(_spec_dict(search=search), base_dir=search_project)


# ------------------------------------------------------------------ the grid


def test_enumerate_grid_is_sorted_and_cartesian() -> None:
    grid = enumerate_grid({"tokens": (64, 128), "dim": (4, 8)})

    assert grid == [
        {"dim": 4, "tokens": 64},
        {"dim": 4, "tokens": 128},
        {"dim": 8, "tokens": 64},
        {"dim": 8, "tokens": 128},
    ]
    # Same grid, same order: the report bytes are reproducible.
    assert grid == enumerate_grid({"dim": (4, 8), "tokens": (64, 128)})


def test_enumerate_grid_single_param() -> None:
    assert enumerate_grid({"tokens": (4,)}) == [{"tokens": 4}]


# ---------------------------------------------------------------- the front


def _row(outcome, **params):
    for row in outcome.rows:
        if all(row.params[name] == value for name, value in params.items()):
            return row
    raise AssertionError(f"no row for {params}")


def test_pareto_front_hand_computed() -> None:
    points = [
        {"ram": 10, "rom": 10, "macs": 1},  # dominated by point 2 (same RAM, less ROM)
        {"ram": 5, "rom": 10, "macs": 3},  # front: least RAM, most capacity
        {"ram": 10, "rom": 5, "macs": 1},  # front: least ROM
        {"ram": 12, "rom": 12, "macs": 0},  # dominated by point 2
    ]

    assert pareto_front(points, ("ram", "rom"), ("macs",)) == [1, 2]


def test_pareto_front_keeps_duplicate_points() -> None:
    """Two shapes can audit to the same footprint; neither dominates the other."""
    point = {"ram": 5, "rom": 5, "macs": 5}

    assert pareto_front([dict(point), dict(point)], ("ram",), ("macs",)) == [0, 1]


def test_pareto_front_respects_the_maximise_direction() -> None:
    assert dominates({"m": 5}, {"m": 3}, (), ("m",)) is True
    assert dominates({"m": 3}, {"m": 5}, (), ("m",)) is False
    assert dominates({"m": 3}, {"m": 3}, (), ("m",)) is False  # ties never dominate


def test_pareto_front_empty_and_missing_objective() -> None:
    assert pareto_front([], ("ram",), ("macs",)) == []
    with pytest.raises(KeyError, match="macs"):
        pareto_front([{"ram": 1}], ("ram",), ("macs",))


# ----------------------------------------------------------------- the engine


def test_run_search_requires_a_search_block(search_project: Path) -> None:
    spec = load_spec_from_dict(_spec_dict(), base_dir=search_project)

    with pytest.raises(ValueError, match="no 'search:' block"):
        run_search(spec)


def test_run_search_toy_grid_matches_the_arena_formula(search_project: Path) -> None:
    """arena = tokens * (dim + labels) -- the relation the reference spec documents."""
    spec = load_spec_from_dict(
        _spec_dict(search={"params": {"tokens": [4, 8], "dim": [2, 4]}}),
        base_dir=search_project,
    )

    outcome = run_search(spec)

    assert len(outcome.rows) == 4
    assert outcome.feasible_count == 4 and outcome.error_count == 0
    for row in outcome.rows:
        tokens, dim = row.params["tokens"], row.params["dim"]
        assert row.arena_ram_bytes == tokens * (dim + 3)
        assert row.total_rom_bytes == 256 * dim + 3 * dim
        assert row.total_macs == tokens * 3 * dim
        assert row.estimated_worst_cycles > 0
        assert row.capacity_per_arena_byte == pytest.approx(row.total_macs / row.arena_ram_bytes)
    assert outcome.pareto == {0, 1, 2, 3}


def test_run_search_reports_a_failing_candidate_instead_of_aborting(
    search_project: Path,
) -> None:
    spec = load_spec_from_dict(
        _spec_dict(
            model={"entry_point": "toy_search_fixture.py:build_with_limit"},
            search={"params": {"tokens": [8, 32], "dim": [4]}},
        ),
        base_dir=search_project,
    )

    outcome = run_search(spec)

    assert len(outcome.rows) == 2
    assert outcome.feasible_count == 1
    assert outcome.error_count == 1
    bad = _row(outcome, tokens=32, dim=4)
    assert bad.error is not None and "not supported" in bad.error
    assert bad.feasible is False
    assert bad.arena_ram_bytes == 0 and bad.capacity_per_arena_byte == 0.0
    assert _row(outcome, tokens=8, dim=4).feasible is True


# ------------------------------------------------------- the shipped reference


def test_build_bag_candidate_is_shape_exact_and_deterministic() -> None:
    """The search entry point: any shape, placeholder weights, same bytes twice."""
    graph, weights = build_bag_candidate(tokens=64, dim=4)

    assert graph.tensors["tokens"].shape == (64,)
    assert graph.tensors["emb"].shape == (64, 4)
    assert len(weights["w_emb"]) == VOCAB_SIZE and len(weights["w_emb"][0]) == 4
    assert len(weights["w_mm"]) == len(HAZARD_CLASSES)
    assert build_bag_candidate(tokens=64, dim=4)[1] == weights


def test_hazard_reference_grid_is_audited_exactly() -> None:
    """Every candidate's footprint follows the documented arena/ROM/MAC relations."""
    spec = load_spec(HAZARD_SEARCH_SPEC)
    assert spec.search is not None, "the shipped search spec must declare a grid"
    # The override is what keeps the shipped trained factory out of the search.
    assert spec.model.entry_point == "examples.hazard.graph:build_hazard_model"
    assert spec.search.entry_point == "examples.hazard.graph:build_bag_candidate"

    outcome = run_search(spec)

    labels = len(HAZARD_CLASSES)
    assert len(outcome.rows) == 15
    assert outcome.error_count == 0
    assert outcome.feasible_count == 10
    for row in outcome.rows:
        tokens, dim = row.params["tokens"], row.params["dim"]
        assert row.arena_ram_bytes == tokens * (dim + labels)
        assert row.total_rom_bytes == VOCAB_SIZE * dim + labels * dim
        assert row.total_macs == tokens * labels * dim
        assert row.total_rom_bytes <= spec.budget.rom_total_max_bytes
        if row.feasible:
            assert row.violations == ()
            assert row.arena_ram_bytes <= spec.budget.ram_arena_max_bytes
        else:
            assert any("arena" in violation for violation in row.violations)

    # This grid is a *tight* RAM/capacity ladder: widening the window or the
    # embedding always buys MACs for bytes, so no feasible shape is wasteful and
    # nothing is dominated.  That is the finding, not an accident of the front.
    assert outcome.pareto == {i for i, row in enumerate(outcome.rows) if row.feasible}
    assert {row.params["tokens"] for row in outcome.rows if not row.feasible} == {192, 256}


def test_capacity_classes_pick_the_ram_cheapest_shape() -> None:
    """At equal capacity the widest embedding is the RAM-cheapest shape."""
    outcome = run_search(load_spec(HAZARD_SEARCH_SPEC))

    classes = capacity_classes(outcome)

    assert [row.total_macs for row in classes] == sorted(row.total_macs for row in classes)
    assert len({row.total_macs for row in classes}) == len(classes)
    # 4,608 MACs is reachable three ways; dim=8 / tokens=96 costs 1,344 B
    # against 1,920 B for dim=4 / tokens=192 -- 576 B for the same capacity.
    best_4608 = next(row for row in classes if row.total_macs == 4608)
    assert (best_4608.params["dim"], best_4608.params["tokens"]) == (8, 96)
    assert best_4608.arena_ram_bytes == 1344
    assert _row(outcome, dim=4, tokens=192).arena_ram_bytes == 1920
    # The efficiency metric is labels * dim / (dim + labels).
    assert best_4608.capacity_per_arena_byte == pytest.approx(48 / 14)
    assert _row(outcome, dim=4, tokens=64).capacity_per_arena_byte == pytest.approx(2.4)
    assert _row(outcome, dim=8, tokens=64).capacity_per_arena_byte == pytest.approx(48 / 14)


def test_capacity_classes_are_empty_when_nothing_fits(search_project: Path) -> None:
    spec = load_spec_from_dict(
        _spec_dict(
            budget={"ram_arena_max_bytes": 4, "rom_total_max_bytes": 40960},
            search={"params": {"tokens": [8], "dim": [4]}},
        ),
        base_dir=search_project,
    )

    outcome = run_search(spec)

    assert outcome.feasible_count == 0 and outcome.pareto == frozenset()
    assert capacity_classes(outcome) == ()


# --------------------------------------------------------------- the artifact


def test_search_report_renders_every_section() -> None:
    spec = load_spec(HAZARD_SEARCH_SPEC)

    report = format_search_report(spec, run_search(spec))

    assert report.startswith("# Kiln architecture search: hazard-sniffer-search")
    assert "`dim` = {4, 6, 8}" in report
    assert "`tokens` = {64, 96, 128, 192, 256}" in report
    assert "15" in report and "10 feasible, 0 errored" in report
    assert "## Pareto front" in report and "## Capacity classes" in report
    # A feasible front row, byte for byte.
    assert "| 8 | 96 | 1,344 | 2,096 | 4,608 | 24,618 | 3.43 | fits (**Pareto**) |" in report
    # An over-budget row names the violation instead of hiding it.
    assert (
        "| 8 | 192 | 2,688 | 2,096 | 9,216 | 49,194 | 3.43 | over budget: arena 2,688 > 2,048 |"
        in report
    )
    assert (
        "Objectives: minimise `arena_ram_bytes`, minimise `total_rom_bytes`, "
        "maximise `total_macs`." in report
    )


def test_search_results_csv_is_machine_readable() -> None:
    spec = load_spec(HAZARD_SEARCH_SPEC)

    text = search_results_csv(run_search(spec), ["dim", "tokens"])
    rows = list(csv.DictReader(io.StringIO(text)))

    assert len(rows) == 15
    assert list(rows[0]) == [
        "dim",
        "tokens",
        "arena_ram_bytes",
        "total_rom_bytes",
        "total_macs",
        "estimated_worst_cycles",
        "capacity_per_arena_byte",
        "feasible",
        "pareto",
        "error",
    ]
    assert (rows[0]["dim"], rows[0]["tokens"]) == ("4", "64")
    assert rows[0]["arena_ram_bytes"] == "640"
    assert rows[0]["capacity_per_arena_byte"] == "2.4000"
    assert rows[0]["feasible"] == "1" and rows[0]["pareto"] == "1" and rows[0]["error"] == ""
    assert sum(1 for row in rows if row["feasible"] == "0") == 5


def test_search_is_deterministic() -> None:
    """Same grid, same bytes -- Kiln's determinism rule applies to the search too."""
    spec = load_spec(HAZARD_SEARCH_SPEC)

    first, second = run_search(spec), run_search(spec)

    assert first.rows == second.rows
    assert first.pareto == second.pareto and first.feasible_count == second.feasible_count
    assert format_search_report(spec, first) == format_search_report(spec, second)
    assert search_results_csv(first, ["dim", "tokens"]) == search_results_csv(
        second, ["dim", "tokens"]
    )


# --------------------------------------------------------- loader overrides


def test_loader_forwards_param_overrides(search_project: Path) -> None:
    spec = load_spec_from_dict(_spec_dict(), base_dir=search_project)

    loaded = resolve_model_graph(spec, param_overrides={"tokens": 8, "dim": 2})

    assert loaded.graph.tensors["tokens"].shape == (8,)
    assert loaded.graph.tensors["emb"].shape == (8, 2)
    assert len(loaded.weights["w_emb"][0]) == 2
    assert loaded.weights["w_mm"] == [[1, 1], [1, 1], [1, 1]]


def test_loader_rejects_an_unknown_override(search_project: Path) -> None:
    spec = load_spec_from_dict(_spec_dict(), base_dir=search_project)

    with pytest.raises(ValueError, match="does not accept parameter 'bogus'"):
        resolve_model_graph(spec, param_overrides={"bogus": 1})


def test_loader_rejects_overrides_for_an_nn_module(temp_dir: Path) -> None:
    """Only factories can be shape-parameterized; an nn.Module has one shape."""
    (temp_dir / "nn_fixture.py").write_text(NN_FIXTURE_SOURCE, encoding="utf-8")
    data = _spec_dict(model={"entry_point": "nn_fixture.py:TinyNet"})
    spec = load_spec_from_dict(data, base_dir=temp_dir)

    with pytest.raises(ValueError, match="callable factory"):
        resolve_model_graph(spec, param_overrides={"dim": 4})


# --------------------------------------------------------------------- the CLI


def test_cli_search_on_the_hazard_reference(
    temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = temp_dir / "search"

    ret = main(["search", str(HAZARD_SEARCH_SPEC), "-o", str(out_dir)])

    assert ret == 0
    assert (out_dir / "search_report.md").is_file()
    assert (out_dir / "search_results.csv").is_file()
    captured = capsys.readouterr()
    assert "[SEARCH] hazard-sniffer-search: 15 candidates" in captured.out
    assert "PARETO" in captured.out and "OVER arena 2,560 > 2,048" in captured.out
    assert "[SEARCH] 10 of 15 candidates fit the budget; 10 on the Pareto front" in captured.out


def test_cli_search_defaults_to_a_search_directory_next_to_the_spec(
    search_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec_path = _write_spec(
        search_project,
        _spec_dict(search={"params": {"tokens": [8], "dim": [4]}}),
    )

    ret = main(["search", str(spec_path)])

    assert ret == 0
    assert (search_project / "search" / "search_report.md").is_file()
    assert "[SEARCH] 1 of 1 candidates fit the budget" in capsys.readouterr().out


def test_cli_search_writes_the_report_even_when_nothing_fits(
    search_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec_path = _write_spec(
        search_project,
        _spec_dict(
            budget={"ram_arena_max_bytes": 4, "rom_total_max_bytes": 40960},
            search={"params": {"tokens": [8], "dim": [4]}},
        ),
    )

    ret = main(["search", str(spec_path)])

    assert ret == 1
    assert "[NO CANDIDATE FITS]" in capsys.readouterr().err
    report = (search_project / "search" / "search_report.md").read_text(encoding="utf-8")
    assert "No candidate fits the declared budget." in report


def test_cli_search_without_a_search_block_errors(
    search_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec_path = _write_spec(search_project, _spec_dict())

    ret = main(["search", str(spec_path)])

    assert ret == 1
    captured = capsys.readouterr()
    assert "[ERROR] Search failed" in captured.err
    assert "no 'search:' block" in captured.err


def test_cli_search_reports_a_failing_candidate_as_an_error_row(
    search_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec_path = _write_spec(
        search_project,
        _spec_dict(
            model={"entry_point": "toy_search_fixture.py:build_with_limit"},
            search={"params": {"tokens": [8, 32], "dim": [4]}},
        ),
    )

    ret = main(["search", str(spec_path)])

    assert ret == 0
    captured = capsys.readouterr()
    assert "ERROR ValueError: window 32 is not supported" in captured.out
    assert "[SEARCH] 1 of 2 candidates fit the budget" in captured.out


def test_search_rejects_int32_overflow_candidate(search_project: Path) -> None:
    """Verifies that an architecture candidate whose int32 accumulator overflows is marked unfeasible."""
    fixture = search_project / "overflow_fixture.py"
    fixture.write_text(
        """
from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc
from kiln.frontend.pytorch import compute_multiplier_and_shift

def build_overflow(tokens: int = 4, dim: int = 4) -> tuple[Graph, dict]:
    mult, shift = compute_multiplier_and_shift(0.5)
    qp = QuantParam(multiplier=mult, shift=shift)
    tensors = {
        "tokens": TensorDesc("tokens", shape=(1, dim), dtype=DType.INT8),
        "y": TensorDesc("y", shape=(1, 1), dtype=DType.INT8, quant=qp),
        "label": TensorDesc("label", shape=(), dtype=DType.INT8),
    }
    nodes = [
        Node(OpType.MATMUL, "op_mm", inputs=["tokens"], outputs=["y"], attrs={"weight": "w_mm"}),
        Node(OpType.ARGMAX, "op_argmax", inputs=["y"], outputs=["label"]),
    ]
    weights = {"w_mm": [[127] * dim]}
    return Graph(inputs=["tokens"], outputs=["label"], nodes=nodes, tensors=tensors), weights
""",
        encoding="utf-8",
    )

    spec_data = _spec_dict(
        budget={"ram_arena_max_bytes": 100_000_000, "rom_total_max_bytes": 100_000_000},
        model={"entry_point": f"{fixture.name}:build_overflow"},
        search={
            "entry_point": f"{fixture.name}:build_overflow",
            "params": {"dim": [10, 200_000]},  # 200,000 * 127 * 127 overflows int32
        },
    )
    spec_path = _write_spec(search_project, spec_data)
    spec = load_spec(spec_path)
    outcome = run_search(spec)

    assert outcome.feasible_count == 1
    safe_row, overflow_row = outcome.rows
    assert safe_row.feasible is True
    assert overflow_row.feasible is False
    assert any("int32 accumulator overflow" in v for v in overflow_row.violations)
