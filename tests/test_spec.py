"""Phase 3 Step 1: declarative ``kiln.yaml`` spec parsing and model/graph resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from kiln.ir.kir import Graph
from kiln.spec import (
    BudgetSpec,
    load_spec,
    load_spec_from_dict,
    resolve_model_graph,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


#: A complete, valid spec mapping used as the base for mutation-based tests.
def _valid_spec() -> dict:
    return {
        "name": "toy",
        "version": 1,
        "task": "classification",
        "domain": "general",
        "input": {"name": "tokens", "shape": [4], "dtype": "uint8"},
        "output": {"name": "label", "classes": ["a", "b", "c"]},
        "model": {"entry_point": "toy_fixture.py:build"},
    }


#: Representative YAML document covering every spec section.
YAML_DOC = """\
name: toy
version: 1
task: classification
domain: text
input:
  name: tokens
  shape: [4]
  dtype: uint8
output:
  name: label
  classes: [a, b, c]
model:
  entry_point: toy_fixture.py:build
  weights: weights.json
budget:
  ram_arena_max_bytes: 1024
  rom_total_max_bytes: 8192
  macs_max: 5000
frontend:
  type: tokenizer
  source_files: [fe.c]
  header_files: [fe.h]
dataset:
  corpus: corpus.json
  manifest: manifest.json
  split_seed: 7
"""


def test_load_spec_from_dict_parses_all_sections(temp_dir: Path) -> None:
    import yaml

    spec = load_spec_from_dict(yaml.safe_load(YAML_DOC), base_dir=temp_dir)

    assert spec.name == "toy"
    assert spec.version == 1
    assert spec.task == "classification"
    assert spec.domain == "text"
    assert spec.input.name == "tokens"
    assert spec.input.shape == (4,)
    assert spec.input.dtype == "uint8"
    assert spec.output.name == "label"
    assert spec.output.classes == ("a", "b", "c")
    assert spec.model.entry_point == "toy_fixture.py:build"
    assert spec.model.weights == "weights.json"
    assert spec.budget.ram_arena_max_bytes == 1024
    assert spec.budget.rom_total_max_bytes == 8192
    assert spec.budget.macs_max == 5000
    assert spec.frontend is not None and spec.frontend.type == "tokenizer"
    assert spec.frontend.source_files == ("fe.c",)
    assert spec.frontend.header_files == ("fe.h",)
    assert spec.dataset is not None and spec.dataset.corpus == "corpus.json"
    assert spec.dataset.split_seed == 7
    assert spec.base_dir == temp_dir.resolve()


def test_budget_defaults_when_section_omitted() -> None:
    spec = load_spec_from_dict(_valid_spec())
    assert spec.budget == BudgetSpec(
        ram_arena_max_bytes=2048,
        rom_total_max_bytes=40960,
        macs_max=None,
        zero_malloc=True,
        deterministic=True,
    )


@pytest.mark.parametrize(
    "missing", ["name", "version", "task", "domain", "input", "output", "model"]
)
def test_missing_required_field_raises(missing: str) -> None:
    data = _valid_spec()
    del data[missing]
    with pytest.raises(ValueError, match=missing):
        load_spec_from_dict(data)


def test_invalid_input_dtype_raises() -> None:
    data = _valid_spec()
    data["input"] = {"name": "x", "shape": [4], "dtype": "float32"}
    with pytest.raises(ValueError, match="dtype"):
        load_spec_from_dict(data)


def test_input_requires_name_and_shape() -> None:
    data = _valid_spec()
    data["input"] = {"shape": [4]}
    with pytest.raises(ValueError, match="shape"):
        load_spec_from_dict(data)


def test_model_requires_entry_point() -> None:
    data = _valid_spec()
    data["model"] = {}
    with pytest.raises(ValueError, match="entry_point"):
        load_spec_from_dict(data)


def test_output_spec_defaults() -> None:
    data = _valid_spec()
    data["output"] = {}
    spec = load_spec_from_dict(data)
    assert spec.output.name == "label"
    assert spec.output.classes == ()


def test_load_spec_from_yaml_file_sets_base_dir(temp_dir: Path) -> None:
    path = temp_dir / "kiln.yaml"
    path.write_text(YAML_DOC, encoding="utf-8")

    spec = load_spec(path)

    assert spec.base_dir == temp_dir.resolve()
    assert spec.resolve_path("weights.json") == (temp_dir / "weights.json").resolve()
    absolute = temp_dir / "sub" / "w.json"
    assert spec.resolve_path(absolute) == absolute


def test_load_spec_from_yaml_string() -> None:
    """A multi-line YAML document must parse without probing the filesystem."""
    spec = load_spec(YAML_DOC)
    assert spec.name == "toy"
    assert spec.model.weights == "weights.json"
    assert spec.base_dir == Path(".").resolve()


def test_load_spec_missing_file_raises(temp_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_spec(temp_dir / "does_not_exist.yaml")


def test_load_spec_non_mapping_yaml_raises(temp_dir: Path) -> None:
    path = temp_dir / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_spec(path)


@pytest.mark.parametrize("relative", ["examples/hazard/kiln.yaml", "examples/kws/kiln.yaml"])
def test_reference_specs_parse(relative: str) -> None:
    """The shipped reference specs validate against the schema."""
    spec = load_spec(REPO_ROOT / relative)
    assert spec.version == 1
    assert spec.task == "classification"
    assert spec.domain in ("text", "audio", "timeseries", "general")
    assert spec.input.dtype in ("uint8", "int8")
    assert len(spec.output.classes) >= 2
    assert spec.budget.ram_arena_max_bytes == 2048
    assert spec.budget.rom_total_max_bytes == 40960
    assert spec.frontend is not None and spec.frontend.source_files
    assert spec.model.entry_point.count(":") == 1


# ---------------------------------------------------------------------------
# Loader / resolver (kiln.spec.loader)
# ---------------------------------------------------------------------------

#: Minimal nn.Module class used to exercise the loader's module-export path.
NN_FIXTURE_SOURCE = '''"""Tiny nn.Module fixture for the spec loader's module-export path."""

from __future__ import annotations

import torch.nn as nn


class TinyNet(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(nn.Embedding(256, 4), nn.Linear(4, 2))
'''


def _loader_spec(toy_project: Path, **model_overrides: str) -> dict:
    data = _valid_spec()
    data["model"] = {**data["model"], **model_overrides}
    return data


def test_resolve_factory_from_file_entry_point(toy_project: Path) -> None:
    spec = load_spec_from_dict(_loader_spec(toy_project), base_dir=toy_project)

    loaded = resolve_model_graph(spec)

    assert isinstance(loaded.graph, Graph)
    assert loaded.graph.inputs == ("tokens",)
    assert loaded.graph.outputs == ("label",)
    assert set(loaded.weights) == {"w_emb", "w_mm"}
    assert loaded.spec is spec


def test_resolve_factory_receives_resolved_weights_path(toy_project: Path) -> None:
    """`model.weights` is resolved against base_dir and passed as `weights_path`."""
    spec = load_spec_from_dict(
        _loader_spec(toy_project, weights="weights.json"), base_dir=toy_project
    )

    loaded = resolve_model_graph(spec)

    expected = json.loads((toy_project / "weights.json").read_text(encoding="utf-8"))
    assert loaded.weights == expected


def test_resolve_module_entry_point_hazard_reference() -> None:
    """`module:attr` form against the shipped hazard reference spec."""
    spec = load_spec(REPO_ROOT / "examples/hazard/kiln.yaml")

    loaded = resolve_model_graph(spec)

    assert loaded.graph.inputs == ("tokens",)
    assert {"w_emb", "w_mm"} <= set(loaded.weights)
    assert len(loaded.weights["w_mm"]) == len(spec.output.classes)


def test_resolve_invalid_entry_point_format(toy_project: Path) -> None:
    spec = load_spec_from_dict(
        _loader_spec(toy_project, entry_point="missing_colon"), base_dir=toy_project
    )
    with pytest.raises(ValueError, match="entry_point"):
        resolve_model_graph(spec)


def test_resolve_missing_attribute(toy_project: Path) -> None:
    spec = load_spec_from_dict(
        _loader_spec(toy_project, entry_point="toy_fixture.py:does_not_exist"),
        base_dir=toy_project,
    )
    with pytest.raises(AttributeError, match="does_not_exist"):
        resolve_model_graph(spec)


def test_resolve_nn_module_class_exports_graph(temp_dir: Path) -> None:
    (temp_dir / "nn_fixture.py").write_text(NN_FIXTURE_SOURCE, encoding="utf-8")
    data = _valid_spec()
    data["model"] = {"entry_point": "nn_fixture.py:TinyNet"}
    spec = load_spec_from_dict(data, base_dir=temp_dir)

    loaded = resolve_model_graph(spec)

    assert isinstance(loaded.graph, Graph)
    assert loaded.graph.inputs == ("graph_in",)
    assert loaded.weights  # quantised embedding/linear weights


def test_resolve_nn_module_loads_state_dict(temp_dir: Path) -> None:
    """`model.weights` state dict is loaded before export (strict keys)."""
    (temp_dir / "nn_fixture.py").write_text(NN_FIXTURE_SOURCE, encoding="utf-8")
    torch.save(
        {
            "0.weight": torch.rand(256, 4) - 0.5,
            "1.weight": torch.rand(2, 4) - 0.5,
            "1.bias": torch.rand(2) - 0.5,
        },
        temp_dir / "tiny.pt",
    )
    data = _valid_spec()
    data["model"] = {"entry_point": "nn_fixture.py:TinyNet", "weights": "tiny.pt"}
    spec = load_spec_from_dict(data, base_dir=temp_dir)

    loaded = resolve_model_graph(spec)

    assert isinstance(loaded.graph, Graph)
    assert loaded.graph.inputs == ("graph_in",)
