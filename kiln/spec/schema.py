"""Declarative task specification (kiln.yaml) schema and parser.

Phase 3 introduces declarative task specs allowing models, pipelines, and static
budgets to be fully specified and built via ``kiln build kiln.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = (
    "BudgetSpec",
    "DatasetSpec",
    "FrontendSpec",
    "InputSpec",
    "KilnSpec",
    "ModelSpec",
    "OutputSpec",
    "SearchSpec",
    "load_spec",
    "load_spec_from_dict",
)


@dataclass(frozen=True, slots=True)
class InputSpec:
    """Input tensor specification."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "uint8"  # "uint8" or "int8"


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """Output tensor specification."""

    name: str
    classes: tuple[str, ...] = ()
    dtype: str = "int8"


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Static resource envelope targets for MCU deployment."""

    ram_arena_max_bytes: int = 2048
    rom_total_max_bytes: int = 40960
    macs_max: int | None = None
    zero_malloc: bool = True
    deterministic: bool = True


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Model graph / weights resolution."""

    entry_point: str
    weights: str | None = None
    architecture: str | None = None


@dataclass(frozen=True, slots=True)
class FrontendSpec:
    """Optional C frontend preprocessor / tokenizer integration."""

    type: str  # e.g., "tokenizer", "audio", "custom", "none"
    source_files: tuple[str, ...] = ()
    header_files: tuple[str, ...] = ()
    entry_point: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Dataset / provenance metadata."""

    corpus: str | None = None
    manifest: str | None = None
    split_seed: int = 1337


@dataclass(frozen=True, slots=True)
class SearchSpec:
    """Compile-time architecture search grid (Phase 3).

    ``params`` maps a model-factory keyword argument to the values to try; the
    Cartesian product of those tuples is the candidate set of ``kiln search``.
    Footprints are exact at compile time, so a candidate needs no training run
    to be measured (see ``kiln.search.engine``).  ``entry_point`` optionally
    overrides ``model.entry_point`` for the search only, which is how a spec
    keeps its shipped (trained, single-shape) factory while searching a
    shape-parameterized one.
    """

    params: Mapping[str, tuple[int, ...]]
    entry_point: str | None = None
    max_candidates: int = 2048


@dataclass(frozen=True, slots=True)
class KilnSpec:
    """Complete declarative task specification (kiln.yaml)."""

    name: str
    version: int
    task: str  # e.g., "classification"
    domain: str  # e.g., "text", "audio", "timeseries", "general"
    input: InputSpec
    output: OutputSpec
    model: ModelSpec
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    frontend: FrontendSpec | None = None
    dataset: DatasetSpec | None = None
    search: SearchSpec | None = None
    base_dir: Path = field(default_factory=lambda: Path("."))

    def resolve_path(self, relative_or_absolute: str | Path) -> Path:
        """Resolves a path relative to the spec's base directory."""
        p = Path(relative_or_absolute)
        if p.is_absolute():
            return p
        return (self.base_dir / p).resolve()


def _parse_input_spec(d: Mapping[str, Any]) -> InputSpec:
    if "name" not in d or "shape" not in d:
        raise ValueError("input spec requires 'name' and 'shape'")
    shape = tuple(int(x) for x in d["shape"])
    dtype = str(d.get("dtype", "uint8")).lower()
    if dtype not in ("uint8", "int8"):
        raise ValueError(f"Unsupported input dtype '{dtype}'; must be 'uint8' or 'int8'")
    return InputSpec(name=str(d["name"]), shape=shape, dtype=dtype)


def _parse_output_spec(d: Mapping[str, Any]) -> OutputSpec:
    name = str(d.get("name", "label"))
    classes = tuple(str(c) for c in d.get("classes", []))
    dtype = str(d.get("dtype", "int8")).lower()
    return OutputSpec(name=name, classes=classes, dtype=dtype)


def _parse_budget_spec(d: Mapping[str, Any] | None) -> BudgetSpec:
    if not d:
        return BudgetSpec()
    return BudgetSpec(
        ram_arena_max_bytes=int(d.get("ram_arena_max_bytes", 2048)),
        rom_total_max_bytes=int(d.get("rom_total_max_bytes", 40960)),
        macs_max=int(d["macs_max"]) if "macs_max" in d and d["macs_max"] is not None else None,
        zero_malloc=bool(d.get("zero_malloc", True)),
        deterministic=bool(d.get("deterministic", True)),
    )


def _parse_model_spec(d: Mapping[str, Any]) -> ModelSpec:
    if "entry_point" not in d:
        raise ValueError("model spec requires 'entry_point'")
    return ModelSpec(
        entry_point=str(d["entry_point"]),
        weights=str(d["weights"]) if "weights" in d and d["weights"] is not None else None,
        architecture=str(d["architecture"])
        if "architecture" in d and d["architecture"] is not None
        else None,
    )


def _parse_frontend_spec(d: Mapping[str, Any] | None) -> FrontendSpec | None:
    if not d:
        return None
    ftype = str(d.get("type", "none"))
    sources = tuple(str(s) for s in d.get("source_files", []))
    raw_headers = d.get("header_files", [])
    headers = tuple(str(h) for h in raw_headers)
    entry = str(d["entry_point"]) if "entry_point" in d and d["entry_point"] is not None else None
    return FrontendSpec(
        type=ftype,
        source_files=sources,
        header_files=headers,
        entry_point=entry,
    )


def _parse_dataset_spec(d: Mapping[str, Any] | None) -> DatasetSpec | None:
    if not d:
        return None
    return DatasetSpec(
        corpus=str(d["corpus"]) if "corpus" in d and d["corpus"] is not None else None,
        manifest=str(d["manifest"]) if "manifest" in d and d["manifest"] is not None else None,
        split_seed=int(d.get("split_seed", 1337)),
    )


def _parse_search_spec(d: Mapping[str, Any] | None) -> SearchSpec | None:
    if not d:
        return None
    raw_params = d.get("params")
    if not isinstance(raw_params, Mapping) or not raw_params:
        raise ValueError("search spec requires a non-empty 'params' mapping")

    params: dict[str, tuple[int, ...]] = {}
    grid_size = 1
    for raw_name, raw_values in raw_params.items():
        name = str(raw_name)
        if not name.isidentifier():
            raise ValueError(
                f"search param {name!r} must be a Python identifier (it is passed as a factory kwarg)"
            )
        if isinstance(raw_values, (str, bytes)) or not hasattr(raw_values, "__iter__"):
            raise ValueError(f"search param {name!r} requires a list of integer values")
        values = tuple(int(v) for v in raw_values)
        if not values:
            raise ValueError(f"search param {name!r} requires at least one value")
        if any(v <= 0 for v in values):
            raise ValueError(f"search param {name!r} values must be positive integers")
        params[name] = values
        grid_size *= len(values)

    max_candidates = int(d.get("max_candidates", 2048))
    if max_candidates < 1:
        raise ValueError(f"search max_candidates must be positive, got {max_candidates}")
    if grid_size > max_candidates:
        raise ValueError(
            f"search grid has {grid_size} candidates, exceeding max_candidates={max_candidates}"
        )

    entry = str(d["entry_point"]) if "entry_point" in d and d["entry_point"] is not None else None
    return SearchSpec(params=params, entry_point=entry, max_candidates=max_candidates)


def load_spec_from_dict(d: Mapping[str, Any], *, base_dir: Path = Path(".")) -> KilnSpec:
    """Validates and constructs a KilnSpec from a dictionary."""
    for required in ("name", "version", "task", "domain", "input", "output", "model"):
        if required not in d:
            raise ValueError(f"Missing required field in kiln.yaml: '{required}'")

    input_spec = _parse_input_spec(d["input"])
    output_spec = _parse_output_spec(d["output"])
    budget_spec = _parse_budget_spec(d.get("budget"))
    model_spec = _parse_model_spec(d["model"])
    frontend_spec = _parse_frontend_spec(d.get("frontend"))
    dataset_spec = _parse_dataset_spec(d.get("dataset"))
    search_spec = _parse_search_spec(d.get("search"))

    return KilnSpec(
        name=str(d["name"]),
        version=int(d["version"]),
        task=str(d["task"]),
        domain=str(d["domain"]),
        input=input_spec,
        output=output_spec,
        model=model_spec,
        budget=budget_spec,
        frontend=frontend_spec,
        dataset=dataset_spec,
        search=search_spec,
        base_dir=base_dir.resolve(),
    )


def load_spec(path_or_str: str | Path) -> KilnSpec:
    """Loads and validates a KilnSpec from a YAML file path or YAML string."""
    text = str(path_or_str)

    # Only probe the filesystem for single-line inputs that could plausibly be a
    # path: Path(...) raises OSError/ValueError on overly long strings and on
    # drive-relative fragments, which a raw YAML document would hit.
    if "\n" not in text:
        try:
            is_file = Path(text).is_file()
        except (OSError, ValueError):
            is_file = False
        if is_file:
            source = Path(text)
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"Expected a YAML mapping in {source}, got {type(raw).__name__}")
            return load_spec_from_dict(raw, base_dir=source.parent)

    raw = yaml.safe_load(text)
    if isinstance(raw, dict):
        return load_spec_from_dict(raw, base_dir=Path("."))

    raise FileNotFoundError(f"Specification file not found: {text}")
