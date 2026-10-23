"""Loader and resolver for KilnSpec models and graphs.

Resolves Python entry points, loads checkpoints, and constructs (Graph, weights) tuples.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from ..frontend.pytorch import export_to_kir
from ..ir.kir import Graph
from .schema import KilnSpec

__all__ = ("LoadedModel", "resolve_model_graph")


class LoadedModel:
    """Encapsulates the resolved computation graph and int8 weights."""

    def __init__(self, graph: Graph, weights: Mapping[str, Any], spec: KilnSpec):
        self.graph = graph
        self.weights = weights
        self.spec = spec


def _import_entry_point(entry_point: str, base_dir: Path) -> Any:
    """
    Imports a callable, class, or object from `module:attribute` or `path/to/file.py:attribute`.
    """
    if ":" not in entry_point:
        raise ValueError(
            f"Invalid entry_point '{entry_point}'; expected format 'module:attr' or 'file.py:attr'"
        )

    module_str, attr_name = entry_point.rsplit(":", 1)
    file_candidate = base_dir / module_str
    if not file_candidate.is_file() and (base_dir / f"{module_str}.py").is_file():
        file_candidate = base_dir / f"{module_str}.py"

    # Imports resolve against the spec's directory and the invocation directory
    # (a `module:attr` entry like `examples.hazard.graph:build_hazard_model` only
    # works from the repo root when the installed CLI does not put cwd on sys.path).
    added_paths: list[str] = []
    for entry in (str(base_dir), os.getcwd()):
        if entry not in sys.path:
            sys.path.insert(0, entry)
            added_paths.append(entry)

    try:
        if file_candidate.is_file():
            # Load from file path directly
            spec = importlib.util.spec_from_file_location(
                f"kiln_dynamic_{file_candidate.stem}", file_candidate
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from path: {file_candidate}")
            module = importlib.util.module_from_spec(spec)
            # Register before executing so inspect/torch can resolve classes and
            # dataclasses defined in the module (mirrors kiln.cli._load_model_and_input).
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            # Standard Python module import
            module = importlib.import_module(module_str)
    finally:
        for entry in added_paths:
            if entry in sys.path:
                sys.path.remove(entry)

    if not hasattr(module, attr_name):
        raise AttributeError(f"Module '{module_str}' has no attribute '{attr_name}'")

    return getattr(module, attr_name)


def resolve_model_graph(
    spec: KilnSpec,
    *,
    entry_point: str | None = None,
    param_overrides: Mapping[str, Any] | None = None,
) -> LoadedModel:
    """
    Builds or extracts the (Graph, weights) pair according to the KilnSpec.

    Supports:
      1. Functions returning (Graph, weights) or a dataclass with `.graph` and `.weights`
      2. PyTorch nn.Module models (exported via `export_to_kir`)

    ``entry_point`` overrides ``spec.model.entry_point`` and ``param_overrides``
    are forwarded as keyword arguments to the resolved factory; ``kiln search``
    uses both to instantiate one candidate per grid point.  A parameter the
    factory does not accept is an error rather than a silently ignored value.
    """
    resolved_entry = entry_point or spec.model.entry_point
    target = _import_entry_point(resolved_entry, spec.base_dir)

    # Case 1: Target is a callable function/factory
    if callable(target) and not (isinstance(target, type) and issubclass(target, nn.Module)):
        sig = inspect.signature(target)
        kwargs: dict[str, Any] = {}
        if "weights_path" in sig.parameters and spec.model.weights:
            kwargs["weights_path"] = str(spec.resolve_path(spec.model.weights))
        elif "weights" in sig.parameters and spec.model.weights:
            kwargs["weights"] = str(spec.resolve_path(spec.model.weights))
        if "architecture" in sig.parameters and spec.model.architecture:
            kwargs["architecture"] = spec.model.architecture

        for name, value in (param_overrides or {}).items():
            if name not in sig.parameters:
                raise ValueError(
                    f"Entry point '{resolved_entry}' does not accept parameter '{name}'"
                )
            kwargs[name] = value

        res = target(**kwargs)

        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], Graph):
            return LoadedModel(res[0], res[1], spec)
        elif hasattr(res, "graph") and hasattr(res, "weights"):
            return LoadedModel(res.graph, res.weights, spec)
        elif isinstance(res, nn.Module):
            # Model returned from factory function
            return _export_module_to_loaded(res, spec)
        else:
            raise TypeError(
                f"Entry point function '{spec.model.entry_point}' returned unexpected type: {type(res)}"
            )

    # Case 2: Target is an nn.Module instance
    if isinstance(target, nn.Module):
        if param_overrides:
            raise ValueError(
                "Search parameters require a callable factory entry point; "
                f"'{resolved_entry}' is an nn.Module instance"
            )
        return _export_module_to_loaded(target, spec)

    # Case 3: Target is an nn.Module class
    if isinstance(target, type) and issubclass(target, nn.Module):
        if param_overrides:
            raise ValueError(
                "Search parameters require a callable factory entry point; "
                f"'{resolved_entry}' is an nn.Module class"
            )
        inst = target()
        return _export_module_to_loaded(inst, spec)

    raise TypeError(f"Target '{resolved_entry}' is neither a callable factory nor an nn.Module")


def _export_module_to_loaded(model: nn.Module, spec: KilnSpec) -> LoadedModel:
    """Loads weights if provided and exports the PyTorch module into KIR."""
    if spec.model.weights:
        w_path = spec.resolve_path(spec.model.weights)
        if w_path.exists():
            state = torch.load(w_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)

    # Construct dummy input from spec shape and dtype
    dtype = torch.uint8 if spec.input.dtype == "uint8" else torch.int8
    dummy = torch.zeros(spec.input.shape, dtype=dtype)
    graph, weights = export_to_kir(model, dummy)
    return LoadedModel(graph, weights, spec)
