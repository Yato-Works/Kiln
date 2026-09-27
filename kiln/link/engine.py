"""``kiln link``: compile, link, measure and run a spec's C artifacts (Phase 4).

The builder proves what a *translation unit* will cost; this proves what a
*program* does.  One call:

1. resolves the spec and emits ``model.c`` + certificate + the declared frontend
   (exactly what ``kiln build`` writes, through the same helper),
2. generates the replay harness (``kiln/link/harness.py``),
3. compiles every translation unit with the strict flags, so a warning that the
   example suites would have caught at test time fails here instead,
4. links a real executable,
5. measures the artifact: section sizes from the linked image and undefined
   symbols of the model object -- the last of which is where the zero-malloc
   guarantee becomes an observation rather than a claim,
6. optionally runs the harness and returns the labels it printed, so a test can
   compare a *linked binary* against ``kiln.ref`` bit for bit.

The compiler decides nothing about the model: a host gcc and a cross
``arm-none-eabi-gcc`` take the same path, and the report states which one ran.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..budget.report import BudgetReport
from ..spec import KilnSpec, resolve_model_graph
from ..spec.pipeline import emit_spec_artifacts
from . import harness as harness_module
from .toolchain import (
    MALLOC_SYMBOLS,
    Toolchain,
    compile_object,
    detect_toolchain,
    inspect_sections,
    link_executable,
    run_executable,
    undefined_symbols,
)

__all__ = ("LinkOutcome", "link_spec")


@dataclass(frozen=True, slots=True)
class LinkOutcome:
    """Everything one ``kiln link`` run measured about a spec build."""
    spec_name: str
    out_dir: Path
    executable: Path
    harness_source: Path
    sources: tuple[Path, ...]
    objects: tuple[Path, ...]
    report: BudgetReport
    sections: Mapping[str, int]
    model_undefined: frozenset[str]
    vectors: tuple[tuple[int, ...], ...]
    predictions: tuple[int, ...]
    toolchain: str
    symbols_inspected: bool = True

    @property
    def bss_bytes(self) -> int | None:
        return self.sections.get(".bss")

    @property
    def rodata_bytes(self) -> int | None:
        """Read-only data, whichever name the object format uses."""
        if not self.sections:
            return None
        return self.sections.get(".rodata", 0) + self.sections.get(".rdata", 0)

    @property
    def text_bytes(self) -> int | None:
        return self.sections.get(".text")

    @property
    def malloc_symbols(self) -> frozenset[str]:
        """Dynamic-allocation entry points the model object references (empty is good)."""
        return frozenset(name for name in self.model_undefined if name in MALLOC_SYMBOLS)

    @property
    def arena_observed(self) -> int | None:
        """Arena bytes actually seen in ``.bss`` (``None`` without binutils)."""
        return self.bss_bytes

    @property
    def arena_matches(self) -> bool | None:
        """True when the linked image really reserves the audited arena."""
        bss = self.bss_bytes
        if bss is None:
            return None
        return bss >= self.report.arena_ram_bytes

    @property
    def rom_covers_weights(self) -> bool | None:
        """True when read-only data is at least the audited weight bytes."""
        rodata = self.rodata_bytes
        if rodata is None:
            return None
        return rodata >= self.report.weights_rom_bytes


def _executable_path(out_dir: Path, spec_name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return out_dir / f"{spec_name}_link{suffix}"


def link_spec(
    spec: KilnSpec,
    *,
    out_dir: Path | None = None,
    cc: str | None = None,
    alignment: int = 4,
    runtime_header: str | None = None,
    run: bool = False,
) -> LinkOutcome:
    """Compiles, links and (optionally) executes one declarative spec.

    ``runtime_header`` defaults to ``None``: the self-contained translation unit
    is the one that can be linked without shipping a project's own shim.
    """
    loaded = resolve_model_graph(spec)
    target = Path(out_dir) if out_dir else spec.base_dir / "link"
    emitted = emit_spec_artifacts(
        spec, loaded, target, alignment=alignment, runtime_header=runtime_header
    )

    input_desc = loaded.graph.tensors[loaded.graph.inputs[0]]
    vectors = harness_module.harness_vectors(input_desc.num_elements)
    harness_source = harness_module.harness_source_path(target)
    harness_source.write_text(
        harness_module.render_harness(input_desc.num_elements, vectors), encoding="utf-8"
    )

    toolchain: Toolchain = detect_toolchain(cc)
    include_dirs = (target, spec.base_dir)
    sources = (emitted.model_c, *emitted.frontend_sources, harness_source)

    objects: list[Path] = []
    for index, source in enumerate(sources):
        obj = target / f"kiln_link_{index}_{source.stem}.o"
        compile_object(toolchain, source, obj, include_dirs=include_dirs)
        objects.append(obj)

    executable = _executable_path(target, spec.name)
    link_executable(toolchain, objects, executable)

    sections = inspect_sections(toolchain, executable)
    model_undefined = undefined_symbols(toolchain, objects[0])
    predictions: tuple[int, ...] = ()
    if run:
        stdout = run_executable(toolchain, executable)
        predictions = harness_module.parse_predictions(stdout, expected=len(vectors))

    return LinkOutcome(
        spec_name=spec.name,
        out_dir=target,
        executable=executable,
        harness_source=harness_source,
        sources=sources,
        objects=tuple(objects),
        report=emitted.report,
        sections=sections,
        model_undefined=frozenset(model_undefined or ()),
        vectors=vectors,
        predictions=predictions,
        toolchain=toolchain.cc,
        symbols_inspected=model_undefined is not None,
    )
