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
from ..ir.serialization import loads_bundle
from ..spec import KilnSpec, resolve_model_graph
from ..spec.pipeline import emit_spec_artifacts
from ..spec.loader import LoadedModel
from ..spec.schema import BudgetSpec, InputSpec, ModelSpec, OutputSpec
from . import harness as harness_module
from .toolchain import (
    MALLOC_SYMBOLS,
    Toolchain,
    ToolchainError,
    compile_object,
    detect_toolchain,
    inspect_sections,
    link_executable,
    run_executable,
    undefined_symbols,
)
from .target_runtime import CORTEX_M_LINKER_SCRIPT, CORTEX_M_SEMIHOST_ASM, CORTEX_M_STARTUP_C
from .runner import (
    RunnerConfig,
    RunnerError,
    RuntimeMetrics,
    load_runner_config,
    parse_runtime_metrics,
)

__all__ = ("LinkOutcome", "bundle_spec", "link_bundle", "link_spec")


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
    target_profile: str = "host"
    execution_mode: str = "native"
    execution_evidence: str = "not-run"
    runtime_metrics: RuntimeMetrics = RuntimeMetrics()
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


def bundle_spec(bundle: Path) -> KilnSpec:
    """Create the conservative declarative envelope for a KIR bundle.

    A bundle carries graph semantics but intentionally does not carry a product
    budget. Direct bundle linking therefore uses the same default envelope as a
    minimal ``kiln.yaml`` and labels the artifact as a bundle-derived build.
    """
    graph, _weights = loads_bundle(bundle.read_text(encoding="utf-8"))
    input_desc = graph.tensors[graph.inputs[0]]
    output_desc = graph.tensors[graph.outputs[0]]
    return KilnSpec(
        name=bundle.name.removesuffix(".kir.json"),
        version=1,
        task="bundle",
        domain="general",
        input=InputSpec(input_desc.name, tuple(input_desc.shape), input_desc.dtype.value),
        output=OutputSpec(output_desc.name, (), output_desc.dtype.value),
        model=ModelSpec(entry_point="bundle"),
        budget=BudgetSpec(),
        base_dir=bundle.parent.resolve(),
    )


def _link_loaded(
    spec: KilnSpec,
    loaded: LoadedModel,
    *,
    out_dir: Path | None = None,
    cc: str | None = None,
    target_profile: str = "host",
    alignment: int = 4,
    runtime_header: str | None = None,
    run: bool = False,
    runner_config: Path | None = None,
) -> LinkOutcome:
    """Compiles, links and (optionally) executes one resolved graph.

    ``runtime_header`` defaults to ``None``: the self-contained translation unit
    is the one that can be linked without shipping a project's own shim.
    """
    target = Path(out_dir) if out_dir else spec.base_dir / "link"
    emitted = emit_spec_artifacts(
        spec, loaded, target, alignment=alignment, runtime_header=runtime_header
    )

    toolchain: Toolchain = detect_toolchain(cc, target=target_profile)
    input_desc = loaded.graph.tensors[loaded.graph.inputs[0]]
    vectors = harness_module.harness_vectors(input_desc.num_elements)
    harness_source = harness_module.harness_source_path(target)
    harness_source.write_text(
        harness_module.render_harness(
            input_desc.num_elements,
            vectors,
            output=toolchain.profile.harness_output,
        ),
        encoding="utf-8",
    )

    include_dirs = (target, spec.base_dir)
    support_sources: tuple[Path, ...] = ()
    linker_script: Path | None = None
    if toolchain.profile.startup == "cortex-m":
        startup_source = target / "kiln_startup.c"
        semihost_source = target / "kiln_semihost.s"
        linker_script = target / "kiln_memory.ld"
        startup_source.write_text(CORTEX_M_STARTUP_C, encoding="utf-8")
        semihost_source.write_text(CORTEX_M_SEMIHOST_ASM, encoding="utf-8")
        linker_script.write_text(CORTEX_M_LINKER_SCRIPT, encoding="utf-8")
        support_sources = (startup_source, semihost_source)
    sources = (emitted.model_c, *emitted.frontend_sources, *support_sources, harness_source)

    objects: list[Path] = []
    for index, source in enumerate(sources):
        obj = target / f"kiln_link_{index}_{source.stem}.o"
        compile_object(toolchain, source, obj, include_dirs=include_dirs)
        objects.append(obj)

    executable = _executable_path(target, spec.name)
    link_executable(toolchain, objects, executable, linker_script=linker_script)

    sections = inspect_sections(toolchain, executable)
    model_undefined = undefined_symbols(toolchain, objects[0])
    predictions: tuple[int, ...] = ()
    execution_evidence = "not-run"
    runtime_metrics = RuntimeMetrics()
    if run:
        hardware_config: RunnerConfig | None = None
        if runner_config is not None:
            try:
                hardware_config = load_runner_config(runner_config)
            except RunnerError as exc:
                raise ToolchainError(str(exc)) from exc
        stdout = run_executable(toolchain, executable, runner_config=hardware_config)
        predictions = harness_module.parse_predictions(stdout, expected=len(vectors))
        runtime_metrics = parse_runtime_metrics(stdout)
        if hardware_config is not None:
            execution_evidence = f"hardware-runner:{hardware_config.name}"
        else:
            execution_evidence = toolchain.profile.execution

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
        target_profile=toolchain.profile.name,
        execution_mode=toolchain.profile.execution,
        execution_evidence=execution_evidence,
        runtime_metrics=runtime_metrics,
        symbols_inspected=model_undefined is not None,
    )


def link_spec(
    spec: KilnSpec,
    *,
    out_dir: Path | None = None,
    cc: str | None = None,
    target_profile: str = "host",
    alignment: int = 4,
    runtime_header: str | None = None,
    run: bool = False,
    runner_config: Path | None = None,
) -> LinkOutcome:
    """Compile, link and (optionally) execute one declarative spec."""
    return _link_loaded(
        spec,
        resolve_model_graph(spec),
        out_dir=out_dir,
        cc=cc,
        target_profile=target_profile,
        alignment=alignment,
        runtime_header=runtime_header,
        run=run,
        runner_config=runner_config,
    )


def link_bundle(
    bundle: Path,
    *,
    out_dir: Path | None = None,
    cc: str | None = None,
    target_profile: str = "host",
    alignment: int = 4,
    runtime_header: str | None = None,
    run: bool = False,
    runner_config: Path | None = None,
) -> LinkOutcome:
    """Compile, link and (optionally) execute a portable KIR v1 bundle."""
    graph, weights = loads_bundle(bundle.read_text(encoding="utf-8"))
    spec = bundle_spec(bundle)
    return _link_loaded(
        spec,
        LoadedModel(graph, weights, spec),
        out_dir=out_dir,
        cc=cc,
        target_profile=target_profile,
        alignment=alignment,
        runtime_header=runtime_header,
        run=run,
        runner_config=runner_config,
    )
