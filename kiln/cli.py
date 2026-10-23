from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import importlib.metadata
from pathlib import Path
from typing import Any

from .backend.c99.emit_c import emit_c, write_c
from .budget.report import audit_graph, format_budget_certificate
from .ir.kir import DType
from .ir.serialization import dumps_bundle, loads_bundle
from .ref.ref import run_ref
from .link import (
    ToolchainError,
    budget_violations,
    format_execution_certificate,
    format_link_report,
    link_spec,
)
from .search import SearchRow, run_search
from .search.report import format_search_report, search_results_csv
from .spec import load_spec, resolve_model_graph
from .spec.pipeline import SafetyViolationError, emit_spec_artifacts
from .link.toolchain import TARGET_PROFILES


def _export_model_to_kir(model: Any, dummy_input: Any) -> tuple[Any, dict[str, Any]]:
    """Import the PyTorch frontend only for commands that receive a model."""
    from .frontend.pytorch import export_to_kir

    return export_to_kir(model, dummy_input)


# ANSI Color formatting
class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def _color(text: str, color_code: str) -> str:
    # If stdout is not a tty or on Windows without ANSI support, colorama / OS handling
    if not sys.stdout.isatty():
        return text
    return f"{color_code}{text}{Style.RESET}"


def _load_model_and_input(model_path_str: str, entry_point: str | None = None) -> tuple[Any, Any]:
    path = Path(model_path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Support importing Python files
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # 1. Resolve model
    model = None
    if entry_point:
        if not hasattr(module, entry_point):
            raise AttributeError(f"Entry point {entry_point!r} not found in {path}")
        target = getattr(module, entry_point)
        model = target() if isinstance(target, type) else target
    else:
        # Search for model or Model or first nn.Module instance / class
        for candidate_name in ("model", "net", "Model", "Net"):
            if hasattr(module, candidate_name):
                target = getattr(module, candidate_name)
                model = target() if isinstance(target, type) else target
                break

        if model is None:
            import torch.nn as nn

            for val in vars(module).values():
                if isinstance(val, nn.Module):
                    model = val
                    break
                elif isinstance(val, type) and issubclass(val, nn.Module) and val is not nn.Module:
                    model = val()
                    break

    if model is None:
        raise ValueError(
            f"Could not locate a PyTorch model in {path}. "
            "Please define 'model = MyModel()' or pass '--entry-point <name>'."
        )

    # 2. Resolve dummy input
    dummy_input = None
    for candidate_name in ("dummy_input", "sample_input", "input_tensor"):
        if hasattr(module, candidate_name):
            dummy_input = getattr(module, candidate_name)
            break

    if dummy_input is None:
        import torch

        # Default to a 2-element uint8 token sequence if not specified
        dummy_input = torch.tensor([1, 2], dtype=torch.uint8)

    return model, dummy_input


def cmd_compile(args: argparse.Namespace) -> int:
    try:
        model, dummy_input = _load_model_and_input(args.model_path, args.entry_point)
        graph, weights = _export_model_to_kir(model, dummy_input)

        # The safety gate runs before anything is written: an int32 accumulator
        # overflow is a miscompile, not a budgeting preference, so no C is
        # emitted at all. A graph that cannot be interval-audited also fails.
        report = audit_graph(graph, weights, alignment=args.alignment)
        cert = format_budget_certificate(report)

        if not report.int32_overflow_free:
            print(_color(cert, Style.CYAN))
            if report.interval_audit is not None:
                msg = (
                    f"[SAFETY VIOLATION] int32 accumulator overflow detected in {args.model_path}; "
                    f"compilation aborted, no C emitted"
                )
            else:
                msg = (
                    f"[SAFETY VIOLATION] interval boundary propagation could not run on {args.model_path}; "
                    f"refusing to emit C without accumulator safety proof"
                )
            print(_color(msg, Style.BOLD + Style.RED), file=sys.stderr)
            return 1

        out_path = Path(args.output) if args.output else Path("model.c")
        # `--runtime-header None` (or an empty string) means "emit a
        # self-contained translation unit", mirroring emit_c's
        # ``runtime_header=None`` contract instead of including a header
        # literally named "None".
        runtime_header = args.runtime_header
        if runtime_header is not None and runtime_header.strip().lower() in ("", "none"):
            runtime_header = None
        write_c(out_path, graph, weights, alignment=args.alignment, runtime_header=runtime_header)

        print(_color(cert, Style.CYAN))
        print(
            _color("[SUCCESS]", Style.BOLD + Style.GREEN)
            + f" Compiled {args.model_path} -> {out_path} "
            + _color(
                f"(Arena RAM: {report.arena_ram_bytes:,} B, Total ROM: {report.total_rom_bytes:,} B, MACs: {report.total_macs:,})",
                Style.BOLD + Style.YELLOW,
            )
        )
        return 0
    except Exception as e:
        print(_color(f"[ERROR] Compilation failed: {e}", Style.BOLD + Style.RED), file=sys.stderr)
        return 1


def cmd_export_kir(args: argparse.Namespace) -> int:
    """Export a PyTorch model into a portable KIR graph + weights bundle."""
    try:
        model, dummy_input = _load_model_and_input(args.model_path, args.entry_point)
        graph, weights = _export_model_to_kir(model, dummy_input)
        output = Path(args.output)
        output.write_text(dumps_bundle(graph, weights) + "\n", encoding="utf-8")
        print(f"[SUCCESS] Exported {args.model_path} -> {output}")
        print(
            f"           tensors={len(graph.tensors)}, nodes={len(graph.nodes)}, "
            f"weights={len(weights)}"
        )
        return 0
    except Exception as exc:
        print(_color(f"[ERROR] KIR export failed: {exc}", Style.BOLD + Style.RED), file=sys.stderr)
        return 1


def cmd_compile_kir(args: argparse.Namespace) -> int:
    """Compile a KIR bundle without importing PyTorch."""
    try:
        bundle_path = Path(args.bundle)
        graph, weights = loads_bundle(bundle_path.read_text(encoding="utf-8"))
        report = audit_graph(graph, weights, alignment=args.alignment)
        if not report.int32_overflow_free:
            print(_color(format_budget_certificate(report), Style.CYAN))
            print(
                _color(
                    "[SAFETY VIOLATION] KIR bundle refused; no C emitted",
                    Style.BOLD + Style.RED,
                ),
                file=sys.stderr,
            )
            return 1
        runtime_header = args.runtime_header
        if runtime_header is not None and runtime_header.strip().lower() in ("", "none"):
            runtime_header = None
        output = Path(args.output)
        write_c(output, graph, weights, alignment=args.alignment, runtime_header=runtime_header)
        print(_color(format_budget_certificate(report), Style.CYAN))
        print(
            _color("[SUCCESS]", Style.BOLD + Style.GREEN) + f" Compiled {bundle_path} -> {output}"
        )
        return 0
    except Exception as exc:
        print(
            _color(f"[ERROR] KIR compilation failed: {exc}", Style.BOLD + Style.RED),
            file=sys.stderr,
        )
        return 1


def _load_kir_bundle(path: str) -> tuple[Any, dict[str, Any]]:
    bundle_path = Path(path)
    return loads_bundle(bundle_path.read_text(encoding="utf-8"))


def cmd_validate_kir(args: argparse.Namespace) -> int:
    """Validate a KIR bundle and fail closed when safety is not established."""
    try:
        graph, weights = _load_kir_bundle(args.bundle)
        report = audit_graph(graph, weights, alignment=args.alignment)
        if not report.int32_overflow_free:
            print(_color(format_budget_certificate(report), Style.CYAN))
            print(
                _color(
                    "[INVALID] KIR safety certificate is unavailable or failed",
                    Style.BOLD + Style.RED,
                ),
                file=sys.stderr,
            )
            return 1
        print(_color("[VALID]", Style.BOLD + Style.GREEN) + f" {args.bundle}")
        print(
            f"         nodes={len(graph.nodes)}, tensors={len(graph.tensors)}, "
            f"arena={report.arena_ram_bytes:,} B, rom={report.total_rom_bytes:,} B, "
            f"macs={report.total_macs:,}"
        )
        return 0
    except Exception as exc:
        print(
            _color(f"[INVALID] KIR validation failed: {exc}", Style.BOLD + Style.RED),
            file=sys.stderr,
        )
        return 1


def cmd_inspect_kir(args: argparse.Namespace) -> int:
    """Inspect a KIR bundle as a human-readable or machine-readable report."""
    try:
        graph, weights = _load_kir_bundle(args.bundle)
        report = audit_graph(graph, weights, alignment=args.alignment)
        interval = report.interval_audit
        payload: dict[str, Any] = {
            "schema": "kiln.kir.bundle",
            "version": 1,
            "nodes": [
                {
                    "name": node.name,
                    "op": node.op_type.value,
                    "inputs": list(node.inputs),
                    "outputs": list(node.outputs),
                }
                for node in graph.nodes
            ],
            "tensors": {
                name: {"shape": list(desc.shape), "dtype": desc.dtype.value}
                for name, desc in graph.tensors.items()
            },
            "weights": sorted(weights),
            "resources": {
                "arena_ram_bytes": report.arena_ram_bytes,
                "total_rom_bytes": report.total_rom_bytes,
                "total_macs": report.total_macs,
                "estimated_worst_cycles": report.estimated_worst_cycles,
            },
            "interval_analysis": {
                "available": interval is not None,
                "int32_overflow_free": report.int32_overflow_free,
                "min_accum": interval.min_accum if interval else None,
                "max_accum": interval.max_accum if interval else None,
            },
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"KIR bundle: {args.bundle}")
            print(f"  graph: {len(graph.nodes)} nodes, {len(graph.tensors)} tensors")
            print(f"  weights: {len(weights)} ({', '.join(sorted(weights)) or 'none'})")
            resources = payload["resources"]
            print(
                f"  resources: RAM {resources['arena_ram_bytes']:,} B, "
                f"ROM {resources['total_rom_bytes']:,} B, MACs {resources['total_macs']:,}"
            )
            print(f"  int32 safety: {'SAFE' if report.int32_overflow_free else 'NOT VERIFIED'}")
            print("  operations: " + ", ".join(node.op_type.value for node in graph.nodes))
        return 0
    except Exception as exc:
        print(
            _color(f"[ERROR] KIR inspection failed: {exc}", Style.BOLD + Style.RED), file=sys.stderr
        )
        return 1


def cmd_build(args: argparse.Namespace) -> int:
    """Declarative pipeline: kiln.yaml -> resolve model -> audit budget -> emit C99."""
    try:
        task = load_spec(args.spec)
        loaded = resolve_model_graph(task)
        graph = loaded.graph

        # Soft contract check: the declared input must match the resolved graph.
        input_desc = graph.tensors[graph.inputs[0]]
        if tuple(input_desc.shape) != tuple(task.input.shape):
            print(
                _color(
                    f"[WARN] spec input shape {task.input.shape} does not match "
                    f"graph input {tuple(input_desc.shape)}",
                    Style.BOLD + Style.YELLOW,
                ),
                file=sys.stderr,
            )

        out_dir = Path(args.output) if args.output else task.base_dir / "build"

        # `--runtime-header None` (or an empty string) means "emit a
        # self-contained translation unit", mirroring cmd_compile.
        runtime_header = args.runtime_header
        if runtime_header is not None and runtime_header.strip().lower() in ("", "none"):
            runtime_header = None

        try:
            emitted = emit_spec_artifacts(
                task, loaded, out_dir, alignment=args.alignment, runtime_header=runtime_header
            )
        except SafetyViolationError as exc:
            # The gate fires before anything is written (see kiln.spec.pipeline),
            # so there is no model.c to flash by accident.  Print the certificate
            # that explains the refusal and fail the build.
            print(_color(format_budget_certificate(exc.report), Style.CYAN))
            print(
                _color(
                    f"[SAFETY VIOLATION] {task.name}: {exc}; build aborted, no artifacts written",
                    Style.BOLD + Style.RED,
                ),
                file=sys.stderr,
            )
            return 1

        report = emitted.report
        out_c = emitted.model_c
        cert = emitted.certificate

        # A declared frontend file that the tree does not have is a warning here
        # (the build products are still useful); `kiln link` decides its own
        # policy, because it has to compile whatever it is given.
        for missing in emitted.missing_frontend:
            print(
                _color(f"[WARN] frontend file not found: {missing}", Style.BOLD + Style.YELLOW),
                file=sys.stderr,
            )

        print(_color(cert, Style.CYAN))

        checks = [
            ("arena RAM", report.arena_ram_bytes, task.budget.ram_arena_max_bytes),
            ("total ROM", report.total_rom_bytes, task.budget.rom_total_max_bytes),
        ]
        if task.budget.macs_max is not None:
            checks.append(("MACs", report.total_macs, task.budget.macs_max))

        violations: list[str] = []
        for label, used, limit in checks:
            if used <= limit:
                status = _color("OK", Style.GREEN)
            else:
                status = _color("EXCEEDED", Style.RED)
                violations.append(f"{label} {used:,} > {limit:,}")
            print(f"[BUDGET] {label:<10} {used:,} <= {limit:,} ... {status}")

        if violations:
            print(
                _color(
                    f"[BUDGET EXCEEDED] {task.name}: " + "; ".join(violations),
                    Style.BOLD + Style.RED,
                ),
                file=sys.stderr,
            )
            return 1

        print(
            _color("[SUCCESS]", Style.BOLD + Style.GREEN)
            + f" Built {task.name} -> {out_c} "
            + _color(
                f"(Arena RAM: {report.arena_ram_bytes:,} B, Total ROM: {report.total_rom_bytes:,} B, MACs: {report.total_macs:,})",
                Style.BOLD + Style.YELLOW,
            )
        )
        return 0
    except Exception as e:
        print(_color(f"[ERROR] Build failed: {e}", Style.BOLD + Style.RED), file=sys.stderr)
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Declarative architecture search: enumerate the declared grid, gate every
    candidate on the spec's budget and report the RAM/ROM/capacity Pareto front."""
    try:
        task = load_spec(args.spec)
        outcome = run_search(task, alignment=args.alignment)
    except Exception as e:
        print(_color(f"[ERROR] Search failed: {e}", Style.BOLD + Style.RED), file=sys.stderr)
        return 1

    # run_search has already rejected a spec without a search block.
    assert task.search is not None
    names = sorted(task.search.params)

    def _label(row: SearchRow) -> str:
        return " ".join(f"{name}={row.params[name]}" for name in names)

    print(_color(f"[SEARCH] {task.name}: {len(outcome.rows)} candidates", Style.BOLD + Style.CYAN))
    print(
        _color(
            f"{'candidate':<24} {'arena RAM':>10} {'ROM':>8} {'MACs':>8} {'cycles':>9}  verdict",
            Style.BOLD + Style.CYAN,
        )
    )
    for index, row in enumerate(outcome.rows):
        if row.error is not None:
            verdict = _color("ERROR", Style.RED) + _color(f" {row.error}", Style.GRAY)
            print(f"{_label(row):<24} {'-':>10} {'-':>8} {'-':>8} {'-':>9}  {verdict}")
            continue
        if row.feasible:
            verdict = _color("PARETO" if index in outcome.pareto else "fits", Style.GREEN)
        else:
            verdict = _color("OVER", Style.RED) + _color(
                " " + "; ".join(row.violations), Style.GRAY
            )
        print(
            f"{_label(row):<24} {row.arena_ram_bytes:>10,} {row.total_rom_bytes:>8,} "
            f"{row.total_macs:>8,} {row.estimated_worst_cycles:>9,}  {verdict}"
        )

    out_dir = Path(args.output) if args.output else task.base_dir / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = out_dir / "search_report.md"
    report_csv = out_dir / "search_results.csv"
    report_md.write_text(format_search_report(task, outcome), encoding="utf-8")
    report_csv.write_text(search_results_csv(outcome, names), encoding="utf-8")

    print()
    print(
        _color(
            f"[SEARCH] {outcome.feasible_count} of {len(outcome.rows)} candidates fit the budget; "
            f"{len(outcome.pareto)} on the Pareto front -> {report_md}",
            Style.BOLD + Style.YELLOW,
        )
    )
    if outcome.feasible_count == 0:
        print(
            _color(
                f"[NO CANDIDATE FITS] {task.name}: the declared grid never meets the budget",
                Style.BOLD + Style.RED,
            ),
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_link(args: argparse.Namespace) -> int:
    """Phase 4: compile + link the spec's C artifacts, measure the image, replay it."""
    try:
        if args.runner_config and not args.run:
            raise ToolchainError("--runner-config requires --run")
        task = load_spec(args.spec)
        runtime_header = args.runtime_header
        if runtime_header is not None and runtime_header.strip().lower() in ("", "none"):
            runtime_header = None
        target_profile = getattr(args, "target", "host")
        outcome = link_spec(
            task,
            out_dir=Path(args.output) if args.output else None,
            cc=args.cc,
            target_profile=target_profile,
            alignment=args.alignment,
            runtime_header=runtime_header,
            run=args.run,
            runner_config=Path(args.runner_config) if args.runner_config else None,
        )
    except ToolchainError as e:
        print(_color(f"[ERROR] Link failed: {e}", Style.BOLD + Style.RED), file=sys.stderr)
        return 1
    except SafetyViolationError as e:
        # Nothing was emitted, so there is nothing to link; say why explicitly.
        print(
            _color(f"[SAFETY VIOLATION] Link refused: {e}", Style.BOLD + Style.RED),
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(_color(f"[ERROR] Link failed: {e}", Style.BOLD + Style.RED), file=sys.stderr)
        return 1

    report_text = format_link_report(outcome, task)
    (outcome.out_dir / "link_report.txt").write_text(report_text + "\n", encoding="utf-8")
    (outcome.out_dir / "execution_certificate.txt").write_text(
        format_execution_certificate(outcome) + "\n", encoding="utf-8"
    )
    print(report_text)
    print(
        _color("[LINKED]", Style.BOLD + Style.GREEN)
        + f" {task.name} -> {outcome.executable} "
        + _color(f"(toolchain: {outcome.toolchain})", Style.BOLD + Style.YELLOW)
    )

    violations = budget_violations(outcome, task)
    if violations:
        print(
            _color(
                f"[BUDGET EXCEEDED] {task.name}: " + "; ".join(violations),
                Style.BOLD + Style.RED,
            ),
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    try:
        model_str = args.model_path
        p = Path(model_str)
        if p.suffix.lower() in (".yaml", ".yml"):
            spec = load_spec(p)
            loaded = resolve_model_graph(spec)
            graph, weights = loaded.graph, loaded.weights
        else:
            model, dummy_input = _load_model_and_input(model_str)
            graph, weights = _export_model_to_kir(model, dummy_input)

        report = audit_graph(graph, weights, alignment=args.alignment)

        if args.json:
            payload: dict[str, Any] = {
                "arena_ram_bytes": report.arena_ram_bytes,
                "peak_concurrent_bytes": report.peak_concurrent_bytes,
                "weights_rom_bytes": report.weights_rom_bytes,
                "lut_rom_bytes": report.lut_rom_bytes,
                "total_rom_bytes": report.total_rom_bytes,
                "total_macs": report.total_macs,
                "total_adds": report.total_adds,
                "total_ops": report.total_ops,
                "estimated_worst_cycles": report.estimated_worst_cycles,
                "dynamic_allocation_bytes": 0,
                "deterministic": True,
            }
            if report.interval_audit is not None:
                ia = report.interval_audit
                payload["interval_analysis"] = {
                    "int32_overflow_free": ia.int32_overflow_free,
                    "total_accum_ops": ia.total_accum_ops,
                    "saturating_ops": ia.saturating_ops,
                    "min_accum": ia.min_accum,
                    "max_accum": ia.max_accum,
                    "tensor_intervals": {
                        t_name: {"lo": inv.lo, "hi": inv.hi}
                        for t_name, inv in ia.tensor_intervals.items()
                    },
                    "node_audits": {
                        n_name: {
                            "op_type": na.op_type.value,
                            "accum_lo": na.accum_lo,
                            "accum_hi": na.accum_hi,
                            "accum_fits_int32": na.accum_fits_int32,
                            "preclamp_lo": na.preclamp_lo,
                            "preclamp_hi": na.preclamp_hi,
                            "saturates": na.saturates,
                        }
                        for n_name, na in ia.node_audits.items()
                    },
                }
            print(json.dumps(payload, indent=2))
        else:
            cert = format_budget_certificate(report)
            print(_color(cert, Style.CYAN))
            if getattr(args, "intervals", False) and report.interval_audit is not None:
                ia = report.interval_audit
                print(
                    "\n"
                    + _color("--- Static Interval Boundary Analysis ---", Style.BOLD + Style.GREEN)
                )
                print(
                    f"Accumulator overflow safety : {'SAFE (int32)' if ia.int32_overflow_free else 'OVERFLOW DETECTED'}"
                )
                print(f"Accumulator value range    : [{ia.min_accum:+d}, {ia.max_accum:+d}]")
                print(f"Saturating requantize ops  : {ia.saturating_ops} / {ia.total_accum_ops}")
                print("\nTensor Value Bounds:")
                for t_name, inv in ia.tensor_intervals.items():
                    print(f"  {t_name:<20} : [{inv.lo:>5}, {inv.hi:>5}]")
                print("\nAccumulator / Requantize Details:")
                for n_name, na in ia.node_audits.items():
                    if na.accum_lo is not None:
                        sat_flag = " (SATURATES)" if na.saturates else ""
                        print(
                            f"  {n_name:<16} ({na.op_type.value:<8}) : "
                            f"acc=[{na.accum_lo:>10}, {na.accum_hi:>10}], "
                            f"preclamp=[{na.preclamp_lo:>6}, {na.preclamp_hi:>6}]{sat_flag}"
                        )
        return 0
    except Exception as e:
        print(_color(f"[ERROR] Audit failed: {e}", Style.BOLD + Style.RED), file=sys.stderr)
        return 1


def _generate_test_samples(
    input_len: int, num_samples: int, is_int8: bool = False
) -> list[list[int]]:
    """
    Generates test vectors combining critical edge patterns (zeros, max, boundaries, alternating,
    sparse) and deterministic pseudo-random vectors.
    """
    samples: list[list[int]] = []
    min_v, max_v = (-128, 127) if is_int8 else (0, 255)
    alt1, alt2 = (-85, 42) if is_int8 else (0x55, 0xAA)

    # 1. Uniform edge patterns
    samples.append([0] * input_len)
    samples.append([max_v] * input_len)
    samples.append([min_v] * input_len)
    samples.append([-1 if is_int8 else 128] * input_len)

    # 2. Alternating patterns
    samples.append([(alt1 if i % 2 == 0 else alt2) for i in range(input_len)])
    samples.append([(alt2 if i % 2 == 0 else alt1) for i in range(input_len)])
    samples.append([(min_v if i % 2 == 0 else max_v) for i in range(input_len)])
    samples.append([(max_v if i % 2 == 0 else min_v) for i in range(input_len)])

    # 3. Sparse / impulse patterns
    for pos in range(min(input_len, 4)):
        s_high = [0] * input_len
        s_high[pos] = max_v
        samples.append(s_high)

        s_low = [max_v] * input_len
        s_low[pos] = min_v
        samples.append(s_low)

    # 4. Fill remaining budget with pseudo-random numbers
    import random

    rng = random.Random(1337)
    while len(samples) < num_samples:
        samples.append([rng.randint(min_v, max_v) for _ in range(input_len)])

    return samples[:num_samples]


def cmd_verify(args: argparse.Namespace) -> int:
    gcc_path = args.gcc or shutil.which("gcc")
    if not gcc_path or not Path(gcc_path).exists():
        print(
            _color(f"[ERROR] GCC compiler not found at: {gcc_path}", Style.BOLD + Style.RED),
            file=sys.stderr,
        )
        return 1

    try:
        model, dummy_input = _load_model_and_input(args.model_path)
        graph, weights = _export_model_to_kir(model, dummy_input)

        c_code = emit_c(graph, weights, runtime_header=None)

        input_len = graph.tensors[graph.inputs[0]].num_elements
        harness = (
            c_code
            + f"""
#include <stdio.h>
#include <stdint.h>

int main(void) {{
    int n;
    if (scanf("%d", &n) != 1) return 1;
    for (int iter = 0; iter < n; ++iter) {{
        int8_t input[{input_len}];
        for (int i = 0; i < {input_len}; ++i) {{
            int val;
            if (scanf("%d", &val) != 1) return 1;
            input[i] = (int8_t)val;
        }}
        int8_t res = kiln_predict((const uint8_t *)input, {input_len});
        printf("%d\\n", (int)res);
    }}
    return 0;
}}
"""
        )
        tmpdir = tempfile.mkdtemp()
        src_file = Path(tmpdir) / "verify_model.c"
        exe_file = Path(tmpdir) / "verify_model.exe"
        try:
            src_file.write_text(harness, encoding="utf-8")

            build_cmd = [
                str(gcc_path),
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                str(src_file),
                "-o",
                str(exe_file),
            ]
            build_res = subprocess.run(build_cmd, capture_output=True, text=True)
            if build_res.returncode != 0:
                print(
                    _color(
                        "[ERROR] GCC build failed:\n" + build_res.stderr, Style.BOLD + Style.RED
                    ),
                    file=sys.stderr,
                )
                return 1

            num_samples = max(1, args.samples)
            print(
                _color(
                    f"[VERIFY] Running {num_samples} differential test vectors against GCC binary...",
                    Style.BOLD + Style.CYAN,
                )
            )

            input_name = graph.inputs[0]
            output_name = graph.outputs[0]

            is_int8 = graph.tensors[input_name].dtype == DType.INT8
            samples = _generate_test_samples(input_len, num_samples, is_int8=is_int8)

            input_str = (
                f"{len(samples)}\n" + "\n".join(" ".join(str(x) for x in s) for s in samples) + "\n"
            )
            run_res = subprocess.run(
                [str(exe_file)], input=input_str, capture_output=True, text=True, check=True
            )
            c_outputs = [int(line) for line in run_res.stdout.strip().split()]

            for i, sample in enumerate(samples):
                c_val = c_outputs[i]
                ref_out = run_ref(graph, {input_name: sample}, weights)[output_name][0]
                if c_val != ref_out:
                    print(
                        _color(
                            f"[MISMATCH at sample #{i + 1}] Input: {sample} -> C: {c_val} != Ref: {ref_out}",
                            Style.BOLD + Style.RED,
                        ),
                        file=sys.stderr,
                    )
                    return 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        print(
            _color(
                f"[PASSED] All {num_samples} differential samples matched bit-for-bit (100% bit-exact).",
                Style.BOLD + Style.GREEN,
            )
        )
        return 0

    except Exception as e:
        print(_color(f"[ERROR] Verification failed: {e}", Style.BOLD + Style.RED), file=sys.stderr)
        return 1


def cmd_ops(_args: argparse.Namespace) -> int:
    header = f"{'Op Type':<12} | {'Category':<14} | {'Description'}"
    divider = "-" * len(header)
    print(_color("\n=== Kiln Supported IR Operations ===", Style.BOLD + Style.CYAN))
    print(header)
    print(divider)
    op_infos = [
        ("EMBEDDING", "Lookup", "Transforms uint8 token indices into dense int8 feature vectors"),
        (
            "CONV1D",
            "Convolution",
            "1D spatial convolution with stride, padding, and requantization",
        ),
        ("MATMUL", "Linear / GEMM", "Quantized matrix multiplication with Q31 scale and shift"),
        ("ADD", "Elementwise", "Elementwise tensor addition with saturating requantization"),
        (
            "LUT",
            "Non-linear",
            "256-entry static lookup table for arbitrary activations (ReLU, GELU, etc.)",
        ),
        (
            "MAXPOOL1D",
            "Pooling",
            "1D windowed max-pooling over the last axis (axis=0 pools the leading axis)",
        ),
        ("ARGMAX", "Reduction", "Finds index of maximum value; serves as deterministic graph exit"),
    ]
    for name, cat, desc in op_infos:
        print(
            f"{_color(name, Style.BOLD + Style.YELLOW):<21} | {_color(cat, Style.BLUE):<23} | {desc}"
        )
    print()
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Report the local tools needed for a reproducible Kiln workflow."""
    checks: list[tuple[str, bool, str]] = []

    def add(label: str, ok: bool, detail: str) -> None:
        checks.append((label, ok, detail))
        status = "OK" if ok else "MISSING"
        print(f"[{status:<7}] {label:<18} {detail}")

    add("Python", sys.version_info >= (3, 10), sys.version.split()[0])
    for import_name, distribution in (("torch", "torch"), ("yaml", "PyYAML"), ("pytest", "pytest")):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            add(import_name, False, "install the test environment")
        else:
            add(import_name, True, version)

    for name in ("gcc", "clang", "ruff", "mypy"):
        path = shutil.which(name)
        add(name, path is not None, path or "not on PATH")

    for target, profile in TARGET_PROFILES.items():
        if target == "host":
            continue
        compiler = shutil.which(profile.default_cc)
        runner = shutil.which(profile.runner) if profile.runner else None
        add(f"{target} compiler", compiler is not None, compiler or profile.default_cc)
        if profile.runner is None:
            add(
                f"{target} runner",
                False,
                f"not configured ({profile.execution})",
            )
        else:
            add(f"{target} runner", runner is not None, runner or profile.runner)

    required = [ok for label, ok, _ in checks if label in {"Python", "torch", "yaml", "gcc"}]
    if all(required):
        print("[READY  ] Kiln core workflow is available")
        return 0
    print("[BLOCKED] Install the missing core requirements before building models", file=sys.stderr)
    return 1


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kiln",
        description="Kiln: Embedded Deep Learning Compiler (Zero-malloc, bit-exact C99 generator)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # compile
    p_compile = subparsers.add_parser("compile", help="Compile a model to C99 code")
    p_compile.add_argument("model_path", help="Path to PyTorch model .py file")
    p_compile.add_argument(
        "-o", "--output", default="model.c", help="Output C source path (default: model.c)"
    )
    p_compile.add_argument("--entry-point", default=None, help="Entry point class or function name")
    p_compile.add_argument(
        "--alignment", type=int, default=4, help="Arena memory alignment in bytes (default: 4)"
    )
    p_compile.add_argument(
        "--runtime-header", default="kiln_runtime.h", help="Runtime header to include (or None)"
    )

    # export-kir / compile-kir
    p_export_kir = subparsers.add_parser(
        "export-kir", help="Export a PyTorch model into a portable KIR bundle"
    )
    p_export_kir.add_argument("model_path", help="Path to PyTorch model .py file")
    p_export_kir.add_argument("-o", "--output", required=True, help="Output KIR bundle JSON path")
    p_export_kir.add_argument(
        "--entry-point", default=None, help="Entry point class or function name"
    )

    p_compile_kir = subparsers.add_parser(
        "compile-kir", help="Compile a KIR bundle without importing PyTorch"
    )
    p_compile_kir.add_argument("bundle", help="Path to a KIR bundle JSON file")
    p_compile_kir.add_argument("-o", "--output", default="model.c", help="Output C source path")
    p_compile_kir.add_argument(
        "--alignment", type=int, default=4, help="Arena memory alignment in bytes (default: 4)"
    )
    p_compile_kir.add_argument(
        "--runtime-header", default=None, help="Runtime header to include (or None)"
    )

    p_validate_kir = subparsers.add_parser(
        "validate", help="Validate a KIR bundle and its static safety certificate"
    )
    p_validate_kir.add_argument("bundle", help="Path to a KIR bundle JSON file")
    p_validate_kir.add_argument(
        "--alignment", type=int, default=4, help="Arena memory alignment in bytes (default: 4)"
    )

    p_inspect_kir = subparsers.add_parser(
        "inspect", help="Inspect a KIR bundle and its resource certificate"
    )
    p_inspect_kir.add_argument("bundle", help="Path to a KIR bundle JSON file")
    p_inspect_kir.add_argument(
        "--alignment", type=int, default=4, help="Arena memory alignment in bytes (default: 4)"
    )
    p_inspect_kir.add_argument("--json", action="store_true", help="Output structured JSON")

    # build (declarative kiln.yaml pipeline)
    p_build = subparsers.add_parser("build", help="Build a declarative kiln.yaml task spec to C99")
    p_build.add_argument("spec", help="Path to the kiln.yaml task specification")
    p_build.add_argument(
        "-o", "--output", default=None, help="Output directory (default: <spec dir>/build)"
    )
    p_build.add_argument(
        "--alignment", type=int, default=4, help="Arena memory alignment in bytes (default: 4)"
    )
    p_build.add_argument(
        "--runtime-header", default="kiln_runtime.h", help="Runtime header to include (or None)"
    )

    # search (compile-time architecture exploration over a `search:` grid)
    p_search = subparsers.add_parser(
        "search",
        help="Enumerate a kiln.yaml search grid and report the RAM/ROM/capacity Pareto front (supported: hazard example)",
    )
    p_search.add_argument(
        "spec", help="Path to the kiln.yaml task specification (needs a search: block)"
    )
    p_search.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory for the report (default: <spec dir>/search)",
    )
    p_search.add_argument(
        "--alignment", type=int, default=4, help="Arena memory alignment in bytes (default: 4)"
    )

    # link (Phase 4: compile + link + measure + replay the emitted C)
    p_link = subparsers.add_parser(
        "link",
        help="Compile and link a kiln.yaml build under strict flags, measure the image and replay it",
    )
    p_link.add_argument("spec", help="Path to the kiln.yaml task specification")
    p_link.add_argument(
        "-o", "--output", default=None, help="Output directory (default: <spec dir>/link)"
    )
    p_link.add_argument(
        "--cc",
        default=None,
        help="C compiler to use (default: $KILN_CC, then gcc/cc/clang on PATH)",
    )
    p_link.add_argument(
        "--target",
        default="host",
        choices=sorted(TARGET_PROFILES),
        help="Target architectural profile for host or cross compilation (default: host)",
    )
    p_link.add_argument(
        "--alignment", type=int, default=4, help="Arena memory alignment in bytes (default: 4)"
    )
    p_link.add_argument(
        "--runtime-header",
        default=None,
        help="Runtime header to include (default: None = self-contained translation unit)",
    )
    p_link.add_argument(
        "--run", action="store_true", help="Execute the linked binary on the generated vectors"
    )
    p_link.add_argument(
        "--runner-config",
        default=None,
        help="JSON flash/run command configuration for a physical hardware target",
    )

    # audit
    p_audit = subparsers.add_parser("audit", help="Perform static footprint & latency audit")
    p_audit.add_argument("model_path", help="Path to PyTorch model .py file")
    p_audit.add_argument(
        "--alignment", type=int, default=4, help="Arena memory alignment in bytes (default: 4)"
    )
    p_audit.add_argument("--json", action="store_true", help="Output report as structured JSON")
    p_audit.add_argument(
        "--intervals",
        action="store_true",
        help="Display static interval boundary propagation analysis",
    )

    # verify
    p_verify = subparsers.add_parser(
        "verify", help="Verify model via GCC build & differential testing"
    )
    p_verify.add_argument("model_path", help="Path to PyTorch model .py file")
    p_verify.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of random differential samples (default: 100)",
    )
    p_verify.add_argument("--gcc", default=None, help="Custom path to GCC compiler executable")

    # ops
    subparsers.add_parser("ops", help="List supported Kiln IR operations")

    # doctor
    subparsers.add_parser("doctor", help="Check local tools for a reproducible Kiln workflow")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "compile":
        return cmd_compile(args)
    elif args.command == "export-kir":
        return cmd_export_kir(args)
    elif args.command == "compile-kir":
        return cmd_compile_kir(args)
    elif args.command == "validate":
        return cmd_validate_kir(args)
    elif args.command == "inspect":
        return cmd_inspect_kir(args)
    elif args.command == "build":
        return cmd_build(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "link":
        return cmd_link(args)
    elif args.command == "audit":
        return cmd_audit(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "ops":
        return cmd_ops(args)
    elif args.command == "doctor":
        return cmd_doctor(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
