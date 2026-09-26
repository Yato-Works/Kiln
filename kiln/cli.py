from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Tuple

from .backend.c99.emit_c import emit_c, write_c
from .budget.report import BudgetReport, audit_graph, format_budget_certificate
from .frontend.pytorch import export_to_kir
from .ir.kir import DType, Graph, OpType
from .ref.ref import run_ref


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
        graph, weights = export_to_kir(model, dummy_input)

        out_path = Path(args.output) if args.output else Path("model.c")
        # `--runtime-header None` (or an empty string) means "emit a
        # self-contained translation unit", mirroring emit_c's
        # ``runtime_header=None`` contract instead of including a header
        # literally named "None".
        runtime_header = args.runtime_header
        if runtime_header is not None and runtime_header.strip().lower() in ("", "none"):
            runtime_header = None
        write_c(out_path, graph, weights, alignment=args.alignment, runtime_header=runtime_header)

        report = audit_graph(graph, weights, alignment=args.alignment)
        cert = format_budget_certificate(report)

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


def cmd_audit(args: argparse.Namespace) -> int:
    try:
        model, dummy_input = _load_model_and_input(args.model_path)
        graph, weights = export_to_kir(model, dummy_input)
        report = audit_graph(graph, weights, alignment=args.alignment)

        if args.json:
            payload = {
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
            print(json.dumps(payload, indent=2))
        else:
            cert = format_budget_certificate(report)
            print(_color(cert, Style.CYAN))
        return 0
    except Exception as e:
        print(_color(f"[ERROR] Audit failed: {e}", Style.BOLD + Style.RED), file=sys.stderr)
        return 1


def _generate_test_samples(input_len: int, num_samples: int, is_int8: bool = False) -> list[list[int]]:
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
        print(_color(f"[ERROR] GCC compiler not found at: {gcc_path}", Style.BOLD + Style.RED), file=sys.stderr)
        return 1

    try:
        import random
        model, dummy_input = _load_model_and_input(args.model_path)
        graph, weights = export_to_kir(model, dummy_input)

        c_code = emit_c(graph, weights, runtime_header=None)

        input_len = graph.tensors[graph.inputs[0]].num_elements
        harness = c_code + f"""
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
        tmpdir = tempfile.mkdtemp()
        src_file = Path(tmpdir) / "verify_model.c"
        exe_file = Path(tmpdir) / "verify_model.exe"
        try:
            src_file.write_text(harness, encoding="utf-8")

            build_cmd = [str(gcc_path), "-O3", "-Wall", "-Wextra", "-Werror", "-pedantic", str(src_file), "-o", str(exe_file)]
            build_res = subprocess.run(build_cmd, capture_output=True, text=True)
            if build_res.returncode != 0:
                print(_color("[ERROR] GCC build failed:\n" + build_res.stderr, Style.BOLD + Style.RED), file=sys.stderr)
                return 1

            num_samples = max(1, args.samples)
            print(_color(f"[VERIFY] Running {num_samples} differential test vectors against GCC binary...", Style.BOLD + Style.CYAN))

            input_name = graph.inputs[0]
            output_name = graph.outputs[0]

            is_int8 = graph.tensors[input_name].dtype == DType.INT8
            samples = _generate_test_samples(input_len, num_samples, is_int8=is_int8)

            input_str = f"{len(samples)}\n" + "\n".join(" ".join(str(x) for x in s) for s in samples) + "\n"
            run_res = subprocess.run([str(exe_file)], input=input_str, capture_output=True, text=True, check=True)
            c_outputs = [int(line) for line in run_res.stdout.strip().split()]

            for i, sample in enumerate(samples):
                c_val = c_outputs[i]
                ref_out = run_ref(graph, {input_name: sample}, weights)[output_name][0]
                if c_val != ref_out:
                    print(
                        _color(f"[MISMATCH at sample #{i+1}] Input: {sample} -> C: {c_val} != Ref: {ref_out}", Style.BOLD + Style.RED),
                        file=sys.stderr,
                    )
                    return 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        print(_color(f"[PASSED] All {num_samples} differential samples matched bit-for-bit (100% bit-exact).", Style.BOLD + Style.GREEN))
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
        ("CONV1D", "Convolution", "1D spatial convolution with stride, padding, and requantization"),
        ("MATMUL", "Linear / GEMM", "Quantized matrix multiplication with Q31 scale and shift"),
        ("ADD", "Elementwise", "Elementwise tensor addition with saturating requantization"),
        ("LUT", "Non-linear", "256-entry static lookup table for arbitrary activations (ReLU, GELU, etc.)"),
        ("MAXPOOL1D", "Pooling", "1D windowed max-pooling over the last axis (axis=0 pools the leading axis)"),
        ("ARGMAX", "Reduction", "Finds index of maximum value; serves as deterministic graph exit"),
    ]
    for name, cat, desc in op_infos:
        print(f"{_color(name, Style.BOLD + Style.YELLOW):<21} | {_color(cat, Style.BLUE):<23} | {desc}")
    print()
    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kiln",
        description="Kiln: Embedded Deep Learning Compiler (Zero-malloc, bit-exact C99 generator)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # compile
    p_compile = subparsers.add_parser("compile", help="Compile a model to C99 code")
    p_compile.add_argument("model_path", help="Path to PyTorch model .py file")
    p_compile.add_argument("-o", "--output", default="model.c", help="Output C source path (default: model.c)")
    p_compile.add_argument("--entry-point", default=None, help="Entry point class or function name")
    p_compile.add_argument("--alignment", type=int, default=4, help="Arena memory alignment in bytes (default: 4)")
    p_compile.add_argument("--runtime-header", default="kiln_runtime.h", help="Runtime header to include (or None)")

    # audit
    p_audit = subparsers.add_parser("audit", help="Perform static footprint & latency audit")
    p_audit.add_argument("model_path", help="Path to PyTorch model .py file")
    p_audit.add_argument("--alignment", type=int, default=4, help="Arena memory alignment in bytes (default: 4)")
    p_audit.add_argument("--json", action="store_true", help="Output report as structured JSON")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify model via GCC build & differential testing")
    p_verify.add_argument("model_path", help="Path to PyTorch model .py file")
    p_verify.add_argument("--samples", type=int, default=100, help="Number of random differential samples (default: 100)")
    p_verify.add_argument("--gcc", default=None, help="Custom path to GCC compiler executable")

    # ops
    subparsers.add_parser("ops", help="List supported Kiln IR operations")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "compile":
        return cmd_compile(args)
    elif args.command == "audit":
        return cmd_audit(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "ops":
        return cmd_ops(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())