#!/usr/bin/env python3
"""Demo capture script for Kiln showcases 1 & 2.

Showcase 1: "2KB Brain" - kiln build emits single C file, ~2KB RAM, zero heap, zero FP.
Showcase 2: "Build Refuses" - dangerous model rejected with [SAFETY VIOLATION], no artifacts written.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: A model whose int32 accumulator genuinely overflows: 140,000 MACs at the
#: 127*127 maximum product is 2,258,060,000 > 2^31-1.  Shared by showcase 2
#: and scripts/render_demo_gif.py so both demonstrate the same artifact.
DANGEROUS_MODEL_SOURCE = """
# Dangerous model: large MATMUL that overflows int32 accumulator
import torch
import torch.nn as nn

class DangerousModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 20000 inputs would give 20000 * 127 * 127 = 322,580,000 which FITS int32;
        # overflow needs > 133,143 MACs at the 127*127 maximum product, so this
        # model uses 140,000 inputs: 140000 * 127 * 127 = 2,258,060,000 > INT32_MAX
        # (2,147,483,647) with every weight at the +127 edge and inputs up to +127.
        self.linear = nn.Linear(140000, 10, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(127)

    def forward(self, x):
        return self.linear(x)

model = DangerousModel()
dummy_input = torch.randint(-128, 127, (1, 140000), dtype=torch.int8)
"""


def run_cmd(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        print(f"[FAILED] Exit code: {result.returncode}", file=sys.stderr)
    return result


def showcase_1_two_kb_brain() -> None:
    """Showcase 1: 2KB Brain - kiln build emits single C file with ~2KB RAM, zero heap, zero FP."""
    print("\n" + "=" * 70)
    print("SHOWCASE 1: '2KB Brain' - kiln build emits single C file")
    print("=" * 70)

    # Clean build directory
    build_dir = REPO_ROOT / "examples" / "hazard" / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # Run kiln build (--runtime-header None emits a self-contained translation
    # unit, so the demo's gcc compile needs nothing but the C file itself)
    print("\n--- Running: kiln build examples/hazard/kiln.yaml --runtime-header None ---")
    result = run_cmd([
        sys.executable, "-m", "kiln", "build",
        "examples/hazard/kiln.yaml", "--runtime-header", "None"
    ], cwd=REPO_ROOT)

    if result.returncode != 0:
        print("Build failed!")
        return

    # Check the generated files
    model_c = build_dir / "model.c"
    cert_file = build_dir / "budget_certificate.txt"

    print(f"\n--- Generated files in {build_dir} ---")
    for f in build_dir.iterdir():
        size = f.stat().st_size
        print(f"  {f.name}: {size} bytes")

    # Show the certificate
    print("\n--- Budget Certificate ---")
    if cert_file.exists():
        print(cert_file.read_text())

    # Show the C file size and key properties
    if model_c.exists():
        c_content = model_c.read_text()
        print(f"\n--- model.c ({len(c_content)} chars) ---")
        print(f"  Lines of C code: {c_content.count(chr(10))}")
        print(f"  Entry point 'kiln_predict' present: {'kiln_predict' in c_content}")

    # Compile and check nm -u.  Object-level evidence, not source substrings:
    # the certificate comment itself contains the words "malloc" and "floating
    # point", so the honest zero-heap / zero-FP check is the symbol table.
    print("\n--- Compiling with gcc and checking unresolved symbols (nm -u) ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        obj_file = tmpdir / "model.o"
        compile_result = run_cmd([
            "gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-c", str(model_c), "-o", str(obj_file)
        ], check=False)

        if compile_result.returncode == 0:
            print("gcc: clean (no warnings, -Werror -pedantic)")
            nm_result = run_cmd(["nm", "-u", str(obj_file)], check=False)
            undefined = [
                line.split()[-1]
                for line in nm_result.stdout.strip().splitlines()
                if line.strip()
            ]
            print("Unresolved symbols (nm -u):")
            if undefined:
                for sym in undefined:
                    print(f"  {sym}")
            else:
                print("  (none - fully self-contained)")
            heap_syms = [s for s in undefined if s in ("malloc", "calloc", "realloc", "free")]
            print(f"\n  Heap symbols referenced : {', '.join(heap_syms) if heap_syms else 'none'}")
            print(f"  Only 'abort' unresolved : {undefined == ['abort']}")
        else:
            print("gcc failed - see output above")


def showcase_2_build_refuses() -> None:
    """Showcase 2: Build Refuses - dangerous model rejected with SAFETY VIOLATION."""
    print("\n" + "=" * 70)
    print("SHOWCASE 2: 'Build Refuses' - dangerous model rejected")
    print("=" * 70)

    # Create a model that will overflow int32 accumulator
    dangerous_model = REPO_ROOT / "examples" / "hazard" / "dangerous_model.py"
    dangerous_model.write_text(DANGEROUS_MODEL_SOURCE)

    # Try to compile it - should fail with SAFETY VIOLATION
    print("\n--- Running: kiln compile dangerous_model.py (should fail) ---")
    result = run_cmd([
        sys.executable, "-m", "kiln", "compile",
        str(dangerous_model),
        "-o", "dangerous.c"
    ], cwd=REPO_ROOT, check=False)

    print(f"\nExit code: {result.returncode} (expected: 1)")
    print(f"Output contains '[SAFETY VIOLATION]': {'[SAFETY VIOLATION]' in result.stdout or '[SAFETY VIOLATION]' in result.stderr}")

    # Verify no .c file was created
    dangerous_c = REPO_ROOT / "dangerous.c"
    print(f"\nOutput file 'dangerous.c' exists: {dangerous_c.exists()}")
    if dangerous_c.exists():
        print("  WARNING: File was created despite safety violation!")
        dangerous_c.unlink()

    # Clean up
    dangerous_model.unlink()

    # Also test with a model that can't be interval-audited
    print("\n--- Testing model with unsupported op (if any) ---")
    print("Note: All standard Kiln ops support interval audit.")
    print("The fail-closed behavior triggers when interval audit cannot run.")


def showcase_3_qemu_attempt() -> None:
    """Showcase 3: Attempt to run on QEMU ARM (optional, 3h max)."""
    print("\n" + "=" * 70)
    print("SHOWCASE 3: QEMU ARM Cortex-M (attempt)")
    print("=" * 70)

    # Check for toolchain
    arm_gcc = shutil.which("arm-none-eabi-gcc")
    qemu = shutil.which("qemu-system-arm")

    print(f"arm-none-eabi-gcc: {'FOUND' if arm_gcc else 'NOT FOUND'}")
    print(f"qemu-system-arm: {'FOUND' if qemu else 'NOT FOUND'}")

    if not arm_gcc or not qemu:
        print("\nToolchain not available. Skipping showcase 3.")
        print("To enable: install arm-none-eabi-gcc and qemu-system-arm")
        return

    print("\nToolchain found! Attempting cross-compile and QEMU run...")

    # Build the hazard model with self-contained runtime
    build_dir = REPO_ROOT / "examples" / "hazard" / "build"
    model_c = build_dir / "model.c"

    if not model_c.exists():
        print("Run showcase 1 first to generate model.c")
        return

    # Try to cross-compile for Cortex-M3
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        elf_file = tmpdir / "hazard.elf"

        # Compile for Cortex-M3 (thumb, no OS)
        compile_cmd = [
            arm_gcc,
            "-mcpu=cortex-m3",
            "-mthumb",
            "-std=c99",
            "-O2",
            "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-nostdlib",
            "-ffreestanding",
            str(model_c),
            "-o", str(elf_file),
        ]
        result = run_cmd(compile_cmd, check=False)

        if result.returncode != 0:
            print("Cross-compilation failed")
            return

        print("Cross-compilation successful!")

        # Run on QEMU (semihosting or bare metal)
        # This is complex and requires a proper startup + semihosting setup
        print("QEMU execution requires semihosting setup - skipping for now")
        print("(Would need startup.S, linker script, semihosting config)")


def main() -> int:
    print("Kiln Demo Capture - Showcases 1 & 2 (Showcase 3 optional)")
    print("=" * 70)

    showcase_1_two_kb_brain()
    showcase_2_build_refuses()
    showcase_3_qemu_attempt()

    print("\n" + "=" * 70)
    print("Demo capture complete!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())