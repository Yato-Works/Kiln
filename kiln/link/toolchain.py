"""C toolchain discovery, strict compilation and binary inspection (``kiln link``).

Phase 4's trust layer starts by *checking* the emitted translation unit instead
of trusting it: ``kiln link`` compiles the model together with the spec's
declared C frontend under the same strict flags the Phase 1/2 harnesses use
(``-std=c99 -O3 -Wall -Wextra -Werror -pedantic``), links a real executable, then
measures the artifact -- section sizes from the linked image and undefined
symbols from the model object, which is where the zero-malloc claim is checked
rather than asserted.

The compiler is discovered, never assumed: an explicit ``--cc`` path, then
``$KILN_CC``, then ``gcc`` / ``cc`` / ``clang`` / ``arm-none-eabi-gcc`` on the
PATH.  A cross toolchain therefore works through the same code path once a real
target profile exists (Phase 4's firmware milestone); on a host without any
compiler the command reports a clear error instead of a traceback.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

__all__ = (
    "MALLOC_SYMBOLS",
    "STRICT_FLAGS",
    "TARGET_PROFILES",
    "TargetProfile",
    "Toolchain",
    "ToolchainError",
    "detect_toolchain",
    "inspect_sections",
    "undefined_symbols",
)

#: Warnings are errors and the dialect is exactly C99, as in the example suites.
STRICT_FLAGS: tuple[str, ...] = ("-std=c99", "-O3", "-Wall", "-Wextra", "-Werror", "-pedantic")

#: Dynamic-allocation entry points: a linked model image must not reference any.
MALLOC_SYMBOLS: frozenset[str] = frozenset(
    {"malloc", "calloc", "realloc", "free", "aligned_alloc", "posix_memalign"}
)

#: Compilers probed in order when nothing is specified.
_CANDIDATES: tuple[str, ...] = ("gcc", "cc", "clang", "arm-none-eabi-gcc")

#: ``objdump -h`` rows: ``  0 .text  00000abc  0000000000401000 ...``.
_SECTION_ROW = re.compile(r"^\s*(?:\d+\s+)?(\.?[\w.$]+)\s+([0-9a-fA-F]+)\s+\S")


class ToolchainError(RuntimeError):
    """A missing compiler, a failed compile/link, or an unusable artifact."""


@dataclass(frozen=True, slots=True)
class TargetProfile:
    """Target architectural profile for host or cross compilation."""

    name: str
    arch_flags: tuple[str, ...] = ()
    link_flags: tuple[str, ...] = ()
    runner: str | None = None
    default_cc: str = "gcc"


TARGET_PROFILES: Mapping[str, TargetProfile] = {
    "host": TargetProfile(name="host", default_cc="gcc"),
    "cortex-m0plus": TargetProfile(
        name="cortex-m0plus",
        arch_flags=("-mcpu=cortex-m0plus", "-mthumb"),
        link_flags=("-mcpu=cortex-m0plus", "-mthumb", "--specs=nosys.specs"),
        runner="qemu-arm",
        default_cc="arm-none-eabi-gcc",
    ),
    "cortex-m4": TargetProfile(
        name="cortex-m4",
        arch_flags=("-mcpu=cortex-m4", "-mthumb"),
        link_flags=("-mcpu=cortex-m4", "-mthumb", "--specs=nosys.specs"),
        runner="qemu-arm",
        default_cc="arm-none-eabi-gcc",
    ),
    "riscv32": TargetProfile(
        name="riscv32",
        arch_flags=("-march=rv32imc", "-mabi=ilp32"),
        link_flags=("-march=rv32imc", "-mabi=ilp32"),
        runner="qemu-riscv32",
        default_cc="riscv64-unknown-elf-gcc",
    ),
}


@dataclass(frozen=True, slots=True)
class Toolchain:
    """A discovered C compiler plus the binutils next to it."""

    cc: str
    profile: TargetProfile = TARGET_PROFILES["host"]
    objdump: str | None = None
    nm: str | None = None

    @property
    def name(self) -> str:
        return Path(self.cc).stem


def _sibling(tool: str, cc: str) -> str | None:
    """The tool shipped next to the compiler, else whatever the PATH offers."""
    candidate = Path(cc).parent / tool
    for suffix in ("", ".exe"):
        if Path(f"{candidate}{suffix}").is_file():
            return str(Path(f"{candidate}{suffix}"))
    return shutil.which(tool)


def detect_toolchain(cc: str | None = None, *, target: str = "host") -> Toolchain:
    """Resolves a C compiler: ``cc`` argument, ``$KILN_CC``, target profile, then PATH."""
    profile = TARGET_PROFILES.get(target)
    if profile is None:
        valid = ", ".join(sorted(TARGET_PROFILES))
        raise ToolchainError(f"unknown target '{target}'; expected one of: {valid}")

    requested = cc or os.environ.get("KILN_CC") or None
    resolved: str | None = None

    if requested:
        resolved = shutil.which(requested)
        if resolved is None and Path(requested).is_file():
            resolved = str(Path(requested))
        if resolved is None:
            raise ToolchainError(f"C compiler not found: {requested}")
    elif target != "host":
        found = shutil.which(profile.default_cc)
        if found:
            resolved = found
        else:
            raise ToolchainError(
                f"target '{target}' compiler '{profile.default_cc}' not found on PATH; "
                f"pass --cc <path> or install the cross toolchain"
            )
    else:
        for candidate in _CANDIDATES:
            found = shutil.which(candidate)
            if found:
                resolved = found
                break

    if resolved is None:
        raise ToolchainError(
            "no C compiler found; pass --cc <path>, set KILN_CC, or install gcc/cc/clang"
        )

    # Resolve target-specific binutils if cross-compiling
    cc_stem = Path(resolved).stem
    prefix = ""
    if "-" in cc_stem:
        prefix = cc_stem.rsplit("gcc", 1)[0]
    objdump_name = f"{prefix}objdump" if prefix else "objdump"
    nm_name = f"{prefix}nm" if prefix else "nm"

    return Toolchain(
        cc=resolved,
        profile=profile,
        objdump=_sibling(objdump_name, resolved),
        nm=_sibling(nm_name, resolved),
    )


def _run(argv: Sequence[str], *, what: str) -> str:
    result = subprocess.run(list(argv), capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ToolchainError(f"{what} failed:\n  {' '.join(argv)}\n{detail}")
    return result.stdout


def compile_object(
    toolchain: Toolchain,
    source: Path,
    obj: Path,
    *,
    include_dirs: Sequence[Path] = (),
) -> None:
    """Compiles one translation unit with the strict flags and target architecture flags."""
    argv = [toolchain.cc, *STRICT_FLAGS, *toolchain.profile.arch_flags]
    argv += [f"-I{directory}" for directory in include_dirs]
    argv += ["-c", str(source), "-o", str(obj)]
    _run(argv, what=f"compiling {source.name}")


def link_executable(toolchain: Toolchain, objects: Sequence[Path], exe: Path) -> None:
    """Links the compiled objects into an executable using target link flags."""
    argv = [
        toolchain.cc,
        *toolchain.profile.link_flags,
        *[str(obj) for obj in objects],
        "-o",
        str(exe),
    ]
    _run(argv, what="linking")


def run_executable(toolchain: Toolchain, exe: Path) -> str:
    """Runs a linked artifact (directly on host or via target runner e.g. qemu-arm)."""
    runner = toolchain.profile.runner
    if runner is not None:
        resolved_runner = shutil.which(runner)
        if not resolved_runner:
            raise ToolchainError(
                f"runner '{runner}' required for target '{toolchain.profile.name}' not found on PATH"
            )
        return _run([resolved_runner, str(exe)], what=f"running {exe.name} via {runner}")
    return _run([str(exe)], what=f"running {exe.name}")


def _tool_output(tool: str | None, argv: Sequence[str]) -> str | None:
    if tool is None:
        return None
    result = subprocess.run([tool, *argv], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout


def inspect_sections(toolchain: Toolchain, artifact: Path) -> dict[str, int]:
    """Section sizes of a linked artifact (``{}`` when binutils are unavailable).

    PE images name read-only data ``.rdata`` where ELF says ``.rodata``; the
    caller sums both, so one report covers host and cross builds alike.
    """
    output = _tool_output(toolchain.objdump, ["-h", str(artifact)])
    if output is None:
        return {}
    sizes: dict[str, int] = {}
    for line in output.splitlines():
        match = _SECTION_ROW.match(line)
        if match and match.group(1).startswith("."):
            sizes[match.group(1)] = int(match.group(2), 16)
    return sizes


def undefined_symbols(toolchain: Toolchain, artifact: Path) -> set[str] | None:
    """Undefined symbols referenced by an object, or ``None`` without binutils."""
    output = _tool_output(toolchain.nm, ["-u", str(artifact)])
    if output is not None:
        names: set[str] = set()
        for line in output.splitlines():
            match = re.search(r"\bU\s+(\S+)\s*$", line)
            if match:
                names.add(match.group(1))
        return names
    output = _tool_output(toolchain.objdump, ["-t", str(artifact)])
    if output is None:
        return None
    return {line.split()[-1] for line in output.splitlines() if "*UND*" in line and line.split()}
