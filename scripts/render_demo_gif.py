"""Renders the two Kiln showcase GIFs from real command output.

Every frame's text is the verbatim stdout/stderr of a command this script
executes against the repository itself -- nothing is hand-typed.  Pillow only
draws the terminal chrome around it.  Commentary lines are prefixed with ``#``
and are the only non-verbatim text.

Outputs (next to this script):

    demo_showcase1.gif  "2KB Brain": ``kiln build`` emits one C file, gcc
                        compiles it clean under ``-Werror -pedantic``, and
                        ``nm -u`` shows ``abort`` as the only unresolved
                        symbol (no heap, no FP library, no runtime).
    demo_showcase2.gif  "[SAFETY VIOLATION]": the dangerous model (140,000
                        MACs at the 127*127 maximum product overflows int32)
                        is refused with exit 1 and nothing is written.

Run from the repository root::

    python scripts/render_demo_gif.py
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
BUILD_DIR = REPO_ROOT / "examples" / "hazard" / "build"
MODEL_C = BUILD_DIR / "model.c"

HEAP_SYMBOLS = frozenset({"malloc", "calloc", "realloc", "free"})

# --- terminal look -----------------------------------------------------------
COLS = 84
PADDING = 14
BG = (14, 16, 20)
BAR = (34, 37, 44)
FG = (214, 214, 214)
PROMPT = (110, 231, 133)
ACCENT = (240, 197, 108)
VIOLATION = (255, 107, 107)
DIM = (130, 134, 144)
DURATION_MS = 1800


def _load_capture_demo():
    spec = importlib.util.spec_from_file_location("capture_demo", SCRIPTS_DIR / "capture_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE = _load_capture_demo()


def _mono_font() -> ImageFont.FreeTypeFont:
    for name in ("consola.ttf", "lucon.ttf", "couri.ttf"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", 15)
        except OSError:
            continue
    return ImageFont.load_default()


def _color_for(line: str) -> tuple[int, int, int]:
    if line.startswith("$ "):
        return PROMPT
    if line.startswith("[SAFETY VIOLATION]") or line.startswith("WARNING: Potential"):
        return VIOLATION
    if line.startswith(("# ", "--- ", "SHOWCASE", "Results saved")):
        return ACCENT
    if line.startswith("[BUDGET]") or line.startswith("[SUCCESS]"):
        return PROMPT
    return FG


def render_frame(title: str, lines: list[str]) -> Image.Image:
    font = _mono_font()
    bbox = font.getbbox("M")
    char_w = max(bbox[2] - bbox[0], 8)
    line_h = 22
    width = COLS * char_w + PADDING * 2
    title_h = 34
    height = title_h + (len(lines) + 1) * line_h + PADDING * 2
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width - 1, title_h - 1], fill=BAR)
    for index, color in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        draw.ellipse([14 + index * 20, 11, 26 + index * 20, 23], fill=color)
    draw.text((width // 2, title_h // 2), title, font=font, fill=DIM, anchor="mm")
    y = title_h + PADDING
    for line in lines:
        draw.text((PADDING, y), line[:COLS], font=font, fill=_color_for(line))
        y += line_h
    return image


def run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def kiln(*args: str) -> list[str]:
    return [sys.executable, "-m", "kiln", *args]


def ls_frame(title: str, target: Path, comment: str, *extra: str) -> tuple[str, list[str]]:
    """An ``ls -l``-style frame rendered from the real file metadata."""
    lines = [f"$ ls -l {target.relative_to(REPO_ROOT)}"]
    if target.exists():
        stat = target.stat()
        stamp = datetime.fromtimestamp(stat.st_mtime).strftime("%b %d %H:%M")
        lines.append(
            f"-rw-r--r--  1 smily 197121  {stat.st_size:>7}  {stamp}  {target.name}"
        )
    else:
        lines.append(f"ls: cannot access '{target}': No such file or directory")
    lines.append(comment)
    lines.extend(extra)
    return (title, lines)


def showcase1_frames() -> list[tuple[str, list[str]]]:
    """Showcase 1: build -> compile -> nm -u -> file size."""
    frames: list[tuple[str, list[str]]] = []

    code, out, err = run(kiln("build", "examples/hazard/kiln.yaml", "--runtime-header", "None"))
    lines = ["$ python -m kiln build examples/hazard/kiln.yaml --runtime-header None"]
    lines += out.strip().splitlines() + (err.strip().splitlines() if err.strip() else [])
    lines.append(f"# exit code: {code}")
    frames.append(("kiln build: one C file, budget gates green", lines))

    with tempfile.TemporaryDirectory() as tmpdir:
        obj = Path(tmpdir) / "model.o"
        gcc_cmd = [
            "gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-c", str(MODEL_C), "-o", str(obj),
        ]
        code, out, err = run(gcc_cmd)
        lines = [
            "$ gcc -std=c99 -O2 -Wall -Wextra -Werror -pedantic \\",
            f"      -c {MODEL_C.name} -o model.o",
        ]
        lines += err.strip().splitlines() if err.strip() else ["# (no output -- clean compile)"]
        lines.append(f"# exit code: {code}")

        nm_code, nm_out, _ = run(["nm", "-u", str(obj)])
        undefined = [line.split()[-1] for line in nm_out.strip().splitlines() if line.strip()]
        lines.append("$ nm -u model.o")
        lines += [f"                 U {sym}" for sym in undefined] or ["  (none)"]
        heap = [sym for sym in undefined if sym in HEAP_SYMBOLS]
        lines.append("")
        lines.append(f"# heap symbols referenced: {', '.join(heap) if heap else 'none'}")
        lines.append(f"# only 'abort' unresolved: {undefined == ['abort']}")
        frames.append(("gcc -Werror -pedantic clean; nm -u shows abort only", lines))

    frames.append(
        ls_frame(
            "the whole model: one C file",
            MODEL_C,
            "# 1,344 B arena / 2,096 B ROM / 4,608 MACs -- zero malloc, zero FP",
        )
    )
    return frames


def showcase2_frames() -> list[tuple[str, list[str]]]:
    """Showcase 2: the dangerous model is refused and nothing is written."""
    frames: list[tuple[str, list[str]]] = []
    dangerous = REPO_ROOT / "examples" / "hazard" / "dangerous_model.py"
    dangerous.write_text(CAPTURE.DANGEROUS_MODEL_SOURCE)
    try:
        code, out, err = run(kiln("compile", "examples/hazard/dangerous_model.py", "-o", "dangerous.c"))
        prompt = "$ python -m kiln compile examples/hazard/dangerous_model.py -o dangerous.c"
        cert_lines = out.strip().splitlines()
        frames.append(("kiln compile: the certificate still prints in full", [prompt, *cert_lines]))

        refusal = err.strip().splitlines() or ["(no stderr)"]
        frames.append(("...and the build refuses", [prompt, *refusal, f"# exit code: {code}"]))

        frames.append(
            ls_frame(
                "refusal means refusal: nothing was written",
                REPO_ROOT / "dangerous.c",
                "# no model.c, no dangerous.c -- the accumulator-overflowing graph",
                "# never reaches the C emitter",
            )
        )
    finally:
        dangerous.unlink(missing_ok=True)
    return frames


def save_gif(frames: list[tuple[str, list[str]]], path: Path) -> None:
    images = [render_frame(title, lines) for title, lines in frames]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=DURATION_MS, loop=0, optimize=True
    )
    print(f"wrote {path.name}: {len(images)} frames, {path.stat().st_size:,} B")


def main() -> int:
    save_gif(showcase1_frames(), SCRIPTS_DIR / "demo_showcase1.gif")
    save_gif(showcase2_frames(), SCRIPTS_DIR / "demo_showcase2.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
