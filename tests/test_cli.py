from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from kiln.cli import main


def _has_gcc() -> bool:
    return shutil.which("gcc") is not None


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_model_file(temp_dir: Path) -> Path:
    code = """
import torch
import torch.nn as nn

class SampleClassifier(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Embedding(256, 8),
            nn.Linear(8, 2),
        )

model = SampleClassifier()
dummy_input = torch.tensor([10, 20], dtype=torch.uint8)
"""
    model_path = temp_dir / "sample_model.py"
    model_path.write_text(code, encoding="utf-8")
    return model_path


def test_cli_ops(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["ops"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Kiln Supported IR Operations" in captured.out
    assert "EMBEDDING" in captured.out
    assert "CONV1D" in captured.out
    assert "MATMUL" in captured.out
    assert "ARGMAX" in captured.out


def test_cli_compile(sample_model_file: Path, temp_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_c = temp_dir / "compiled_model.c"
    ret = main(["compile", str(sample_model_file), "-o", str(out_c)])
    assert ret == 0
    assert out_c.exists()
    content = out_c.read_text(encoding="utf-8")
    assert "Kiln Model Static Budget Certificate" in content
    assert "kiln_predict" in content

    captured = capsys.readouterr()
    assert "[SUCCESS]" in captured.out
    assert "compiled_model.c" in captured.out


def test_cli_compile_runtime_header_none_is_self_contained(
    sample_model_file: Path, temp_dir: Path
) -> None:
    """`--runtime-header None` must drop the include, not emit `#include "None"`."""
    out_c = temp_dir / "self_contained.c"
    ret = main(
        ["compile", str(sample_model_file), "-o", str(out_c), "--runtime-header", "None"]
    )
    assert ret == 0
    content = out_c.read_text(encoding="utf-8")
    assert '#include "None"' not in content
    assert "#include <stdint.h>" in content


def test_cli_compile_refuses_int32_overflow(
    temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An overflowing graph is refused before any C is written (the compile gate).

    133,200 int8 weights of 127 times an int8 input domain of 127 reaches
    2,148,382,800 > INT32_MAX, so the *safety* gate -- not the budget checks,
    which ``compile`` does not run -- must stop the emission.
    """
    kernel = 133_200
    code = f"""
import torch
import torch.nn as nn

conv = nn.Conv1d(1, 1, kernel_size={kernel}, bias=False)
nn.init.constant_(conv.weight, 1.0)

model = nn.Sequential(conv)
dummy_input = torch.zeros(1, {kernel}, dtype=torch.uint8)
"""
    model_path = temp_dir / "overflow_model.py"
    model_path.write_text(code, encoding="utf-8")
    out_c = temp_dir / "overflow.c"

    ret = main(["compile", str(model_path), "-o", str(out_c)])

    assert ret == 1
    captured = capsys.readouterr()
    assert "[SAFETY VIOLATION]" in captured.err
    assert "int32 accumulator overflow detected" in captured.err
    assert "no C emitted" in captured.err
    assert "WARNING: Potential int32 overflow detected" in captured.out
    assert not out_c.exists()


def test_cli_compile_runtime_header_is_forwarded(
    sample_model_file: Path, temp_dir: Path
) -> None:
    """A custom header name is passed through verbatim (the default behaviour)."""
    out_c = temp_dir / "with_shim.c"
    ret = main(
        ["compile", str(sample_model_file), "-o", str(out_c), "--runtime-header", "my_shim.h"]
    )
    assert ret == 0
    assert '#include "my_shim.h"' in out_c.read_text(encoding="utf-8")


def test_cli_audit_text(sample_model_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["audit", str(sample_model_file)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Kiln Model Static Budget Certificate" in captured.out
    assert "Arena (.bss) RAM" in captured.out
    assert "Accumulator Safety" in captured.out


def test_cli_audit_intervals(sample_model_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--intervals` adds the detailed Interval Boundary Propagation report."""
    ret = main(["audit", str(sample_model_file), "--intervals"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Static Interval Boundary Analysis" in captured.out
    assert "Accumulator overflow safety" in captured.out
    assert "Tensor Value Bounds" in captured.out


def test_cli_audit_json(sample_model_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["audit", str(sample_model_file), "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "arena_ram_bytes" in data
    assert "weights_rom_bytes" in data
    assert "total_macs" in data
    assert data["dynamic_allocation_bytes"] == 0
    assert data["deterministic"] is True
    # Interval Boundary Propagation results ship in the JSON payload
    intervals = data["interval_analysis"]
    assert intervals["int32_overflow_free"] is True
    assert intervals["total_accum_ops"] == 1  # the Linear/matmul layer
    assert isinstance(intervals["tensor_intervals"], dict)
    assert isinstance(intervals["node_audits"], dict)


@pytest.mark.skipif(not _has_gcc(), reason="gcc required for cli verify test")
def test_cli_verify(sample_model_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["verify", str(sample_model_file), "--samples", "25"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "[PASSED]" in captured.out
    assert "100% bit-exact" in captured.out