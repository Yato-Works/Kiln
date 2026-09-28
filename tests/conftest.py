"""Shared fixtures for the declarative spec (kiln.yaml) test modules."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml


@pytest.fixture
def temp_dir() -> Path:
    """A private temp directory (repo convention: pytest's tmp_path base can be
    ACL-locked on some hosts, while ``tempfile.mkdtemp`` always works)."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)

#: Self-contained kiln.ir model factory shared by test_spec.py and
#: test_cli_build.py: a 4-token embedding classifier with no training-data
#: dependency.  The factory honours the spec loader's ``weights_path`` kwarg
#: convention (defaults to deterministic placeholder weights).
TOY_MODEL_SOURCE = '''"""Toy kiln spec fixture: a 4-token bag-of-embeddings classifier."""

from __future__ import annotations

import json
from pathlib import Path

from kiln.frontend.pytorch import compute_multiplier_and_shift
from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc

VOCAB = 256
DIM = 4
LABELS = 3


def build(weights_path: str | None = None) -> tuple[Graph, dict]:
    """Graph + weights factory honouring the spec loader's `weights_path` kwarg."""
    tensors = {
        "tokens": TensorDesc("tokens", shape=(4,), dtype=DType.UINT8),
        "emb": TensorDesc("emb", shape=(4, DIM), dtype=DType.INT8),
        "logits": TensorDesc(
            "logits",
            shape=(4, LABELS),
            dtype=DType.INT8,
            quant=QuantParam(*compute_multiplier_and_shift(127.0 / 37772)),
        ),
        "pooled": TensorDesc("pooled", shape=(1, LABELS), dtype=DType.INT8),
        "label": TensorDesc("label", shape=(), dtype=DType.INT8),
    }
    nodes = [
        Node(OpType.EMBEDDING, "op_emb", inputs=["tokens"], outputs=["emb"], attrs={"weight": "w_emb"}),
        Node(OpType.MATMUL, "op_mm", inputs=["emb"], outputs=["logits"], attrs={"weight": "w_mm"}),
        Node(
            OpType.MAXPOOL1D,
            "op_pool",
            inputs=["logits"],
            outputs=["pooled"],
            attrs={"axis": 0, "kernel_size": 4, "stride": 4, "padding": 0},
        ),
        Node(OpType.ARGMAX, "op_argmax", inputs=["pooled"], outputs=["label"]),
    ]
    graph = Graph(inputs=["tokens"], outputs=["label"], nodes=nodes, tensors=tensors)
    if weights_path is None:
        weights = {
            "w_emb": [[(row + col) % 5 - 2 for col in range(DIM)] for row in range(VOCAB)],
            "w_mm": [[1, -1, 1, -1] for _ in range(LABELS)],
        }
        return graph, weights
    return graph, json.loads(Path(weights_path).read_text(encoding="utf-8"))
'''


@pytest.fixture
def toy_project(temp_dir: Path) -> Path:
    """A spec project directory: toy model fixture module + JSON weights."""
    (temp_dir / "toy_fixture.py").write_text(TOY_MODEL_SOURCE, encoding="utf-8")
    weights: dict[str, Any] = {
        "w_emb": [[(row + col) % 5 - 2 for col in range(4)] for row in range(256)],
        "w_mm": [[1, -1, 1, -1] for _ in range(3)],
    }
    (temp_dir / "weights.json").write_text(json.dumps(weights), encoding="utf-8")
    return temp_dir


@pytest.fixture
def toy_spec_factory(toy_project: Path) -> Callable[..., Path]:
    """Writes ``kiln.yaml`` into the toy project and returns its path."""

    def _write(
        *,
        budget: dict[str, Any] | None = None,
        frontend: dict[str, Any] | None = None,
        input_shape: tuple[int, ...] = (4,),
        entry_point: str = "toy_fixture.py:build",
        weights: str | None = "weights.json",
        name: str = "toy-task",
    ) -> Path:
        model: dict[str, Any] = {"entry_point": entry_point}
        if weights is not None:
            model["weights"] = weights
        spec: dict[str, Any] = {
            "name": name,
            "version": 1,
            "task": "classification",
            "domain": "general",
            "input": {"name": "tokens", "shape": list(input_shape), "dtype": "uint8"},
            "output": {"name": "label", "classes": ["alpha", "beta", "gamma"]},
            "model": model,
            "budget": budget
            or {"ram_arena_max_bytes": 2048, "rom_total_max_bytes": 40960},
        }
        if frontend is not None:
            spec["frontend"] = frontend
        path = toy_project / "kiln.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        return path

    return _write


#: A spec project whose MATMUL cannot fit an int32 accumulator: 200,000 int8
#: weights of 127 times the int8 input domain [-128, 127] reaches 3,225,800,000 >
#: INT32_MAX -- the same failure ``tests/test_interval.py`` pins down at the
#: propagation layer, reused here to exercise the *gate* the build puts on it.
#: The declared budget is deliberately huge so only the safety gate can fail.
OVERFLOW_MODEL_SOURCE = '''"""Synthetic kiln spec fixture with a guaranteed int32 accumulator overflow."""

from __future__ import annotations

from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc

COUNT = 200000


def build(weights_path: str | None = None) -> tuple[Graph, dict]:
    """Graph + weights factory honouring the spec loader's `weights_path` kwarg."""
    tensors = {
        "x": TensorDesc("x", shape=(1, COUNT), dtype=DType.INT8),
        "logits": TensorDesc(
            "logits",
            shape=(1, 1),
            dtype=DType.INT8,
            quant=QuantParam(multiplier=1073741824, shift=1),
        ),
        "label": TensorDesc("label", shape=(), dtype=DType.INT8),
    }
    nodes = [
        Node(
            OpType.MATMUL,
            "fc_overflow",
            inputs=["x"],
            outputs=["logits"],
            attrs={"weight": "w_fc"},
        ),
        Node(OpType.ARGMAX, "argmax", inputs=["logits"], outputs=["label"]),
    ]
    graph = Graph(inputs=["x"], outputs=["label"], nodes=nodes, tensors=tensors)
    weights = {"w_fc": [[127 for _ in range(COUNT)]]}
    return graph, weights
'''


@pytest.fixture
def overflow_spec_factory(temp_dir: Path) -> Callable[[], Path]:
    """Writes the overflowing ``kiln.yaml`` project and returns its path."""

    def _write() -> Path:
        project = temp_dir / "overflow"
        project.mkdir(parents=True, exist_ok=True)
        (project / "overflow_fixture.py").write_text(OVERFLOW_MODEL_SOURCE, encoding="utf-8")
        spec = {
            "name": "overflow-task",
            "version": 1,
            "task": "classification",
            "domain": "general",
            "input": {"name": "x", "shape": [1, 200000], "dtype": "int8"},
            "output": {"name": "label", "classes": ["c0"]},
            "model": {"entry_point": "overflow_fixture.py:build", "weights": None},
            "budget": {"ram_arena_max_bytes": 10000000, "rom_total_max_bytes": 10000000},
        }
        path = project / "kiln.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        return path

    return _write
