"""Phase 2's graph shape: a bag of tokens classified by ``ARGMAX`` (M2.2).

``EMBEDDING -> MATMUL -> MAXPOOL1D(axis=0) -> ARGMAX``: the embedding table turns
each ``uint8`` token id into ``dim`` int8 values, ``MATMUL`` scores every token
against every label, ``MAXPOOL1D(axis=0)`` keeps the strongest evidence per label
and ``ARGMAX`` returns the label.  The *shape* is fixed by M2.0/M2.2; the weights
are the distillation deliverable, so this module builds graphs and offers a
deterministic placeholder initialiser for the IR/contract tests.

Sizing (measured, see ROADMAP.md Phase 2): the arena is exactly
``tokens * (dim + labels)``, so the 2 KB budget decides the feature window.
"""

from __future__ import annotations

import random
from typing import Any, Final

from examples.hazard.tokenizer import TOKEN_COUNT, VOCAB_SIZE
from kiln.frontend.pytorch import compute_multiplier_and_shift
from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc

__all__ = (
    "DEFAULT_DIM",
    "DEFAULT_LABELS",
    "build_bag_graph",
    "label_quant",
    "random_weights",
)

#: Embedding width of the M2.0 contract: 96 tokens x (8 + 6) = 1,344 B of arena.
DEFAULT_DIM: Final = 8

#: Label count of the proposed hazard taxonomy (ROADMAP Phase 2, question 2).
DEFAULT_LABELS: Final = 6

#: Requantisation of the logits, i.e. the int8 scale the calibration would find.
_LABEL_SCALE: Final = 1.0 / 40.0


def label_quant() -> QuantParam:
    """The logit quantisation M2.2's calibration is expected to land on."""
    return QuantParam(*compute_multiplier_and_shift(_LABEL_SCALE))


def random_weights(
    dim: int = DEFAULT_DIM,
    labels: int = DEFAULT_LABELS,
    *,
    seed: int = 1337,
) -> dict[str, Any]:
    """Deterministic placeholder weights: the shape, without the training."""
    generator = random.Random(seed)
    return {
        "w_emb": [[generator.randint(-40, 40) for _ in range(dim)] for _ in range(VOCAB_SIZE)],
        "w_mm": [[generator.randint(-20, 20) for _ in range(dim)] for _ in range(labels)],
    }


def build_bag_graph(
    tokens: int = TOKEN_COUNT,
    dim: int = DEFAULT_DIM,
    labels: int = DEFAULT_LABELS,
) -> Graph:
    """Builds the Phase 2 classifier over ``tokens`` uint8 ids.

    ``MAXPOOL1D(axis=0)`` is what makes a text model expressible: without it the
    pool would reduce the *class* axis and ``ARGMAX`` would return a token index
    instead of a label (see ROADMAP.md, Phase 2 IR amendment).
    """
    tensors = {
        "tokens": TensorDesc("tokens", shape=(tokens,), dtype=DType.UINT8),
        "emb": TensorDesc("emb", shape=(tokens, dim), dtype=DType.INT8),
        "logits": TensorDesc("logits", shape=(tokens, labels), dtype=DType.INT8, quant=label_quant()),
        "pooled": TensorDesc("pooled", shape=(1, labels), dtype=DType.INT8),
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
            attrs={"axis": 0, "kernel_size": tokens, "stride": tokens, "padding": 0},
        ),
        Node(OpType.ARGMAX, "op_argmax", inputs=["pooled"], outputs=["label"]),
    ]
    return Graph(inputs=["tokens"], outputs=["label"], nodes=nodes, tensors=tensors)
