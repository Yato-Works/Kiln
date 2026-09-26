"""Shared Speech-Commands -> Kiln IR bridge for the Phase 1 KWS benchmarks.

Every Phase 1a script (``tests/test_kws.py``, ``evaluate.py``, ``ptq_test.py``,
``qat_eval.py``) used to hand-write the same 9 tensors, 8 nodes and calibration
math.  That duplication made architecture changes expensive, so the graph is now
derived from the ``nn.Sequential`` model itself:

* convolution kernels/weights come from the PyTorch layers,
* activation scales come from the ``CALIBRATION_QUANTILE`` quantile of each
  convolution output on the validation split,
* ``ReLU`` becomes a ``LUT`` and ``MaxPool1d`` becomes ``MAXPOOL1D``.

Changing ``KWS_ARCHITECTURE`` in ``examples/kws/model.py`` therefore updates the
emitted C, the budget certificate and every benchmark at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping

import torch
import torch.nn as nn

from examples.kws.model import N_FRAMES, N_MELS, quantize_logmel_to_int8
from kiln.frontend.pytorch import compute_multiplier_and_shift, quantize_tensor
from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc

__all__ = (
    "CALIBRATION_QUANTILE",
    "KWS_GRAPH_INPUT",
    "KWS_GRAPH_OUTPUT",
    "KwsKilnGraph",
    "calibrate_kws_graph",
    "split_indices",
)

KWS_GRAPH_INPUT: Final = "graph_in"
KWS_GRAPH_OUTPUT: Final = "graph_out"

#: int8 log-mel inputs are interpreted as ``q / 127.0``, i.e. S_in = 1/127.
INPUT_SCALE: Final = 1.0 / 127.0

#: Per-tensor activation scale = quantile(|activation|) / 127 (clip-and-scale).
CALIBRATION_QUANTILE: Final = 0.999

#: ReLU as a 256-entry int8 lookup table indexed by the raw byte value.
RELU_TABLE: Final[tuple[int, ...]] = tuple(max(0, index) for index in range(-128, 128))


@dataclass(frozen=True, slots=True)
class KwsKilnGraph:
    """A quantized Kiln graph plus the exact weights it references."""

    graph: Graph
    weights: Mapping[str, Any]
    activation_scales: tuple[float, ...]


def split_indices(
    num_total: int,
    *,
    seed: int = 1337,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic (train, validation, test) index split shared by all scripts."""
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_total, generator=generator)
    n_train = int(num_total * train_fraction)
    n_val = int(num_total * val_fraction)
    return (
        permutation[:n_train],
        permutation[n_train : n_train + n_val],
        permutation[n_train + n_val :],
    )


def _layer_outputs(model: nn.Sequential, inputs: torch.Tensor) -> list[torch.Tensor]:
    """Runs the model layer by layer, returning every layer's output tensor."""
    outputs: list[torch.Tensor] = []
    activations = inputs
    with torch.no_grad():
        for layer in model:
            activations = layer(activations)
            outputs.append(activations)
    return outputs


def _scalar_attr(value: int | tuple[int, ...]) -> int:
    return int(value[0]) if isinstance(value, tuple) else int(value)


def calibrate_kws_graph(
    model: nn.Sequential,
    calibration_specs: torch.Tensor,
) -> KwsKilnGraph:
    """Lowers ``model`` into Kiln IR with int8 weights and calibration scales."""
    calibration = quantize_logmel_to_int8(calibration_specs).float() / 127.0
    layer_outputs = _layer_outputs(model, calibration)

    tensors: dict[str, TensorDesc] = {
        KWS_GRAPH_INPUT: TensorDesc(
            KWS_GRAPH_INPUT,
            shape=(N_MELS, N_FRAMES),
            dtype=DType.INT8,
        ),
    }
    weights: dict[str, Any] = {}
    nodes: list[Node] = []
    scales: list[float] = []

    source = KWS_GRAPH_INPUT
    source_scale = INPUT_SCALE
    stage = 0

    for index, layer in enumerate(model):
        output = layer_outputs[index]
        shape = tuple(int(dim) for dim in output.shape[1:])

        if isinstance(layer, nn.Conv1d):
            weight_name = f"w{stage}"
            weight_q, weight_scale = quantize_tensor(layer.weight.data)
            activation = output.abs().flatten()
            output_scale = float(
                torch.quantile(activation, CALIBRATION_QUANTILE).item()
            ) / 127.0
            target = f"t_conv{stage}"
            weights[weight_name] = weight_q
            tensors[target] = TensorDesc(
                target,
                shape=shape,
                dtype=DType.INT8,
                quant=QuantParam(
                    *compute_multiplier_and_shift(
                        (source_scale * weight_scale) / output_scale
                    )
                ),
            )
            nodes.append(
                Node(
                    OpType.CONV1D,
                    f"node_{len(nodes)}",
                    inputs=[source],
                    outputs=[target],
                    attrs={
                        "weight": weight_name,
                        "stride": 1,
                        "padding": _scalar_attr(layer.padding),
                    },
                )
            )
            scales.append(output_scale)
            source, source_scale = target, output_scale

        elif isinstance(layer, nn.ReLU):
            target = f"t_relu{stage}"
            tensors[target] = TensorDesc(target, shape=shape, dtype=DType.INT8)
            nodes.append(
                Node(
                    OpType.LUT,
                    f"node_{len(nodes)}",
                    inputs=[source],
                    outputs=[target],
                    attrs={"table": RELU_TABLE},
                )
            )
            source = target

        elif isinstance(layer, nn.MaxPool1d):
            target = f"t_pool{stage}"
            tensors[target] = TensorDesc(target, shape=shape, dtype=DType.INT8)
            nodes.append(
                Node(
                    OpType.MAXPOOL1D,
                    f"node_{len(nodes)}",
                    inputs=[source],
                    outputs=[target],
                    attrs={
                        "kernel_size": _scalar_attr(layer.kernel_size),
                        "stride": _scalar_attr(layer.stride),
                        "padding": _scalar_attr(layer.padding),
                    },
                )
            )
            source = target
            stage += 1

        else:
            raise TypeError(f"unsupported KWS layer: {type(layer).__name__}")

    tensors[KWS_GRAPH_OUTPUT] = TensorDesc(
        KWS_GRAPH_OUTPUT,
        shape=(),
        dtype=DType.INT8,
    )
    nodes.append(
        Node(
            OpType.ARGMAX,
            f"node_{len(nodes)}",
            inputs=[source],
            outputs=[KWS_GRAPH_OUTPUT],
        )
    )

    graph = Graph(
        inputs=[KWS_GRAPH_INPUT],
        outputs=[KWS_GRAPH_OUTPUT],
        nodes=nodes,
        tensors=tensors,
    )
    return KwsKilnGraph(
        graph=graph,
        weights=weights,
        activation_scales=tuple(scales),
    )
