from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc

__all__ = ("export_to_kir", "quantize_tensor", "compute_multiplier_and_shift")


def compute_multiplier_and_shift(scale: float) -> tuple[int, int]:
    """
    Computes Q31 multiplier in [1<<30, 1<<31) and power-of-two right shift in [0, 31]
    such that scale ~= (multiplier / 2^31) * 2^(-shift).

    Fails early on non-positive/non-finite scales or scales >= 1.0 to preserve
    Kiln invariant (0 <= shift <= 31).
    """
    if math.isnan(scale) or math.isinf(scale):
        raise ValueError(f"quantization scale must be finite, got {scale}")
    if scale <= 0.0:
        raise ValueError(f"quantization scale must be strictly positive, got {scale}")
    if scale >= 1.0:
        raise ValueError(
            f"effective scale {scale} >= 1.0 is unsupported "
            f"(requires shift < 0, but Kiln invariant requires 0 <= shift <= 31)"
        )

    q, exp = math.frexp(scale)
    multiplier = round(q * (1 << 31))
    if multiplier >= (1 << 31):
        multiplier = (1 << 31) - 1
    elif multiplier < (1 << 30):
        multiplier = 1 << 30

    shift = -exp
    if shift < 0:
        raise ValueError(
            f"effective scale {scale} produces negative shift {shift} (unsupported left-shift)"
        )
    if shift > 31:
        raise ValueError(
            f"effective scale {scale} underflows shift range (shift {shift} > 31)"
        )

    return int(multiplier), int(shift)


def quantize_tensor(tensor: torch.Tensor) -> tuple[Any, float]:
    """
    Quantizes a floating point tensor to int8 range [-127, 127] symmetrically.
    Returns (nested list of int8, scale_factor).
    """
    t = tensor.detach().cpu().to(torch.float32)
    max_val = float(torch.max(torch.abs(t)).item())
    if max_val < 1e-7:
        scale = 1.0
        q = torch.zeros_like(t, dtype=torch.int8)
    else:
        scale = max_val / 127.0
        q = torch.clamp(torch.round(t / scale), -127, 127).to(torch.int8)

    # Convert to nested list
    return q.tolist(), scale


def _generate_lut(func: Any) -> list[int]:
    """Generates a 256-entry lookup table for int8 inputs [-128..127]."""
    x = torch.arange(-128, 128, dtype=torch.float32)
    y = func(x)
    y_clamped = torch.clamp(torch.round(y), -128, 127).to(torch.int8)
    return [int(val.item()) for val in y_clamped]


def export_to_kir(
    model: nn.Module,
    dummy_input: torch.Tensor,
) -> tuple[Graph, dict[str, Any]]:
    """
    Exports a sequential-like PyTorch model into a validated kiln.ir.Graph and weights dict.
    Supports:
      - nn.Embedding
      - nn.Conv1d
      - nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid (as LUT)
      - nn.Linear / MatMul
      - nn.MaxPool1d, nn.AdaptiveMaxPool1d
    Automatically adds an ARGMAX op at the end to satisfy kiln Phase 0 execution.
    """
    model = model.eval()

    # Flatten modules if Sequential or container
    modules: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            modules.append((name, mod))

    if not modules:
        raise ValueError("Model contains no leaf modules to export")

    weights: dict[str, Any] = {}
    nodes: list[Node] = []
    tensors: dict[str, TensorDesc] = {}

    input_shape = tuple(dummy_input.shape)
    first_mod = modules[0][1]
    input_dtype = DType.UINT8 if isinstance(first_mod, nn.Embedding) else DType.INT8

    current_tensor_name = "graph_in"
    tensors[current_tensor_name] = TensorDesc(
        name=current_tensor_name,
        shape=input_shape,
        dtype=input_dtype,
    )
    graph_inputs = [current_tensor_name]

    # Run dummy input forward tracking shapes
    with torch.no_grad():
        current_x = dummy_input.clone()
        node_idx = 0

        for name, mod in modules:
            out_tensor_name = f"t_{name or 'node'}_{node_idx}"
            node_idx += 1

            if isinstance(mod, nn.Embedding):
                w_list, scale = quantize_tensor(mod.weight.data)
                w_name = f"w_{name or 'emb'}"
                weights[w_name] = w_list

                current_x = mod(current_x.long())
                out_shape = tuple(current_x.shape)
                tensors[out_tensor_name] = TensorDesc(
                    name=out_tensor_name,
                    shape=out_shape,
                    dtype=DType.INT8,
                )
                nodes.append(
                    Node(
                        OpType.EMBEDDING,
                        f"node_{name or 'emb'}_{node_idx}",
                        inputs=[current_tensor_name],
                        outputs=[out_tensor_name],
                        attrs={"weight": w_name},
                    )
                )

            elif isinstance(mod, nn.Conv1d):
                w_list, scale = quantize_tensor(mod.weight.data)
                w_name = f"w_{name or 'conv'}"
                weights[w_name] = w_list

                # Approximate scale factor for accumulation
                # Scale factor to map integer accumulator into [-127, 127]
                in_ch, k_len = mod.weight.shape[1], mod.weight.shape[2]
                fan_in = in_ch * k_len
                # Max possible accumulator in int8 dot-product is fan_in * 127 * 127
                # With normal distribution, 3-sigma peak is roughly 127 * 127 * sqrt(fan_in) / 3
                # We scale so expected peak fits in int8:
                est_peak = max(127.0, (127.0 * 127.0 * math.sqrt(max(1, fan_in))) / 3.0)
                eff_scale = 127.0 / est_peak
                mult, shift = compute_multiplier_and_shift(eff_scale)
                qp = QuantParam(multiplier=mult, shift=shift)

                # Forward dummy through conv1d
                if current_x.dim() == 2:
                    current_x = current_x.float()
                current_x = mod(current_x.float())
                out_shape = tuple(current_x.shape)

                tensors[out_tensor_name] = TensorDesc(
                    name=out_tensor_name,
                    shape=out_shape,
                    dtype=DType.INT8,
                    quant=qp,
                )
                stride = mod.stride[0] if isinstance(mod.stride, (list, tuple)) else mod.stride
                padding = mod.padding[0] if isinstance(mod.padding, (list, tuple)) else mod.padding
                nodes.append(
                    Node(
                        OpType.CONV1D,
                        f"node_{name or 'conv'}_{node_idx}",
                        inputs=[current_tensor_name],
                        outputs=[out_tensor_name],
                        attrs={"weight": w_name, "stride": stride, "padding": padding},
                    )
                )

            elif isinstance(mod, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
                if isinstance(mod, nn.ReLU):
                    lut = _generate_lut(F.relu)
                elif isinstance(mod, nn.GELU):
                    lut = _generate_lut(F.gelu)
                elif isinstance(mod, nn.Tanh):
                    lut = _generate_lut(lambda x: 127.0 * torch.tanh(x / 32.0))
                else:
                    lut = _generate_lut(lambda x: 127.0 * torch.sigmoid(x / 16.0))

                current_x = mod(current_x)
                out_shape = tuple(current_x.shape)
                tensors[out_tensor_name] = TensorDesc(
                    name=out_tensor_name,
                    shape=out_shape,
                    dtype=DType.INT8,
                )
                nodes.append(
                    Node(
                        OpType.LUT,
                        f"node_{name or 'lut'}_{node_idx}",
                        inputs=[current_tensor_name],
                        outputs=[out_tensor_name],
                        attrs={"table": lut},
                    )
                )

            elif isinstance(mod, nn.Linear):
                # PyTorch linear: weight is (out_features, in_features)
                w_list, scale = quantize_tensor(mod.weight.data)
                w_name = f"w_{name or 'fc'}"
                weights[w_name] = w_list

                in_f = mod.weight.shape[1]
                est_peak = max(127.0, (127.0 * 127.0 * math.sqrt(max(1, in_f))) / 3.0)
                eff_scale = 127.0 / est_peak
                mult, shift = compute_multiplier_and_shift(eff_scale)
                qp = QuantParam(multiplier=mult, shift=shift)

                current_x = mod(current_x.float())
                out_shape = tuple(current_x.shape)
                tensors[out_tensor_name] = TensorDesc(
                    name=out_tensor_name,
                    shape=out_shape,
                    dtype=DType.INT8,
                    quant=qp,
                )
                nodes.append(
                    Node(
                        OpType.MATMUL,
                        f"node_{name or 'fc'}_{node_idx}",
                        inputs=[current_tensor_name],
                        outputs=[out_tensor_name],
                        attrs={"weight": w_name},
                    )
                )

            elif isinstance(mod, (nn.MaxPool1d, nn.AdaptiveMaxPool1d)):
                if isinstance(mod, nn.AdaptiveMaxPool1d):
                    length = current_x.shape[-1]
                    target_out = mod.output_size
                    if target_out != 1:
                        raise ValueError(f"AdaptiveMaxPool1d only supports output_size=1, got {target_out}")
                    k = length
                    stride = 1
                    padding = 0
                else:
                    k = mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[0]
                    stride = mod.stride if isinstance(mod.stride, int) else mod.stride[0]
                    padding = mod.padding if isinstance(mod.padding, int) else mod.padding[0]

                current_x = mod(current_x)
                out_shape = tuple(current_x.shape)
                tensors[out_tensor_name] = TensorDesc(
                    name=out_tensor_name,
                    shape=out_shape,
                    dtype=DType.INT8,
                )
                nodes.append(
                    Node(
                        OpType.MAXPOOL1D,
                        f"node_{name or 'pool'}_{node_idx}",
                        inputs=[current_tensor_name],
                        outputs=[out_tensor_name],
                        attrs={"kernel_size": k, "stride": stride, "padding": padding},
                    )
                )

            current_tensor_name = out_tensor_name

        # Ensure ARGMAX terminates the model for Phase 0
        final_out_name = "graph_out"
        tensors[final_out_name] = TensorDesc(
            name=final_out_name,
            shape=(),
            dtype=DType.INT8,
        )
        nodes.append(
            Node(
                OpType.ARGMAX,
                "node_final_argmax",
                inputs=[current_tensor_name],
                outputs=[final_out_name],
            )
        )
        graph_outputs = [final_out_name]

    graph = Graph(
        inputs=graph_inputs,
        outputs=graph_outputs,
        nodes=nodes,
        tensors=tensors,
    )
    graph.validate()
    return graph, weights