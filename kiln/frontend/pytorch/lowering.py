from __future__ import annotations

import math
import operator
import re
from pathlib import Path
from typing import Any
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx

from ...ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc
from .diagnostics import LoweringDiagnostic, UnsupportedGraphError
from .quantize import compute_multiplier_and_shift, quantize_tensor


def _node_location(node: torch.fx.Node) -> tuple[str | None, int | None]:
    """Extract a best-effort source location retained by torch.fx."""
    trace = getattr(node, "stack_trace", None) or ""
    match = re.search(r'File ["\']([^"\']+)["\'], line (\d+)', trace)
    if match:
        return str(Path(match.group(1))), int(match.group(2))
    return None, None


def _diagnostic(code: str, message: str, node: torch.fx.Node) -> LoweringDiagnostic:
    source, line = _node_location(node)
    return LoweringDiagnostic(code, message, node=node.name, source=source, line=line)


def _generate_lut(func: Any) -> list[int]:
    """Generates a 256-entry lookup table for int8 inputs [-128..127]."""
    x = torch.arange(-128, 128, dtype=torch.float32)
    y = func(x)
    y_clamped = torch.clamp(torch.round(y), -128, 127).to(torch.int8)
    return [int(val.item()) for val in y_clamped]


def _export_fx(
    model: nn.Module,
    dummy_input: torch.Tensor,
) -> tuple[Graph, dict[str, Any]]:
    """Exports a general PyTorch model DAG (including residual/skip connections) using torch.fx."""
    traced = torch.fx.symbolic_trace(model)

    weights: dict[str, Any] = {}
    nodes: list[Node] = []
    tensors: dict[str, TensorDesc] = {}
    env: dict[torch.fx.Node, tuple[str, torch.Tensor]] = {}

    submodules = dict(model.named_modules())
    graph_inputs: list[str] = []
    node_idx = 0

    with torch.no_grad():
        for n in traced.graph.nodes:
            if n.op == "placeholder":
                t_name = "graph_in" if node_idx == 0 else f"graph_in_{node_idx}"
                node_idx += 1
                shape = tuple(dummy_input.shape)
                # First module or input dtype
                first_mod = next(
                    (m for m in submodules.values() if len(list(m.children())) == 0), None
                )
                dtype = DType.UINT8 if isinstance(first_mod, nn.Embedding) else DType.INT8
                tensors[t_name] = TensorDesc(name=t_name, shape=shape, dtype=dtype)
                graph_inputs.append(t_name)
                env[n] = (t_name, dummy_input.clone())

            elif n.op == "call_module":
                mod = submodules[n.target]
                in_name, in_val = env[n.args[0]]
                out_name = f"t_{n.name.replace('.', '_')}_{node_idx}"
                node_idx += 1

                if isinstance(mod, nn.Embedding):
                    w_list, scale = quantize_tensor(mod.weight.data)
                    w_name = f"w_{n.name.replace('.', '_')}"
                    weights[w_name] = w_list
                    out_val = mod(in_val.long())
                    tensors[out_name] = TensorDesc(
                        name=out_name, shape=tuple(out_val.shape), dtype=DType.INT8
                    )
                    nodes.append(
                        Node(
                            OpType.EMBEDDING,
                            f"node_{out_name}",
                            inputs=[in_name],
                            outputs=[out_name],
                            attrs={"weight": w_name},
                        )
                    )
                    env[n] = (out_name, out_val)

                elif isinstance(mod, nn.Conv1d):
                    w_list, scale = quantize_tensor(mod.weight.data)
                    w_name = f"w_{n.name.replace('.', '_')}"
                    weights[w_name] = w_list
                    in_ch, k_len = mod.weight.shape[1], mod.weight.shape[2]
                    fan_in = in_ch * k_len
                    est_peak = max(127.0, (127.0 * 127.0 * math.sqrt(max(1, fan_in))) / 3.0)
                    eff_scale = 127.0 / est_peak
                    mult, shift = compute_multiplier_and_shift(eff_scale)
                    qp = QuantParam(multiplier=mult, shift=shift)

                    out_val = mod(in_val.float())
                    tensors[out_name] = TensorDesc(
                        name=out_name, shape=tuple(out_val.shape), dtype=DType.INT8, quant=qp
                    )
                    stride = mod.stride[0] if isinstance(mod.stride, (list, tuple)) else mod.stride
                    padding = (
                        mod.padding[0] if isinstance(mod.padding, (list, tuple)) else mod.padding
                    )
                    nodes.append(
                        Node(
                            OpType.CONV1D,
                            f"node_{out_name}",
                            inputs=[in_name],
                            outputs=[out_name],
                            attrs={"weight": w_name, "stride": stride, "padding": padding},
                        )
                    )
                    env[n] = (out_name, out_val)

                elif isinstance(mod, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
                    if isinstance(mod, nn.ReLU):
                        lut = _generate_lut(F.relu)
                    elif isinstance(mod, nn.GELU):
                        lut = _generate_lut(F.gelu)
                    elif isinstance(mod, nn.Tanh):
                        lut = _generate_lut(lambda x: 127.0 * torch.tanh(x / 32.0))
                    else:
                        lut = _generate_lut(lambda x: 127.0 * torch.sigmoid(x / 16.0))
                    out_val = mod(in_val)
                    tensors[out_name] = TensorDesc(
                        name=out_name, shape=tuple(out_val.shape), dtype=DType.INT8
                    )
                    nodes.append(
                        Node(
                            OpType.LUT,
                            f"node_{out_name}",
                            inputs=[in_name],
                            outputs=[out_name],
                            attrs={"table": lut},
                        )
                    )
                    env[n] = (out_name, out_val)

                elif isinstance(mod, nn.Linear):
                    w_list, scale = quantize_tensor(mod.weight.data)
                    w_name = f"w_{n.name.replace('.', '_')}"
                    weights[w_name] = w_list
                    in_f = mod.weight.shape[1]
                    est_peak = max(127.0, (127.0 * 127.0 * math.sqrt(max(1, in_f))) / 3.0)
                    eff_scale = 127.0 / est_peak
                    mult, shift = compute_multiplier_and_shift(eff_scale)
                    qp = QuantParam(multiplier=mult, shift=shift)

                    in_val_2d = in_val.float()
                    if in_val_2d.dim() > 2:
                        in_val_2d = in_val_2d.flatten(start_dim=1)
                    out_val = mod(in_val_2d)
                    tensors[out_name] = TensorDesc(
                        name=out_name, shape=tuple(out_val.shape), dtype=DType.INT8, quant=qp
                    )
                    nodes.append(
                        Node(
                            OpType.MATMUL,
                            f"node_{out_name}",
                            inputs=[in_name],
                            outputs=[out_name],
                            attrs={"weight": w_name},
                        )
                    )
                    env[n] = (out_name, out_val)

                elif isinstance(mod, (nn.MaxPool1d, nn.AdaptiveMaxPool1d)):
                    if isinstance(mod, nn.AdaptiveMaxPool1d):
                        length = in_val.shape[-1]
                        target_out = mod.output_size
                        if target_out != 1:
                            raise ValueError(
                                f"AdaptiveMaxPool1d only supports output_size=1, got {target_out}"
                            )
                        k = length
                        stride = 1
                        padding = 0
                    else:
                        k = (
                            mod.kernel_size
                            if isinstance(mod.kernel_size, int)
                            else mod.kernel_size[0]
                        )
                        stride = mod.stride if isinstance(mod.stride, int) else mod.stride[0]
                        padding = mod.padding if isinstance(mod.padding, int) else mod.padding[0]
                    out_val = mod(in_val)
                    tensors[out_name] = TensorDesc(
                        name=out_name, shape=tuple(out_val.shape), dtype=DType.INT8
                    )
                    nodes.append(
                        Node(
                            OpType.MAXPOOL1D,
                            f"node_{out_name}",
                            inputs=[in_name],
                            outputs=[out_name],
                            attrs={"kernel_size": k, "stride": stride, "padding": padding},
                        )
                    )
                    env[n] = (out_name, out_val)
                else:
                    raise UnsupportedGraphError(
                        _diagnostic(
                            "KILN-FX-001",
                            f"unsupported module type: {type(mod).__name__}",
                            n,
                        )
                    )

            elif n.op == "call_function":
                if n.target in (operator.add, torch.add) or (
                    hasattr(n.target, "__name__") and n.target.__name__ == "add"
                ):
                    arg0, arg1 = n.args[0], n.args[1]
                    left_name, left_val = env[arg0]
                    right_name, right_val = env[arg1]
                    out_name = f"t_add_{node_idx}"
                    node_idx += 1

                    out_val = left_val + right_val
                    mult, shift = compute_multiplier_and_shift(0.5)
                    qp = QuantParam(multiplier=mult, shift=shift)

                    tensors[out_name] = TensorDesc(
                        name=out_name, shape=tuple(left_val.shape), dtype=DType.INT8, quant=qp
                    )
                    nodes.append(
                        Node(
                            OpType.ADD,
                            f"node_{out_name}",
                            inputs=[left_name, right_name],
                            outputs=[out_name],
                        )
                    )
                    env[n] = (out_name, out_val)
                else:
                    raise UnsupportedGraphError(
                        _diagnostic("KILN-FX-002", f"unsupported function call: {n.target}", n)
                    )

            elif n.op == "call_method":
                # Handle view/flatten/reshape methods without compute
                if n.target in ("view", "reshape", "flatten", "squeeze", "unsqueeze"):
                    in_name, in_val = env[n.args[0]]
                    method = getattr(in_val, n.target)
                    out_val = method(*n.args[1:], **n.kwargs)
                    env[n] = (in_name, out_val)
                else:
                    raise UnsupportedGraphError(
                        _diagnostic("KILN-FX-003", f"unsupported method call: {n.target}", n)
                    )

            elif n.op == "output":
                final_node = n.args[0]
                final_name, final_val = env[final_node]

                final_out_name = "graph_out"
                tensors[final_out_name] = TensorDesc(
                    name=final_out_name, shape=(), dtype=DType.INT8
                )
                nodes.append(
                    Node(
                        OpType.ARGMAX,
                        "node_final_argmax",
                        inputs=[final_name],
                        outputs=[final_out_name],
                    )
                )
                graph_outputs = [final_out_name]

    graph = Graph(inputs=graph_inputs, outputs=graph_outputs, nodes=nodes, tensors=tensors)
    graph.validate()
    return graph, weights


def _export_sequential(
    model: nn.Module,
    dummy_input: torch.Tensor,
) -> tuple[Graph, dict[str, Any]]:
    """Fallback sequential exporter."""
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

                in_ch, k_len = mod.weight.shape[1], mod.weight.shape[2]
                fan_in = in_ch * k_len
                est_peak = max(127.0, (127.0 * 127.0 * math.sqrt(max(1, fan_in))) / 3.0)
                eff_scale = 127.0 / est_peak
                mult, shift = compute_multiplier_and_shift(eff_scale)
                qp = QuantParam(multiplier=mult, shift=shift)

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
                        raise ValueError(
                            f"AdaptiveMaxPool1d only supports output_size=1, got {target_out}"
                        )
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


def export_to_kir(
    model: nn.Module,
    dummy_input: torch.Tensor,
) -> tuple[Graph, dict[str, Any]]:
    """
    Exports a PyTorch model into a validated kiln.ir.Graph and weights dict.
    Supports:
      - nn.Embedding
      - nn.Conv1d
      - nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid (as LUT)
      - nn.Linear / MatMul
      - nn.MaxPool1d, nn.AdaptiveMaxPool1d
      - Residual / skip connections (+ or torch.add)
    Automatically adds an ARGMAX op at the end to satisfy kiln Phase 0 execution.
    """
    model = model.eval()
    try:
        return _export_fx(model, dummy_input)
    except UnsupportedGraphError:
        raise
    except Exception as exc:
        source = None
        line = None
        try:
            source = inspect.getsourcefile(model.forward)
            line = inspect.getsourcelines(model.forward)[1]
        except (OSError, TypeError):
            pass
        raise UnsupportedGraphError(
            LoweringDiagnostic(
                "KILN-FX-000",
                f"FX tracing failed; dynamic control flow and unsupported graph constructs "
                f"must be made explicit: {exc}",
                source=source,
                line=line,
            )
        ) from exc
