from typing import Any

__all__ = ("export_to_kir", "quantize_tensor", "compute_multiplier_and_shift")


def __getattr__(name: str) -> Any:
    """Load the optional PyTorch frontend only when one of its APIs is used."""
    if name == "export_to_kir":
        from .pytorch import export_to_kir

        return export_to_kir
    if name in {"quantize_tensor", "compute_multiplier_and_shift"}:
        from .pytorch.quantize import compute_multiplier_and_shift, quantize_tensor

        return {
            "quantize_tensor": quantize_tensor,
            "compute_multiplier_and_shift": compute_multiplier_and_shift,
        }[name]
    raise AttributeError(name)
