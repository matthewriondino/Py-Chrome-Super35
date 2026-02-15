"""
Native acceleration helpers for Py-Chrome video processing.

This package is optional at runtime:
- If compiled/native components are unavailable, callers should fall back to NumPy paths.
"""

from .backend_manager import (
    FrameProcessor,
    apply_white_balance_numpy,
    scientific_irg_transform_numpy,
    params_from_preset,
    normalize_backend_request,
    cpu_extension_available,
    cupy_backend_available,
    metal_backend_available,
)

__all__ = [
    "FrameProcessor",
    "apply_white_balance_numpy",
    "scientific_irg_transform_numpy",
    "params_from_preset",
    "normalize_backend_request",
    "cpu_extension_available",
    "cupy_backend_available",
    "metal_backend_available",
]
