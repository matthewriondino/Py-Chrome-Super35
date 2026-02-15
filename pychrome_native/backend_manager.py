"""
Backend selection and compute paths for Py-Chrome.

Resolution order for auto mode:
- metal (if available and healthy)
- cupy (if available and healthy)
- cpu_ext (compiled C extension)
- numpy

Environment controls:
- PYCHROME_ENABLE_NATIVE=0 disables native backends and forces NumPy path.
- PYCHROME_BACKEND=<auto|numpy|cpu_ext|metal|cupy|gpu> forces request globally.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def _env_flag_true(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


_NATIVE_ENABLED = _env_flag_true("PYCHROME_ENABLE_NATIVE", True)

_CPU_CORE_IMPORT_ERROR: Optional[Exception] = None
try:
    if _NATIVE_ENABLED:
        from . import _cpu_core  # type: ignore
    else:
        _cpu_core = None
except Exception as exc:  # pragma: no cover
    _cpu_core = None
    _CPU_CORE_IMPORT_ERROR = exc

_CUPY_IMPORT_ERROR: Optional[Exception] = None
try:
    if _NATIVE_ENABLED:
        import cupy as cp  # type: ignore
    else:
        cp = None
except Exception as exc:  # pragma: no cover
    cp = None
    _CUPY_IMPORT_ERROR = exc


def clamp(x):
    return np.clip(x, 0.0, 1.0)


def kelvin_to_rgb(kelvin: float) -> np.ndarray:
    kelvin = float(np.clip(kelvin, 1000.0, 40000.0))
    tmp = kelvin / 100.0
    if tmp <= 66.0:
        red = 255.0
    else:
        red = 329.698727446 * ((tmp - 60.0) ** -0.1332047592)
    if tmp <= 66.0:
        green = 99.4708025861 * math.log(tmp) - 161.1195681661
    else:
        green = 288.1221695283 * ((tmp - 60.0) ** -0.0755148492)
    if tmp >= 66.0:
        blue = 255.0
    elif tmp <= 19.0:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(tmp - 10.0) - 305.0447927307

    def _cl(v: float) -> float:
        return float(np.clip(v, 0.0, 255.0) / 255.0)

    return np.array([_cl(red), _cl(green), _cl(blue)], dtype=np.float32)


NEUTRAL_KELVIN = 6500.0


def apply_white_balance_numpy(img: np.ndarray, temp_kelvin: float, tint_value: float) -> np.ndarray:
    if img is None:
        return None
    src_rgb = kelvin_to_rgb(temp_kelvin)
    ref_rgb = kelvin_to_rgb(NEUTRAL_KELVIN)
    eps = 1e-8
    gains = ref_rgb / (src_rgb + eps)
    tint_norm = float(np.clip(tint_value, -100.0, 100.0)) / 100.0
    gains[1] *= 1.0 - 0.15 * tint_norm
    out = np.asarray(img, dtype=np.float32) * gains.reshape((1, 1, 3))
    return clamp(out)


def scientific_irg_transform_numpy(
    img: np.ndarray,
    fracRx: float,
    fracGx: float,
    fracBy: float,
    gammaRx: float,
    gammaRy: float,
    gammaGx: float,
    gammaGy: float,
    gammaBY: float,
    exposure: float,
) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    Z1 = arr[:, :, 0]
    Z2 = arr[:, :, 1]
    Z3 = arr[:, :, 2]

    eps = 1e-6
    fracBy = max(float(fracBy), eps)
    fracRx = max(float(fracRx), eps)
    fracGx = max(float(fracGx), eps)
    fracRy = 1.0 - fracRx
    fracGy = 1.0 - fracGx

    innerY = np.clip(1.0 - (Z3 / fracBy), 0.0, 1.0)
    Y = 1.0 - innerY ** (1.0 / float(gammaBY))

    tmp1 = (1.0 - Y) ** float(gammaRy)
    termR = fracRy * (1.0 - tmp1)
    innerX1 = np.clip(1.0 - ((Z1 - termR) / fracRx), 0.0, 1.0)
    X1 = 1.0 - innerX1 ** (1.0 / float(gammaRx))

    tmp2 = (1.0 - Y) ** float(gammaGy)
    termG = fracGy * (1.0 - tmp2)
    innerX2 = np.clip(1.0 - ((Z2 - termG) / fracGx), 0.0, 1.0)
    X2 = 1.0 - innerX2 ** (1.0 / float(gammaGx))

    out = np.dstack([np.clip(Y, 0.0, 1.0), np.clip(X1, 0.0, 1.0), np.clip(X2, 0.0, 1.0)])
    return clamp(out * float(exposure))


def params_from_preset(preset: Dict) -> Dict[str, float]:
    return {
        "wb_temp": float(preset.get("wb_temp", 6500.0)),
        "wb_tint": float(preset.get("wb_tint", 0.0)),
        "fracRx": float(preset.get("fracRx", 0.7)),
        "fracGx": float(preset.get("fracGx", 0.7)),
        "fracBY": float(preset.get("fracBY", 1.0)),
        "gammaRx": float(preset.get("gammaRx", 1.0)),
        "gammaRy": float(preset.get("gammaRy", 1.0)),
        "gammaGx": float(preset.get("gammaGx", 1.0)),
        "gammaGy": float(preset.get("gammaGy", 1.0)),
        "gammaBY": float(preset.get("gammaBY", 1.0)),
        "exposure": float(preset.get("exposure", 1.0)),
    }


def _params_tuple(params: Dict[str, float]) -> Tuple[float, ...]:
    return (
        float(params["wb_temp"]),
        float(params["wb_tint"]),
        float(params["fracRx"]),
        float(params["fracGx"]),
        float(params["fracBY"]),
        float(params["gammaRx"]),
        float(params["gammaRy"]),
        float(params["gammaGx"]),
        float(params["gammaGy"]),
        float(params["gammaBY"]),
        float(params["exposure"]),
    )


def normalize_backend_request(backend: Optional[str]) -> str:
    if backend is None:
        return "auto"
    b = str(backend).strip().lower()
    if b in {"cpu", "c", "cpuext", "cpu-extension"}:
        return "cpu_ext"
    if b in {"gpu", "metal-gpu"}:
        return "gpu"
    if b in {"auto", "numpy", "cpu_ext", "metal", "cupy", "gpu"}:
        return b
    return "auto"


def _effective_backend_request(requested: Optional[str]) -> str:
    forced = os.getenv("PYCHROME_BACKEND")
    if forced:
        return normalize_backend_request(forced)
    return normalize_backend_request(requested)


def cpu_extension_available() -> bool:
    return _NATIVE_ENABLED and (_cpu_core is not None)


def cupy_backend_available() -> bool:
    return _NATIVE_ENABLED and (cp is not None)


_METAL_PROBE_LOCK = threading.Lock()
_METAL_PROBED = False
_METAL_AVAILABLE = False


def metal_backend_available() -> bool:
    global _METAL_PROBED, _METAL_AVAILABLE
    if not _NATIVE_ENABLED:
        return False
    if _METAL_PROBED:
        return _METAL_AVAILABLE
    with _METAL_PROBE_LOCK:
        if _METAL_PROBED:
            return _METAL_AVAILABLE
        try:
            from .metal_backend import metal_backend_runtime_available

            _METAL_AVAILABLE = bool(metal_backend_runtime_available())
        except Exception:
            _METAL_AVAILABLE = False
        _METAL_PROBED = True
        return _METAL_AVAILABLE


class FrameProcessor:
    """Unified frame processor with backend selection and auto fallback."""

    def __init__(self, requested_backend: str = "auto"):
        self.requested_backend = _effective_backend_request(requested_backend)
        self.active_backend = "numpy"
        self.last_error = None
        self._degraded = set()
        self._metal = None
        self._last_metal_shape = None
        self._last_metal_params = None

    def close(self):
        if self._metal is not None:
            try:
                self._metal.close()
            except Exception:
                pass
        self._metal = None
        self._last_metal_shape = None
        self._last_metal_params = None

    def _candidate_order(self) -> Iterable[str]:
        req = self.requested_backend
        if req == "metal":
            order = ["metal", "cupy", "cpu_ext", "numpy"]
        elif req == "cupy":
            order = ["cupy", "cpu_ext", "numpy"]
        elif req == "gpu":
            order = ["metal", "cupy", "cpu_ext", "numpy"]
        elif req == "cpu_ext":
            order = ["cpu_ext", "numpy"]
        elif req == "numpy":
            order = ["numpy"]
        else:
            order = ["metal", "cupy", "cpu_ext", "numpy"]
        for name in order:
            if name in self._degraded:
                continue
            yield name
        if "numpy" not in self._degraded and "numpy" not in order:
            yield "numpy"

    def _prepare_input(self, rgb: np.ndarray) -> np.ndarray:
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("Expected float32 frame with shape [H, W, 3].")
        arr = arr[:, :, :3]
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        return arr

    def _process_numpy(self, arr: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        wb = apply_white_balance_numpy(arr, params["wb_temp"], params["wb_tint"])
        return scientific_irg_transform_numpy(
            wb,
            params["fracRx"],
            params["fracGx"],
            params["fracBY"],
            params["gammaRx"],
            params["gammaRy"],
            params["gammaGx"],
            params["gammaGy"],
            params["gammaBY"],
            params["exposure"],
        )

    def _process_cpu_ext(self, arr: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        if _cpu_core is None:
            raise RuntimeError("CPU extension unavailable")
        return _cpu_core.process_frame_float32(arr, *_params_tuple(params))

    def _ensure_metal(self, shape_hw: Tuple[int, int]):
        h, w = shape_hw
        if self._metal is None:
            from .metal_backend import MetalBackend

            self._metal = MetalBackend()
            self._metal.start(width=w, height=h)
            self._last_metal_shape = (h, w)
            self._last_metal_params = None
            return
        if self._last_metal_shape != (h, w):
            self._metal.ensure_dimensions(width=w, height=h)
            self._last_metal_shape = (h, w)

    def _process_metal(self, arr: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        if not metal_backend_available():
            raise RuntimeError("Metal backend unavailable")
        h, w = int(arr.shape[0]), int(arr.shape[1])
        self._ensure_metal((h, w))
        pt = _params_tuple(params)
        if self._last_metal_params != pt:
            self._metal.set_params(
                wb_temp=params["wb_temp"],
                wb_tint=params["wb_tint"],
                fracRx=params["fracRx"],
                fracGx=params["fracGx"],
                fracBY=params["fracBY"],
                gammaRx=params["gammaRx"],
                gammaRy=params["gammaRy"],
                gammaGx=params["gammaGx"],
                gammaGy=params["gammaGy"],
                gammaBY=params["gammaBY"],
                exposure=params["exposure"],
            )
            self._last_metal_params = pt
        return self._metal.process_frame(arr)

    def _process_cupy(self, arr: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        if cp is None:
            raise RuntimeError("CuPy backend unavailable")

        arr_cp = cp.asarray(np.asarray(arr, dtype=np.float32), dtype=cp.float32)
        src_rgb = kelvin_to_rgb(params["wb_temp"])
        ref_rgb = kelvin_to_rgb(NEUTRAL_KELVIN)
        eps = 1e-8
        gains = ref_rgb / (src_rgb + eps)
        tint_norm = float(np.clip(params["wb_tint"], -100.0, 100.0)) / 100.0
        gains[1] *= (1.0 - 0.15 * tint_norm)
        gains_cp = cp.asarray(gains.reshape((1, 1, 3)), dtype=cp.float32)
        wb = cp.clip(arr_cp * gains_cp, 0.0, 1.0)

        Z1 = wb[:, :, 0]
        Z2 = wb[:, :, 1]
        Z3 = wb[:, :, 2]
        eps2 = 1e-6
        fracBy = max(float(params["fracBY"]), eps2)
        fracRx = max(float(params["fracRx"]), eps2)
        fracGx = max(float(params["fracGx"]), eps2)
        fracRy = 1.0 - fracRx
        fracGy = 1.0 - fracGx

        gammaBy = max(float(params["gammaBY"]), eps2)
        gammaRx = max(float(params["gammaRx"]), eps2)
        gammaRy = max(float(params["gammaRy"]), eps2)
        gammaGx = max(float(params["gammaGx"]), eps2)
        gammaGy = max(float(params["gammaGy"]), eps2)
        exposure = float(params["exposure"])

        innerY = cp.clip(1.0 - (Z3 / fracBy), 0.0, 1.0)
        Y = 1.0 - innerY ** (1.0 / gammaBy)
        tmp1 = (1.0 - Y) ** gammaRy
        termR = fracRy * (1.0 - tmp1)
        innerX1 = cp.clip(1.0 - ((Z1 - termR) / fracRx), 0.0, 1.0)
        X1 = 1.0 - innerX1 ** (1.0 / gammaRx)
        tmp2 = (1.0 - Y) ** gammaGy
        termG = fracGy * (1.0 - tmp2)
        innerX2 = cp.clip(1.0 - ((Z2 - termG) / fracGx), 0.0, 1.0)
        X2 = 1.0 - innerX2 ** (1.0 / gammaGx)
        out = cp.clip(
            cp.stack([cp.clip(Y, 0.0, 1.0), cp.clip(X1, 0.0, 1.0), cp.clip(X2, 0.0, 1.0)], axis=2) * exposure,
            0.0,
            1.0,
        )
        return cp.asnumpy(out)

    def process(self, rgb: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        arr = self._prepare_input(rgb)
        last_exc = None
        for backend in self._candidate_order():
            try:
                if backend == "metal":
                    out = self._process_metal(arr, params)
                elif backend == "cupy":
                    out = self._process_cupy(arr, params)
                elif backend == "cpu_ext":
                    out = self._process_cpu_ext(arr, params)
                else:
                    out = self._process_numpy(arr, params)
                self.active_backend = backend
                self.last_error = None
                return np.asarray(out, dtype=np.float32)
            except Exception as exc:
                last_exc = exc
                if backend != "numpy":
                    self._degraded.add(backend)
                    self.last_error = f"{backend}: {exc}"
                    if backend == "metal" and self._metal is not None:
                        try:
                            self._metal.close()
                        except Exception:
                            pass
                        self._metal = None
                        self._last_metal_shape = None
                        self._last_metal_params = None
                    continue
                raise

        out = self._process_numpy(arr, params)
        self.active_backend = "numpy"
        self.last_error = str(last_exc) if last_exc else None
        return np.asarray(out, dtype=np.float32)

    def process_frame_float32(
        self,
        rgb: np.ndarray,
        wb_temp: float,
        wb_tint: float,
        fracRx: float,
        fracGx: float,
        fracBY: float,
        gammaRx: float,
        gammaRy: float,
        gammaGx: float,
        gammaGy: float,
        gammaBY: float,
        exposure: float,
    ) -> np.ndarray:
        params = {
            "wb_temp": float(wb_temp),
            "wb_tint": float(wb_tint),
            "fracRx": float(fracRx),
            "fracGx": float(fracGx),
            "fracBY": float(fracBY),
            "gammaRx": float(gammaRx),
            "gammaRy": float(gammaRy),
            "gammaGx": float(gammaGx),
            "gammaGy": float(gammaGy),
            "gammaBY": float(gammaBY),
            "exposure": float(exposure),
        }
        return self.process(rgb, params)

    def process_with_preset(self, rgb: np.ndarray, preset: Dict) -> np.ndarray:
        return self.process(rgb, params_from_preset(preset))
