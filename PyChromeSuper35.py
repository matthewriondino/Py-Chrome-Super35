#!/usr/bin/env python3
"""
PyChromeSuper35.py — Video-only Py-Chrome GUI with Queue / Batch support

- Adds a job queue (Add to Queue / Remove / Move Up / Move Down)
- Process Queue -> writes batch.json and spawns convert_clip.py --batch <batch.json>
- Conversion modal streams converter stdout (progress + log). Supports JOB_START/TOTAL_FRAMES/PROCESSED/JOB_DONE lines.
- Timeline scrubber remains under main preview (right panel).
- Requires: dearpygui, numpy, pillow, opencv-python (cv2)
"""

import os
import sys
import math
import json
import tempfile
import threading
import subprocess
import shutil
import queue
import re
import time
import uuid
from collections import deque

from pathlib import Path
import numpy as np
from PIL import Image
import dearpygui.dearpygui as dpg

from pcv_color import (
    NEUTRAL_KELVIN,
    apply_white_balance,
    clamp,
    find_wb_from_gray_sample,
    kelvin_to_rgb,
    scientific_irg_transform as scientific_irg_transform_core,
)
from pcv_lut import apply_lut_preview as apply_lut_preview_core
from pcv_lut import parse_cube_lut_file
from pcv_scopes import (
    build_histogram_texture as build_histogram_texture_core,
    build_waveform_texture as build_waveform_texture_core,
    compute_vectorscope_series,
    sample_points as scope_sample_points,
)

# OpenCV for video preview / frame reading
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

# Optional native backend package (Metal + CPU extension with NumPy fallback)
try:
    from pychrome_native import (
        FrameProcessor,
        cpu_extension_available,
        cupy_backend_available,
        metal_backend_available,
    )
    NATIVE_BACKEND_IMPORT_ERROR = None
except Exception as e:
    FrameProcessor = None
    NATIVE_BACKEND_IMPORT_ERROR = e
    cpu_extension_available = lambda: False
    cupy_backend_available = lambda: False
    metal_backend_available = lambda: False

# ----------------------------
# Constants / Globals
# ----------------------------
MAX_PREVIEW_W, MAX_PREVIEW_H = 800, 500
MAX_CH_W, MAX_CH_H = MAX_PREVIEW_W // 4, MAX_PREVIEW_H // 4
HIST_W, HIST_H = 256, 140
WAVE_W, WAVE_H = 420, 240
PREVIEW_PROC_W, PREVIEW_PROC_H = 1280, 720
PREVIEW_QUALITY_MODES = ("Fast", "Balanced", "Full")
COMPUTE_BACKEND_MODES = ("Auto", "CPU", "GPU")
PREVIEW_BIT_DEPTH_MODES = ("8-bit", "10-bit")
CHANNEL_PREVIEWS = ["Original", "IR", "R", "G"]
SUPPORTED_VIDEO_EXTENSIONS = (
    ".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm",
    ".wmv", ".mpg", ".mpeg", ".mxf", ".mts", ".m2ts"
)

preview_img = None          # currently displayed frame (HxWx3 float [0..1])
preview_work_img = None     # downscaled working frame for responsive GUI processing

video_capture = None
video_path = None
video_frame_count = 0
video_fps = 0.0
video_width = 0
video_height = 0
_preview_ffmpeg_path = None
_preview_10bit_warned_once = False

def _app_user_data_dir():
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "PyChromeSuper35"
    if os.name == "nt":
        base = Path(os.getenv("APPDATA") or str(home))
        return base / "PyChromeSuper35"
    return home / ".pychromesuper35"


def _bundled_presets_dir():
    candidates = [Path(__file__).resolve().parent / "presets"]
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "presets")
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend(
            [
                exe_dir / "presets",
                exe_dir.parent / "Resources" / "presets",
                exe_dir.parent.parent / "Resources" / "presets",
            ]
        )
    for c in candidates:
        try:
            if c.exists() and c.is_dir():
                return c
        except Exception:
            pass
    return None


def _resolve_presets_dir():
    if getattr(sys, "frozen", False):
        return _app_user_data_dir() / "presets"
    return Path(__file__).resolve().parent / "presets"


PRESETS_DIR = str(_resolve_presets_dir())
try:
    os.makedirs(PRESETS_DIR, exist_ok=True)
except Exception:
    fallback = _app_user_data_dir() / "presets"
    os.makedirs(fallback, exist_ok=True)
    PRESETS_DIR = str(fallback)


def _seed_presets_if_empty():
    if not getattr(sys, "frozen", False):
        return
    try:
        has_files = any(Path(PRESETS_DIR).iterdir())
    except Exception:
        has_files = True
    if has_files:
        return
    src = _bundled_presets_dir()
    if src is None:
        return
    try:
        for p in src.glob("*.json"):
            shutil.copy2(p, Path(PRESETS_DIR) / p.name)
    except Exception:
        pass


_seed_presets_if_empty()

# default preset values
DEFAULT_PRESET = {
    "wb_temp": 6500,
    "wb_tint": 0,
    "fracRx": 0.7,
    "fracGx": 0.7,
    "fracBY": 1.0,
    "gammaRx": 1.0,
    "gammaRy": 1.0,
    "gammaGx": 1.0,
    "gammaGy": 1.0,
    "gammaBY": 1.0,
    "exposure": 1.0
}
PRESET_SLIDERS = list(DEFAULT_PRESET.keys())
DEFAULT_MARKER_SIZE = 2
DEFAULT_SCOPE_POINT_SIZE = 1
MAX_SCOPE_POINTS = 24000
SCOPE_MODES = ("Histogram", "Waveform", "Vectorscope")
_scope_mode = "Histogram"
wb_dropper_active = False

# Theme ids for dynamic point sizes/colors
_SCATTER_THEME_IDS = {"rg": None, "rb": None, "gb": None}
_SCOPE_THEME_IDS = {
    "vec_r": None, "vec_y": None, "vec_g": None, "vec_c": None, "vec_b": None, "vec_m": None,
}

_SCATTER_CACHE_N = None
_SCATTER_CACHE_CAP = None
_SCATTER_CACHE_IDX = None
_SCATTER_RNG = np.random.default_rng()

# queue of jobs (in-memory)
queue_items = []  # each item: dict {id,label,input,output,preset}

# converter communication
_conv_queue = None
_conv_raw_queue = None
_convert_thread = None
_conv_worker_thread = None
_convert_proc = None
_convert_tmpdir = None
_convert_cancel_requested = False
_conv_state_lock = threading.Lock()
_conv_state = None
_conv_state_rev = 0
_conv_ui_applied_rev = -1
_conv_ui_last_log_push_ts = 0.0
_conv_ui_heartbeat_running = False
_conv_success_autoclose_queued = False
_conv_status_pipeline = str(os.getenv("PYCHROME_STATUS_PIPELINE", "v2")).strip().lower()
if _conv_status_pipeline not in {"v1", "v2"}:
    _conv_status_pipeline = "v2"
_conv_status_debug = str(os.getenv("PYCHROME_STATUS_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}

# per-batch tracking (for modal)
_conv_current_job_index = None
_conv_current_job_total = None
_conv_current_job_processed = None
_conv_total_jobs = 0
_conv_done_jobs = 0
_conv_current_job_stage = 0.0
_conv_duration_ms = None
_conv_job_frame_frac = 0.0
_conv_job_encode_frac = 0.0
_conv_last_status_update = 0.0
_conv_log_lines = deque(maxlen=400)
_conv_log_last_ui_push = 0.0
_conv_poll_frame_step = 1
_conv_poll_max_items = 240
_conv_log_processed_step = 20
_conv_last_logged_processed = 0
_conv_poll_last_ts = 0.0
_conv_render_poll_min_interval_s = 0.02
_conv_use_render_tick = False
_conv_render_heartbeat_min_interval_s = 0.06
_conv_render_last_heartbeat_ts = 0.0
_ui_heartbeat_frame_step = 2
_ui_heartbeat_stale_poll_s = 0.20
_conv_stall_warn_after_s = 1.0
_conv_status_log_max_lines = 300
_conv_status_log_every_frames = 20
_conv_status_log_min_interval_s = 0.5
_conv_status_log_ui_interval_s = 0.35
_conv_raw_queue_maxsize = 4096

# Preview redraw coalescing
_preview_update_scheduled = False
_preview_update_dirty = False
_preview_last_update_ts = 0.0
_preview_min_interval_s = 1.0 / 30.0
_preview_min_interval_during_convert_s = 1.0 / 10.0
_last_warning_text = None
_preview_processor = None
_preview_processor_request = None
_preview_backend_active = "numpy"
_scope_last_update_ts = 0.0
_scatter_last_update_ts = 0.0
_scope_min_interval_idle_s = 1.0 / 20.0
_scatter_min_interval_idle_s = 1.0 / 15.0
_scope_min_interval_during_convert_s = 0.35
_scatter_min_interval_during_convert_s = 0.35
_channel_min_interval_idle_s = 1.0 / 12.0
_channel_min_interval_during_convert_s = 0.30
_channel_last_update_ts = 0.0
_scope_input_max_width = 640
_scope_input_max_height = 360

# LUT preview state (non-destructive GUI preview only)
_lut_title = ""
_lut_path = None
_lut_size = 0
_lut_flat = None
_lut_domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
_lut_domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

# ----------------------------
# Utilities
# ----------------------------
def _as_texture_value(rgba):
    return np.ascontiguousarray(rgba, dtype=np.float32).reshape(-1)

def _extract_path_from_file_dialog_appdata(app_data):
    """handy helper to extract selected path from DPG file dialog app_data"""
    if not app_data:
        return None
    if isinstance(app_data, dict):
        file_path = app_data.get("file_path_name") or app_data.get("file_path") or app_data.get("file_name")
        if isinstance(file_path, (list, tuple)) and file_path:
            return file_path[0]
        return file_path
    if isinstance(app_data, (list, tuple)) and app_data:
        return app_data[0]
    if isinstance(app_data, str):
        return app_data
    return None

def toggle_wb_dropper(sender=None, app_data=None):
    global wb_dropper_active
    wb_dropper_active = not wb_dropper_active
    if wb_dropper_active:
        dpg.configure_item("wb_dropper_btn", label="Click preview to set WB...")
        print("WB Dropper active — click on a neutral area in main preview.")
    else:
        dpg.configure_item("wb_dropper_btn", label="Set WB Reference")
        print("WB Dropper deactivated.")

# ----------------------------
# Aspect-correct texture rendering (DearPyGui textures expect flattened RGBA floats)
# ----------------------------
def render_into_texture(img, tex_w, tex_h, mono=False):
    if img is None:
        return np.zeros((tex_h, tex_w, 4), dtype=np.float32)
    if img.ndim == 2:
        h, w = img.shape
        src_rgb = np.stack([img, img, img], axis=2)
    else:
        h, w = img.shape[:2]
        src_rgb = img
    scale = min(tex_w / w, tex_h / h)
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))
    if (h == target_h and w == target_w):
        resized = np.asarray(src_rgb, dtype=np.float32)
    else:
        if CV2_AVAILABLE:
            interp = cv2.INTER_AREA if (target_w < w or target_h < h) else cv2.INTER_LINEAR
            resized = cv2.resize(np.asarray(src_rgb, dtype=np.float32), (target_w, target_h), interpolation=interp)
            if resized.ndim == 2:
                resized = np.stack([resized, resized, resized], axis=2)
        else:
            pil = Image.fromarray((src_rgb * 255.0).astype(np.uint8))
            pil = pil.resize((target_w, target_h), Image.LANCZOS)
            resized = np.asarray(pil, dtype=np.float32) / 255.0
    canvas = np.zeros((tex_h, tex_w, 4), dtype=np.float32)
    x0 = (tex_w - target_w) // 2
    y0 = (tex_h - target_h) // 2
    canvas[y0:y0+target_h, x0:x0+target_w, :3] = resized
    canvas[y0:y0+target_h, x0:x0+target_w, 3] = 1.0
    return canvas

def _preview_quality_mode():
    try:
        if dpg.does_item_exist("preview_quality"):
            val = dpg.get_value("preview_quality")
            if val in PREVIEW_QUALITY_MODES:
                return val
    except Exception:
        pass
    return "Balanced"

def _preview_bit_depth_mode():
    try:
        if dpg.does_item_exist("preview_bit_depth"):
            val = dpg.get_value("preview_bit_depth")
            if val in PREVIEW_BIT_DEPTH_MODES:
                return val
    except Exception:
        pass
    return "8-bit"

def _preview_target_dimensions():
    mode = _preview_quality_mode()
    if _is_conversion_active():
        if mode == "Fast":
            return 640, 360
        if mode == "Full":
            return 960, 540
        return 800, 450
    if mode == "Fast":
        return 960, 540
    if mode == "Full":
        return None, None
    return PREVIEW_PROC_W, PREVIEW_PROC_H

def _is_conversion_active():
    try:
        if _conv_status_pipeline == "v2":
            with _conv_state_lock:
                st = _conv_state
                if st is not None and bool(st.get("active")) and not bool(st.get("done")):
                    return True
    except Exception:
        pass
    try:
        return (_convert_thread is not None) and _convert_thread.is_alive()
    except Exception:
        return False


def _preview_update_interval_s():
    if _is_conversion_active():
        return max(float(_preview_min_interval_s), float(_preview_min_interval_during_convert_s))
    return float(_preview_min_interval_s)


def rebuild_preview_work_image():
    global preview_work_img
    if preview_img is None:
        preview_work_img = None
        return
    src = np.asarray(preview_img, dtype=np.float32)
    if src.ndim != 3 or src.shape[2] < 3:
        preview_work_img = src
        return
    target = _preview_target_dimensions()
    if target == (None, None):
        preview_work_img = src
        return
    target_w, target_h = target
    h, w = src.shape[:2]
    scale = min(1.0, float(target_w) / float(max(1, w)), float(target_h) / float(max(1, h)))
    if scale >= 0.999:
        preview_work_img = src
        return
    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale)))
    try:
        preview_work_img = cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32, copy=False)
    except Exception:
        preview_work_img = src

# ----------------------------
# Scientific IR->R/G transform (unchanged core math)
# ----------------------------
def scientific_irg_transform(img):
    return scientific_irg_transform_core(
        img,
        fracRx=dpg.get_value("fracRx"),
        fracGx=dpg.get_value("fracGx"),
        fracBy=dpg.get_value("fracBY"),
        gammaRx=dpg.get_value("gammaRx"),
        gammaRy=dpg.get_value("gammaRy"),
        gammaGx=dpg.get_value("gammaGx"),
        gammaGy=dpg.get_value("gammaGy"),
        gammaBy=dpg.get_value("gammaBY"),
        exposure=dpg.get_value("exposure"),
    )

def _compute_backend_mode():
    try:
        if dpg.does_item_exist("compute_backend"):
            val = dpg.get_value("compute_backend")
            if val in COMPUTE_BACKEND_MODES:
                return val
    except Exception:
        pass
    return "Auto"

def _backend_request_from_mode(mode):
    if mode == "CPU":
        return "cpu_ext"
    if mode == "GPU":
        return "gpu"
    return "auto"

def _backend_label(name):
    if name == "metal":
        return "GPU (Metal)"
    if name == "cupy":
        return "GPU (CuPy)"
    if name == "cpu_ext":
        return "CPU (Native)"
    return "CPU (NumPy)"

def _reset_preview_processor():
    global _preview_processor, _preview_processor_request, _preview_backend_active
    if _preview_processor is not None:
        try:
            _preview_processor.close()
        except Exception:
            pass
    _preview_processor = None
    _preview_processor_request = None
    _preview_backend_active = "numpy"

def _ensure_preview_processor():
    global _preview_processor, _preview_processor_request
    if FrameProcessor is None:
        return None
    requested = _backend_request_from_mode(_compute_backend_mode())
    if _preview_processor is None or _preview_processor_request != requested:
        _reset_preview_processor()
        _preview_processor_request = requested
        try:
            _preview_processor = FrameProcessor(requested_backend=requested)
        except Exception as e:
            _preview_processor = None
            print("Native preview backend unavailable; using NumPy:", e)
    return _preview_processor

def _set_compute_status():
    if not dpg.does_item_exist("compute_status_text"):
        return
    mode = _compute_backend_mode()
    if FrameProcessor is None:
        msg = "Compute: CPU (NumPy fallback)"
    elif mode == "GPU":
        if _preview_backend_active in {"metal", "cupy"}:
            msg = f"Compute: {_backend_label(_preview_backend_active)}"
        else:
            if metal_backend_available() or cupy_backend_available():
                msg = f"Compute: GPU requested, fallback {_backend_label(_preview_backend_active)}"
            else:
                msg = f"Compute: GPU requested, no GPU backend available ({_backend_label(_preview_backend_active)})"
    elif mode == "CPU":
        if _preview_backend_active == "cpu_ext":
            msg = "Compute: CPU (Native)"
        elif cpu_extension_available():
            msg = "Compute: CPU requested, fallback CPU (NumPy)"
        else:
            msg = "Compute: CPU (NumPy)"
    else:
        if _preview_backend_active == "metal":
            msg = "Compute: Auto (GPU Metal)"
        elif _preview_backend_active == "cupy":
            msg = "Compute: Auto (GPU CuPy)"
        elif _preview_backend_active == "cpu_ext":
            msg = "Compute: Auto (CPU Native)"
        else:
            msg = "Compute: Auto (CPU NumPy)"
    dpg.set_value("compute_status_text", msg)

def _current_preview_params(temp, tint):
    return {
        "wb_temp": float(temp),
        "wb_tint": float(tint),
        "fracRx": float(dpg.get_value("fracRx")),
        "fracGx": float(dpg.get_value("fracGx")),
        "fracBY": float(dpg.get_value("fracBY")),
        "gammaRx": float(dpg.get_value("gammaRx")),
        "gammaRy": float(dpg.get_value("gammaRy")),
        "gammaGx": float(dpg.get_value("gammaGx")),
        "gammaGy": float(dpg.get_value("gammaGy")),
        "gammaBY": float(dpg.get_value("gammaBY")),
        "exposure": float(dpg.get_value("exposure")),
    }

def compute_preview_pipeline(img, temp, tint):
    global _preview_backend_active
    wb_preview = apply_white_balance(img, temp, tint)
    processor = _ensure_preview_processor()
    if processor is None:
        _preview_backend_active = "numpy"
        _set_compute_status()
        return wb_preview, scientific_irg_transform(wb_preview)
    try:
        out = processor.process(np.asarray(img, dtype=np.float32), _current_preview_params(temp, tint))
        _preview_backend_active = getattr(processor, "active_backend", "numpy")
    except Exception as e:
        print("Preview compute backend failed; falling back to NumPy:", e)
        _preview_backend_active = "numpy"
        _reset_preview_processor()
        out = scientific_irg_transform(wb_preview)
    _set_compute_status()
    return wb_preview, np.asarray(out, dtype=np.float32)

def on_preview_quality_changed(sender=None, app_data=None):
    global _scope_last_update_ts, _scatter_last_update_ts, _channel_last_update_ts
    rebuild_preview_work_image()
    _scope_last_update_ts = 0.0
    _scatter_last_update_ts = 0.0
    _channel_last_update_ts = 0.0
    # Force one immediate draw on clip load so main/after previews appear without needing a scrub event.
    update_main_preview()

def on_compute_backend_changed(sender=None, app_data=None):
    global _scope_last_update_ts, _scatter_last_update_ts, _channel_last_update_ts
    _reset_preview_processor()
    _set_compute_status()
    _scope_last_update_ts = 0.0
    _scatter_last_update_ts = 0.0
    _channel_last_update_ts = 0.0
    update_main_preview()

def _set_preview_frame(frame):
    global preview_img
    if frame is None:
        return
    preview_img = np.asarray(frame, dtype=np.float32)
    rebuild_preview_work_image()
    try:
        rgba_full = render_into_texture(preview_img, MAX_CH_W, MAX_CH_H)
        dpg.set_value("tex_before_Original", _as_texture_value(rgba_full))
        dpg.set_value("tex_before_IR", _as_texture_value(render_into_texture(preview_img[:, :, 2], MAX_CH_W, MAX_CH_H, mono=True)))
        dpg.set_value("tex_before_R", _as_texture_value(render_into_texture(preview_img[:, :, 0], MAX_CH_W, MAX_CH_H, mono=True)))
        dpg.set_value("tex_before_G", _as_texture_value(render_into_texture(preview_img[:, :, 1], MAX_CH_W, MAX_CH_H, mono=True)))
    except Exception:
        pass

def on_preview_bit_depth_changed(sender=None, app_data=None):
    global _scope_last_update_ts, _scatter_last_update_ts, _channel_last_update_ts
    idx = 0
    try:
        if dpg.does_item_exist("timeline_slider"):
            idx = int(dpg.get_value("timeline_slider") or 0)
    except Exception:
        idx = 0
    frame = read_frame_from_capture(idx)
    if frame is not None:
        _set_preview_frame(frame)
        _scope_last_update_ts = 0.0
        _scatter_last_update_ts = 0.0
        _channel_last_update_ts = 0.0
        update_main_preview()
    else:
        request_preview_update()

# ----------------------------
# Warnings & channel previews
# ----------------------------
def update_warnings():
    global _last_warning_text
    warnings = []
    if dpg.get_value("fracBY") < 1.0:
        warnings.append("⚠ Blue fraction < 1 → highlights may clip")
    if dpg.get_value("exposure") > 2.0:
        warnings.append("⚠ Exposure high → risk of clipping")
    text = "\n".join(warnings)
    if text == _last_warning_text:
        return
    _last_warning_text = text
    dpg.configure_item("warning_text", default_value=text)

def update_channel_previews(out):
    if preview_img is None:
        return
    for ch in CHANNEL_PREVIEWS:
        if ch == "Original":
            try:
                unswapped_display = np.dstack([
                    out[:, :, 1],  # Red = recovered red
                    out[:, :, 2],  # Green = recovered green
                    out[:, :, 0],  # Blue = IR
                ])
                rgba = render_into_texture(unswapped_display, MAX_CH_W, MAX_CH_H)
            except Exception:
                rgba = render_into_texture(out, MAX_CH_W, MAX_CH_H)
        elif ch == "IR":
            rgba = render_into_texture(out[:, :, 0], MAX_CH_W, MAX_CH_H, mono=True)
        elif ch == "R":
            rgba = render_into_texture(out[:, :, 1], MAX_CH_W, MAX_CH_H, mono=True)
        elif ch == "G":
            rgba = render_into_texture(out[:, :, 2], MAX_CH_W, MAX_CH_H, mono=True)
        else:
            rgba = np.zeros((MAX_CH_H, MAX_CH_W, 4), dtype=np.float32)
        dpg.set_value(f"tex_{ch}", _as_texture_value(rgba))

# ----------------------------
# Analysis graphs (Scatter + Histogram/Waveform/Vectorscope)
# ----------------------------
def _delete_scatter_themes():
    for k in list(_SCATTER_THEME_IDS.keys()):
        tid = _SCATTER_THEME_IDS.get(k)
        if tid is not None:
            try:
                dpg.delete_item(tid)
            except Exception:
                pass
            _SCATTER_THEME_IDS[k] = None

def rebuild_scatter_themes(marker_size):
    _delete_scatter_themes()
    specs = [
        ("rg", "series_rg", (255, 255, 0, 210)),   # IR (red) + Red (green) -> yellow
        ("rb", "series_rb", (255, 0, 255, 210)),   # IR (red) + Green (blue) -> magenta
        ("gb", "series_gb", (0, 255, 255, 210)),   # Red (green) + Green (blue) -> cyan
    ]
    for key, series_tag, color in specs:
        try:
            with dpg.theme() as tid:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, int(marker_size), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
            _SCATTER_THEME_IDS[key] = tid
            dpg.bind_item_theme(series_tag, tid)
        except Exception:
            pass

def _delete_scope_themes():
    for k in list(_SCOPE_THEME_IDS.keys()):
        tid = _SCOPE_THEME_IDS.get(k)
        if tid is not None:
            try:
                dpg.delete_item(tid)
            except Exception:
                pass
            _SCOPE_THEME_IDS[k] = None

def rebuild_scope_point_themes(point_size):
    _delete_scope_themes()
    specs = [
        ("vec_r", "vectorscope_series_r", (255, 0, 0, 160)),
        ("vec_y", "vectorscope_series_y", (255, 255, 0, 160)),
        ("vec_g", "vectorscope_series_g", (0, 255, 0, 160)),
        ("vec_c", "vectorscope_series_c", (0, 255, 255, 160)),
        ("vec_b", "vectorscope_series_b", (0, 110, 255, 160)),
        ("vec_m", "vectorscope_series_m", (255, 0, 255, 160)),
    ]
    for key, series_tag, color in specs:
        try:
            with dpg.theme() as tid:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, int(point_size), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
            _SCOPE_THEME_IDS[key] = tid
            dpg.bind_item_theme(series_tag, tid)
        except Exception:
            pass

def on_marker_size_changed(sender, app_data):
    global _scatter_last_update_ts
    try:
        size = int(app_data)
    except Exception:
        size = dpg.get_value("scatter_marker_size")
    rebuild_scatter_themes(size)
    _scatter_last_update_ts = 0.0
    request_preview_update()

def on_scope_point_size_changed(sender, app_data):
    global _scope_last_update_ts
    try:
        size = int(app_data)
    except Exception:
        size = dpg.get_value("scope_point_size")
    rebuild_scope_point_themes(size)
    _scope_last_update_ts = 0.0
    request_preview_update()

def _apply_vectorscope_zoom():
    try:
        zoom2 = bool(dpg.get_value("vectorscope_x2"))
    except Exception:
        zoom2 = False
    lim = 64.0 if zoom2 else 128.0
    try:
        dpg.set_axis_limits("vectorscope_x_axis", -lim, lim)
        dpg.set_axis_limits("vectorscope_y_axis", -lim, lim)
    except Exception:
        pass

def on_vectorscope_zoom_changed(sender=None, app_data=None):
    global _scope_last_update_ts
    _apply_vectorscope_zoom()
    _scope_last_update_ts = 0.0
    request_preview_update()

def _lut_status_text():
    if _lut_flat is None or _lut_size <= 1:
        return "No LUT loaded."
    name = os.path.basename(_lut_path) if _lut_path else (_lut_title or "Untitled")
    return f"LUT loaded: {name} ({_lut_size}^3)"

def _set_lut_status(text=None):
    try:
        dpg.set_value("lut_status_text", text if text is not None else _lut_status_text())
    except Exception:
        pass

def clear_lut_preview(sender=None, app_data=None):
    global _lut_title, _lut_path, _lut_size, _lut_flat
    _lut_title = ""
    _lut_path = None
    _lut_size = 0
    _lut_flat = None
    _set_lut_status("No LUT loaded.")
    request_preview_update()

def _parse_cube_lut_file(path):
    return parse_cube_lut_file(path)

def load_lut_from_path(path):
    global _lut_title, _lut_path, _lut_size, _lut_flat, _lut_domain_min, _lut_domain_max
    if not path or not os.path.exists(path):
        _set_lut_status("LUT file not found.")
        return False
    try:
        title, size, flat, dmin, dmax = _parse_cube_lut_file(path)
        _lut_title = title or ""
        _lut_path = os.path.abspath(path)
        _lut_size = int(size)
        _lut_flat = flat
        _lut_domain_min = np.asarray(dmin, dtype=np.float32)
        _lut_domain_max = np.asarray(dmax, dtype=np.float32)
        _set_lut_status()
        print(f"Loaded LUT: {_lut_path} ({_lut_size}^3)")
        update_main_preview()
        return True
    except Exception as e:
        _set_lut_status(f"Failed to load LUT: {e}")
        print("Failed to load LUT:", e)
        return False

def open_lut_file_callback(sender, app_data):
    path = _extract_path_from_file_dialog_appdata(app_data)
    if not path:
        return
    load_lut_from_path(path)

def _lut_flat_lookup_indices(r, g, b, size):
    return (r + (g * size) + (b * size * size)).astype(np.int64)

def apply_lut_preview(img):
    try:
        enabled = bool(dpg.get_value("lut_enable"))
    except Exception:
        enabled = False
    return apply_lut_preview_core(
        img,
        enabled=enabled,
        lut_flat=_lut_flat,
        lut_size=_lut_size,
        domain_min=_lut_domain_min,
        domain_max=_lut_domain_max,
    )

def _analysis_use_converted():
    try:
        return bool(dpg.get_value("analysis_use_converted"))
    except Exception:
        return True

def _sample_points(xs, ys, limit=MAX_SCOPE_POINTS):
    return scope_sample_points(xs, ys, limit=limit)

def _scatter_source_image(wb_preview, out):
    use_converted = _analysis_use_converted()
    return (out if use_converted else wb_preview), use_converted

def _scope_source_image(wb_preview, out):
    use_converted = _analysis_use_converted()
    if use_converted:
        # Converted result channels are [IR, R, G] mapped to display [R,G,B].
        return out
    return wb_preview

def _downsample_scope_source(src_img):
    if src_img is None:
        return None
    arr = np.asarray(src_img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    h, w = arr.shape[:2]
    max_w = int(_scope_input_max_width)
    max_h = int(_scope_input_max_height)
    if w <= max_w and h <= max_h:
        return arr
    scale = min(float(max_w) / float(max(1, w)), float(max_h) / float(max(1, h)))
    if scale >= 0.999:
        return arr
    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale)))
    if CV2_AVAILABLE:
        try:
            return cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32, copy=False)
        except Exception:
            pass
    step_x = max(1, int(math.ceil(float(w) / float(max(1, tw)))))
    step_y = max(1, int(math.ceil(float(h) / float(max(1, th)))))
    return arr[::step_y, ::step_x, :]

def _update_scatterplots(src_img, use_converted):
    global _SCATTER_CACHE_N, _SCATTER_CACHE_CAP, _SCATTER_CACHE_IDX
    if src_img is None:
        dpg.set_value("series_rg", [[], []])
        dpg.set_value("series_rb", [[], []])
        dpg.set_value("series_gb", [[], []])
        return
    flat = src_img.reshape(-1, 3)
    n = flat.shape[0]
    if n <= 0:
        dpg.set_value("series_rg", [[], []])
        dpg.set_value("series_rb", [[], []])
        dpg.set_value("series_gb", [[], []])
        return
    cap = min(MAX_SCOPE_POINTS, n)
    if n > cap:
        if _SCATTER_CACHE_IDX is None or _SCATTER_CACHE_N != n or _SCATTER_CACHE_CAP != cap:
            _SCATTER_CACHE_IDX = _SCATTER_RNG.choice(n, cap, replace=False)
            _SCATTER_CACHE_N = n
            _SCATTER_CACHE_CAP = cap
        sample = flat[_SCATTER_CACHE_IDX]
    else:
        sample = flat
    if use_converted:
        ir = sample[:, 0]
        r = sample[:, 1]
        g = sample[:, 2]
    else:
        # Use input B channel as IR proxy on source view.
        r = sample[:, 0]
        g = sample[:, 1]
        ir = sample[:, 2]
    dpg.set_value("series_rg", [np.asarray(ir * 255.0, dtype=np.float32), np.asarray(r * 255.0, dtype=np.float32)])
    dpg.set_value("series_rb", [np.asarray(ir * 255.0, dtype=np.float32), np.asarray(g * 255.0, dtype=np.float32)])
    dpg.set_value("series_gb", [np.asarray(r * 255.0, dtype=np.float32), np.asarray(g * 255.0, dtype=np.float32)])

def _hist_counts_bincount(values, bins):
    # Kept for backward compatibility with local references; implementation lives in pcv_scopes.
    vals = np.asarray(values).ravel()
    if vals.size == 0:
        return np.zeros((bins,), dtype=np.int64)
    vals = np.clip(vals, 0.0, 1.0)
    inds = np.floor(vals * bins).astype(np.int64)
    inds[inds >= bins] = bins - 1
    return np.bincount(inds, minlength=bins)[:bins]

def build_histogram_texture(src_img, bins=HIST_W, height=HIST_H):
    try:
        gain = float(dpg.get_value("hist_gain"))
        if gain <= 0:
            gain = 1.0
    except Exception:
        gain = 10.0
    return build_histogram_texture_core(src_img, bins=bins, height=height, gain=gain)

def _update_histogram_scope(src_img):
    tex = build_histogram_texture(src_img, bins=HIST_W, height=HIST_H)
    dpg.set_value("hist_texture", _as_texture_value(tex))

def build_waveform_texture(src_img, width=WAVE_W, height=WAVE_H):
    try:
        point_size = int(dpg.get_value("scope_point_size"))
    except Exception:
        point_size = 1
    return build_waveform_texture_core(
        src_img,
        width=width,
        height=height,
        point_size=point_size,
    )

def _update_waveform_scope(src_img):
    tex = build_waveform_texture(src_img, width=WAVE_W, height=WAVE_H)
    dpg.set_value("wave_texture", _as_texture_value(tex))

def _update_vectorscope(src_img):
    _apply_vectorscope_zoom()
    series = compute_vectorscope_series(src_img, max_scope_points=MAX_SCOPE_POINTS)
    for tag, (xo, yo) in series.items():
        dpg.set_value(tag, [xo, yo])

def update_scope_graph(src_img):
    if src_img is None:
        return
    try:
        if _scope_mode == "Histogram":
            _update_histogram_scope(src_img)
        elif _scope_mode == "Waveform":
            _update_waveform_scope(src_img)
        else:
            _update_vectorscope(src_img)
    except Exception:
        pass

def set_scope_mode(mode):
    global _scope_mode, _scope_last_update_ts
    if mode not in SCOPE_MODES:
        return
    _scope_mode = mode
    if dpg.does_item_exist("scope_mode_display"):
        dpg.set_value("scope_mode_display", f"Active graph: {mode}")
    if dpg.does_item_exist("scope_histogram_group"):
        dpg.configure_item("scope_histogram_group", show=(mode == "Histogram"))
    if dpg.does_item_exist("scope_waveform_group"):
        dpg.configure_item("scope_waveform_group", show=(mode == "Waveform"))
    if dpg.does_item_exist("scope_vectorscope_group"):
        dpg.configure_item("scope_vectorscope_group", show=(mode == "Vectorscope"))
    _apply_vectorscope_zoom()
    _scope_last_update_ts = 0.0
    request_preview_update()

def _schedule_preview_update():
    global _preview_update_scheduled, _preview_update_dirty
    running = False
    try:
        running = bool(dpg.is_dearpygui_running())
    except Exception:
        running = False
    if not running:
        # If the frame loop is not running yet, render immediately and clear schedule state.
        _preview_update_scheduled = False
        _preview_update_dirty = False
        try:
            update_main_preview()
        except Exception:
            pass
        return
    try:
        dpg.set_frame_callback(dpg.get_frame_count() + 1, _drain_preview_update)
    except Exception:
        # Fallback path if frame callbacks are temporarily unavailable.
        _preview_update_scheduled = False
        _preview_update_dirty = False
        update_main_preview()

def _drain_preview_update(sender=None, app_data=None):
    global _preview_update_scheduled, _preview_update_dirty, _preview_last_update_ts
    if not _preview_update_dirty:
        _preview_update_scheduled = False
        return

    now = time.monotonic()
    min_interval = _preview_update_interval_s()
    if _preview_last_update_ts > 0.0 and (now - _preview_last_update_ts) < min_interval:
        _schedule_preview_update()
        return

    _preview_update_dirty = False
    try:
        update_main_preview()
    finally:
        _preview_last_update_ts = time.monotonic()

    if _preview_update_dirty:
        _schedule_preview_update()
    else:
        _preview_update_scheduled = False

def request_preview_update(sender=None, app_data=None):
    global _preview_update_scheduled, _preview_update_dirty
    _preview_update_dirty = True
    if _preview_update_scheduled:
        return
    _preview_update_scheduled = True
    _schedule_preview_update()

# ----------------------------
# Main preview (applies WB before transform)
# ----------------------------
def update_main_preview(sender=None, app_data=None):
    global preview_work_img, _scope_last_update_ts, _scatter_last_update_ts, _channel_last_update_ts
    if preview_img is None:
        _set_compute_status()
        preview_work_img = None
        try:
            dpg.set_value("series_rg", [[], []])
            dpg.set_value("series_rb", [[], []])
            dpg.set_value("series_gb", [[], []])
            dpg.set_value("hist_texture", _as_texture_value(np.zeros((HIST_H, HIST_W, 4), dtype=np.float32)))
            dpg.set_value("wave_texture", _as_texture_value(np.zeros((WAVE_H, WAVE_W, 4), dtype=np.float32)))
            dpg.set_value("vectorscope_series_r", [[], []])
            dpg.set_value("vectorscope_series_y", [[], []])
            dpg.set_value("vectorscope_series_g", [[], []])
            dpg.set_value("vectorscope_series_c", [[], []])
            dpg.set_value("vectorscope_series_b", [[], []])
            dpg.set_value("vectorscope_series_m", [[], []])
        except Exception:
            pass
        return
    work_img = preview_work_img if preview_work_img is not None else preview_img
    temp = dpg.get_value("wb_temp")
    tint = dpg.get_value("wb_tint")
    wb_preview, out = compute_preview_pipeline(work_img, temp, tint)
    out_preview = apply_lut_preview(out)
    rgba = render_into_texture(out_preview, MAX_PREVIEW_W, MAX_PREVIEW_H)
    dpg.set_value("texture", _as_texture_value(rgba))
    update_warnings()
    now = time.monotonic()
    converting = _is_conversion_active()
    channel_min_interval = _channel_min_interval_during_convert_s if converting else _channel_min_interval_idle_s
    if (now - _channel_last_update_ts) >= float(channel_min_interval):
        update_channel_previews(out_preview)
        _channel_last_update_ts = now
    scatter_min_interval = _scatter_min_interval_during_convert_s if converting else _scatter_min_interval_idle_s
    if converting:
        scope_min_interval = _scope_min_interval_during_convert_s
    else:
        if _scope_mode == "Histogram":
            scope_min_interval = _scope_min_interval_idle_s
        elif _scope_mode == "Waveform":
            scope_min_interval = max(_scope_min_interval_idle_s, 0.14)
        else:
            scope_min_interval = max(_scope_min_interval_idle_s, 0.12)
    try:
        scatter_open = bool(dpg.get_value("scatter_header")) if dpg.does_item_exist("scatter_header") else True
    except Exception:
        scatter_open = True
    if scatter_open and (now - _scatter_last_update_ts) >= float(scatter_min_interval):
        scatter_src, scatter_converted = _scatter_source_image(wb_preview, out)
        _update_scatterplots(scatter_src, use_converted=scatter_converted)
        _scatter_last_update_ts = now
    try:
        scope_open = bool(dpg.get_value("scope_header")) if dpg.does_item_exist("scope_header") else True
    except Exception:
        scope_open = True
    if scope_open and (now - _scope_last_update_ts) >= float(scope_min_interval):
        scope_src = _downsample_scope_source(_scope_source_image(wb_preview, out_preview))
        update_scope_graph(scope_src)
        _scope_last_update_ts = now

# ----------------------------
# File dialog callback -> dispatch video load
# ----------------------------
def open_file_callback(sender, app_data):
    path = _extract_path_from_file_dialog_appdata(app_data)
    if not path:
        print("No file selected")
        return
    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_VIDEO_EXTENSIONS:
        load_video_from_path(path)
    else:
        print("Unsupported file type (video-only):", path)

# ----------------------------
# Video load & timeline scrubber
# ----------------------------
def load_video_from_path(path):
    global video_capture, video_path, video_frame_count, video_fps, preview_img
    global video_width, video_height, _preview_ffmpeg_path, _preview_10bit_warned_once
    global _scope_last_update_ts, _scatter_last_update_ts, _channel_last_update_ts
    if not CV2_AVAILABLE:
        dpg.show_item("cv2_missing_text")
        print("OpenCV not available — video disabled.")
        return
    if not os.path.exists(path):
        print("Video not found:", path)
        return
    # release prior capture
    try:
        if video_capture is not None:
            video_capture.release()
    except Exception:
        pass
    cap = None
    try:
        if hasattr(cv2, "CAP_FFMPEG"):
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
            if cap is None or not cap.isOpened():
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                cap = cv2.VideoCapture(path)
        else:
            cap = cv2.VideoCapture(path)
    except Exception:
        cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Failed to open video:", path)
        return
    video_capture = cap
    video_path = path
    _preview_ffmpeg_path = None
    _preview_10bit_warned_once = False
    _update_add_queue_button_label()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    # update global (local copies)
    globals()["video_frame_count"] = frame_count if frame_count > 0 else 0
    globals()["video_fps"] = fps if fps > 0.0 else 25.0
    globals()["video_width"] = width if width > 0 else 0
    globals()["video_height"] = height if height > 0 else 0
    frame = read_frame_from_capture(0)
    if frame is None:
        print("Failed to read first frame")
        return
    _set_preview_frame(frame)
    _scope_last_update_ts = 0.0
    _scatter_last_update_ts = 0.0
    _channel_last_update_ts = 0.0
    try:
        dpg.set_value("file", path)
    except Exception:
        pass
    # create/config timeline slider under right_preview_controls
    try:
        if not dpg.does_item_exist("timeline_slider"):
            dpg.add_slider_int(label="Timeline (frame)", tag="timeline_slider",
                               default_value=0, min_value=0, max_value=max(0, frame_count - 1),
                               callback=on_timeline_changed, parent="right_preview_controls", width=-1)
        else:
            dpg.configure_item("timeline_slider", min_value=0, max_value=max(0, frame_count - 1))
            dpg.set_value("timeline_slider", 0)
    except Exception as e:
        print("Failed to create/config timeline slider:", e)
    request_preview_update()

def _preview_ffmpeg_for_10bit():
    global _preview_ffmpeg_path
    if _preview_ffmpeg_path and os.path.exists(_preview_ffmpeg_path) and os.access(_preview_ffmpeg_path, os.X_OK):
        return _preview_ffmpeg_path
    _preview_ffmpeg_path = _find_ffmpeg_candidates()
    return _preview_ffmpeg_path

def _read_frame_with_ffmpeg_10bit(frame_index):
    global _preview_10bit_warned_once
    if not video_path or video_width <= 0 or video_height <= 0:
        return None
    ffmpeg_path = _preview_ffmpeg_for_10bit()
    if not ffmpeg_path:
        if not _preview_10bit_warned_once:
            print("10-bit preview requested, but ffmpeg was not found. Falling back to 8-bit preview.")
            _preview_10bit_warned_once = True
        return None
    fps = float(video_fps) if float(video_fps) > 0.0 else 25.0
    ts = max(0.0, float(frame_index) / fps)
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{ts:.6f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt",
        "rgb48le",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        expected = int(video_width) * int(video_height) * 3 * 2
        if proc.returncode != 0 or len(proc.stdout) < expected:
            if not _preview_10bit_warned_once:
                err = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()
                print("10-bit preview decode failed; falling back to 8-bit preview.", err if err else "")
                _preview_10bit_warned_once = True
            return None
        buf = proc.stdout[:expected]
        frame = np.frombuffer(buf, dtype="<u2").reshape((int(video_height), int(video_width), 3)).astype(np.float32) / 65535.0
        return frame
    except Exception as e:
        if not _preview_10bit_warned_once:
            print("10-bit preview error; falling back to 8-bit preview:", e)
            _preview_10bit_warned_once = True
        return None

def read_frame_from_capture(frame_index):
    global video_capture
    if _preview_bit_depth_mode() == "10-bit":
        frame10 = _read_frame_with_ffmpeg_10bit(frame_index)
        if frame10 is not None:
            return frame10
    if video_capture is None:
        return None
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ret, frame = video_capture.read()
    if not ret:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return frame_rgb

def on_timeline_changed(sender, app_data):
    global preview_img, _scope_last_update_ts, _scatter_last_update_ts, _channel_last_update_ts
    if not CV2_AVAILABLE:
        return
    idx = dpg.get_value("timeline_slider")
    frame = read_frame_from_capture(idx)
    if frame is None:
        print("Failed to read frame at", idx)
        return
    _set_preview_frame(frame)
    _scope_last_update_ts = 0.0
    _scatter_last_update_ts = 0.0
    _channel_last_update_ts = 0.0
    update_main_preview()

def on_preview_image_click(sender, app_data):
    global wb_dropper_active
    if not wb_dropper_active or preview_img is None:
        return
    if not dpg.does_item_exist("preview_image") or not dpg.is_item_hovered("preview_image"):
        return
    try:
        mouse_pos = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min("preview_image")
        rel_x = mouse_pos[0] - rect_min[0]
        rel_y = mouse_pos[1] - rect_min[1]
        h, w = preview_img.shape[:2]
        scale = min(MAX_PREVIEW_W / w, MAX_PREVIEW_H / h)
        target_w = int(round(w * scale))
        target_h = int(round(h * scale))
        pad_x = (MAX_PREVIEW_W - target_w) // 2
        pad_y = (MAX_PREVIEW_H - target_h) // 2
        img_x = rel_x - pad_x
        img_y = rel_y - pad_y
        if img_x < 0 or img_x >= target_w or img_y < 0 or img_y >= target_h:
            return
        orig_x = int(np.clip(int(img_x / scale), 0, w - 1))
        orig_y = int(np.clip(int(img_y / scale), 0, h - 1))
        sampled_rgb = preview_img[orig_y, orig_x, :]
        temp, tint = find_wb_from_gray_sample(sampled_rgb)
        dpg.set_value("wb_temp", int(temp))
        dpg.set_value("wb_tint", int(tint))
        wb_dropper_active = False
        dpg.configure_item("wb_dropper_btn", label="Reference Set")
        update_main_preview()
        print(f"WB sampled at ({orig_x},{orig_y}) -> temp {int(temp)}K tint {int(tint)}")
    except Exception as e:
        print("WB dropper click failed:", e)

# ----------------------------
# Preset helpers: save/load/list/delete
# ----------------------------
def _preset_filename_from_name(name):
    safe = "".join(c for c in name if c.isalnum() or c in "-_. ").strip()
    if not safe:
        raise ValueError("Invalid preset name")
    return os.path.join(PRESETS_DIR, safe + ".json")

def list_presets_on_disk():
    items = []
    try:
        for fname in sorted(os.listdir(PRESETS_DIR)):
            if fname.lower().endswith(".json"):
                items.append(os.path.splitext(fname)[0])
    except Exception:
        pass
    return items

def refresh_presets_dropdown():
    items = list_presets_on_disk()
    try:
        dpg.configure_item("preset_combo", items=items)
        if items:
            current = dpg.get_value("preset_combo")
            if not current or current not in items:
                dpg.set_value("preset_combo", items[0])
        else:
            dpg.set_value("preset_combo", "")
    except Exception:
        pass

def save_preset_to_folder(name):
    if not name or not name.strip():
        print("Preset name empty.")
        return False
    try:
        path = _preset_filename_from_name(name)
    except ValueError:
        print("Preset name invalid. Use alphanumerics, -, _, . and spaces.")
        return False
    if os.path.exists(path):
        try:
            dpg.set_value("preset_to_overwrite_path", path)
            dpg.set_value("preset_to_overwrite_name", os.path.splitext(os.path.basename(path))[0])
            dpg.set_value(
                "overwrite_preset_name_display",
                f"Preset '{os.path.splitext(os.path.basename(path))[0]}' already exists. Overwrite it?"
            )
            dpg.show_item("overwrite_preset_modal")
            return False
        except Exception:
            pass
    return _write_preset_file(path)

def _write_preset_file(path):
    data = {}
    for tag in PRESET_SLIDERS:
        try:
            data[tag] = dpg.get_value(tag)
        except Exception:
            data[tag] = None
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Preset saved: {path}")
        refresh_presets_dropdown()
        dpg.set_value("preset_combo", os.path.splitext(os.path.basename(path))[0])
        return True
    except Exception as e:
        print("Failed to save preset:", e)
        return False

def confirm_overwrite_preset(sender=None, app_data=None):
    path = dpg.get_value("preset_to_overwrite_path")
    dpg.hide_item("overwrite_preset_modal")
    dpg.set_value("overwrite_preset_name_display", "")
    dpg.set_value("preset_to_overwrite_name", "")
    dpg.set_value("preset_to_overwrite_path", "")
    if not path:
        return
    _write_preset_file(path)

def cancel_overwrite_preset(sender=None, app_data=None):
    dpg.hide_item("overwrite_preset_modal")
    dpg.set_value("overwrite_preset_name_display", "")
    dpg.set_value("preset_to_overwrite_name", "")
    dpg.set_value("preset_to_overwrite_path", "")

def load_preset_from_folder(name):
    if not name:
        print("No preset name selected.")
        return False
    path = os.path.join(PRESETS_DIR, name + ".json")
    if not os.path.exists(path):
        print("Preset file missing:", path)
        refresh_presets_dropdown()
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Failed to read preset:", e)
        return False
    changed = False
    for tag in PRESET_SLIDERS:
        if tag in data:
            try:
                dpg.set_value(tag, data[tag])
                changed = True
            except Exception:
                pass
    if changed:
        update_main_preview()
        print("Preset loaded:", path)
        return True
    else:
        print("Preset had no known keys.")
        return False

def delete_selected_preset(name):
    if not name:
        print("No preset selected to delete.")
        return False
    path = os.path.join(PRESETS_DIR, name + ".json")
    if not os.path.exists(path):
        print("Preset file not found to delete:", path)
        refresh_presets_dropdown()
        return False
    try:
        os.remove(path)
        print("Deleted preset:", path)
        refresh_presets_dropdown()
        return True
    except Exception as e:
        print("Failed to delete preset:", e)
        return False

def show_delete_preset_confirm(sender=None, app_data=None):
    sel = dpg.get_value("preset_combo")
    if not sel:
        print("No preset selected to delete.")
        return
    dpg.set_value("preset_to_delete", sel)
    dpg.show_item("delete_preset_modal")

def confirm_delete_preset(sender=None, app_data=None):
    name = dpg.get_value("preset_to_delete")
    if not name:
        dpg.hide_item("delete_preset_modal")
        return
    ok = delete_selected_preset(name)
    dpg.hide_item("delete_preset_modal")
    if ok:
        dpg.configure_item("preset_combo", items=list_presets_on_disk())
        dpg.set_value("preset_to_delete", "")

def reset_to_defaults(sender=None, app_data=None):
    for tag, val in DEFAULT_PRESET.items():
        try:
            dpg.set_value(tag, val)
        except Exception:
            pass
    update_main_preview()
    print("Sliders reset to defaults.")

# ----------------------------
# Queue UI helpers (Add / Remove / Move)
# ----------------------------
def _queue_path_key(path):
    if not path:
        return ""
    try:
        return os.path.normcase(os.path.realpath(os.path.abspath(path)))
    except Exception:
        return os.path.normcase(str(path))

def _find_queue_index_by_input(path):
    key = _queue_path_key(path)
    if not key:
        return None
    for i, job in enumerate(queue_items):
        if _queue_path_key(job.get("input")) == key:
            return i
    return None

def _selected_queue_index(sel=None):
    if sel is None:
        try:
            sel = dpg.get_value("queue_list")
        except Exception:
            sel = None
    if isinstance(sel, (list, tuple)):
        sel = sel[0] if sel else None
    if not sel:
        return None
    m = re.match(r"^\s*(\d+)\s+", str(sel))
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(queue_items):
            return idx
    for i, job in enumerate(queue_items):
        if os.path.basename(job.get("input", "")) in str(sel):
            return i
    return None

def _update_add_queue_button_label():
    try:
        if not dpg.does_item_exist("add_queue_button"):
            return
        idx = _find_queue_index_by_input(video_path)
        dpg.configure_item("add_queue_button", label=("Update to Queue" if idx is not None else "Add to Queue"))
    except Exception:
        pass

def _queue_label_for_item(item, index=None):
    fn = os.path.basename(item["input"])
    lbl = f"{index if index is not None else ''} {fn} — {item['preset'].get('wb_temp', '')}K"
    return lbl.strip()

def refresh_queue_listbox():
    items = [_queue_label_for_item(q, i) for i, q in enumerate(queue_items)]
    try:
        if not dpg.does_item_exist("queue_list"):
            return
        dpg.configure_item("queue_list", items=items)
        # if nothing selected, set selection to first
        sel = dpg.get_value("queue_list")
        if sel not in items and items:
            dpg.set_value("queue_list", items[0])
    except Exception:
        pass
    _update_add_queue_button_label()

def load_queue_item_for_edit(index):
    if index is None or index < 0 or index >= len(queue_items):
        return
    job = queue_items[index]
    in_path = job.get("input")
    if not in_path:
        return
    if not os.path.exists(in_path):
        print("Queued file missing:", in_path)
        return
    load_video_from_path(in_path)
    preset = job.get("preset", {}) or {}
    changed = False
    for tag in PRESET_SLIDERS:
        if tag in preset:
            try:
                dpg.set_value(tag, preset[tag])
                changed = True
            except Exception:
                pass
    if changed:
        update_main_preview()
    try:
        dpg.set_value("queue_list", _queue_label_for_item(job, index))
    except Exception:
        pass
    _update_add_queue_button_label()
    print("Loaded queue item for edit:", in_path)

def on_queue_list_double_click(sender=None, app_data=None):
    idx = _selected_queue_index()
    if idx is None:
        return
    load_queue_item_for_edit(idx)

def add_current_to_queue(sender=None, app_data=None):
    global queue_items, video_path
    if video_path is None:
        print("No video loaded to add to queue.")
        return
    # snapshot preset values
    preset = {}
    for k in PRESET_SLIDERS:
        try:
            preset[k] = dpg.get_value(k)
        except Exception:
            preset[k] = DEFAULT_PRESET.get(k)
    # propose output path
    folder, name = os.path.split(video_path)
    base, ext = os.path.splitext(name)
    out_ext = ext if ext and ext.lower() in SUPPORTED_VIDEO_EXTENSIONS else ".mp4"
    out_path = os.path.join(folder, base + "_IRG_converted" + out_ext)
    job = {
        "id": str(uuid.uuid4()),
        "input": video_path,
        "output": out_path,
        "preset": preset,
        "label": os.path.basename(video_path)
    }
    idx = _find_queue_index_by_input(video_path)
    if idx is not None:
        queue_items[idx]["input"] = video_path
        queue_items[idx]["preset"] = preset
        queue_items[idx]["label"] = os.path.basename(video_path)
        if not queue_items[idx].get("output"):
            queue_items[idx]["output"] = out_path
        refresh_queue_listbox()
        try:
            dpg.set_value("queue_list", _queue_label_for_item(queue_items[idx], idx))
        except Exception:
            pass
        _update_add_queue_button_label()
        print("Updated queue:", queue_items[idx]["input"])
    else:
        queue_items.append(job)
        refresh_queue_listbox()
        try:
            dpg.set_value("queue_list", _queue_label_for_item(job, len(queue_items) - 1))
        except Exception:
            pass
        _update_add_queue_button_label()
        print("Added to queue:", job["input"])

def remove_selected_from_queue(sender=None, app_data=None):
    global queue_items
    idx = _selected_queue_index()
    if idx is None:
        return
    if idx < 0 or idx >= len(queue_items):
        return
    removed = queue_items.pop(idx)
    refresh_queue_listbox()
    print("Removed from queue:", removed["input"])

def move_selected_up(sender=None, app_data=None):
    global queue_items
    idx = _selected_queue_index()
    if idx is None:
        return
    if idx <= 0 or idx >= len(queue_items):
        return
    queue_items[idx-1], queue_items[idx] = queue_items[idx], queue_items[idx-1]
    refresh_queue_listbox()

def move_selected_down(sender=None, app_data=None):
    global queue_items
    idx = _selected_queue_index()
    if idx is None:
        return
    if idx < 0 or idx >= len(queue_items)-1:
        return
    queue_items[idx+1], queue_items[idx] = queue_items[idx], queue_items[idx+1]
    refresh_queue_listbox()

# ----------------------------
# Converter helpers: find ffmpeg
# ----------------------------
def _find_ffmpeg_candidates():
    p = shutil.which("ffmpeg")
    if p:
        return p
    candidates = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "/bin/ffmpeg",
        "/snap/bin/ffmpeg",
        "/usr/local/opt/ffmpeg/bin/ffmpeg"
    ]
    for c in candidates:
        try:
            if c and os.path.exists(c) and os.access(c, os.X_OK):
                return c
        except Exception:
            pass
    return None


def _resolve_converter_command(batch_path, ffmpeg_path=None):
    """
    Resolve converter launch command for source and frozen builds.

    Source mode:
      [python, convert_clip.py, --batch ...]

    Frozen mode (PyInstaller):
      [convert_clip_bin, --batch ...] where convert_clip_bin is bundled as a binary.
    """
    if getattr(sys, "frozen", False):
        candidates = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(os.path.join(meipass, "convert_clip_bin"))
            candidates.append(os.path.join(meipass, "convert_clip_bin.exe"))
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.append(os.path.join(exe_dir, "convert_clip_bin"))
        candidates.append(os.path.join(exe_dir, "convert_clip_bin.exe"))
        for conv_bin in candidates:
            if conv_bin and os.path.exists(conv_bin) and os.access(conv_bin, os.X_OK):
                cmd = [conv_bin, "--batch", batch_path]
                if ffmpeg_path:
                    cmd += ["--ffmpeg-path", ffmpeg_path]
                return cmd, None
        return None, "Error: bundled convert_clip_bin not found."

    gui_dir = os.path.dirname(os.path.abspath(__file__))
    convert_script = os.path.join(gui_dir, "convert_clip.py")
    if not os.path.exists(convert_script):
        return None, f"Error: convert_clip.py not found at expected path: {convert_script}"
    cmd = [sys.executable, convert_script, "--batch", batch_path]
    if ffmpeg_path:
        cmd += ["--ffmpeg-path", ffmpeg_path]
    return cmd, None

# ----------------------------
# Converter status state (v2)
# ----------------------------
def _cleanup_convert_tmpdir():
    global _convert_tmpdir
    if _convert_tmpdir:
        try:
            shutil.rmtree(_convert_tmpdir, ignore_errors=True)
        except Exception:
            pass
        _convert_tmpdir = None

def _status_line_should_forward(line):
    if _conv_status_debug:
        return True
    txt = str(line).strip()
    if not txt:
        return False
    if txt.startswith(
        (
            "JOB_START:",
            "TOTAL_FRAMES:",
            "PROCESSED:",
            "JOB_DONE:",
            "BATCH_DONE",
            "DURATION_MS:",
            "BACKEND_REQUESTED:",
            "FFMPEG:",
            "COMPUTE_BACKEND_ACTIVE:",
            "WIDTH:",
            "HEIGHT:",
            "FPS:",
            "INPUT_BIT_DEPTH:",
            "INPUT_PIX_FMT:",
            "OUTPUT_BIT_DEPTH:",
            "OUTPUT_PIX_FMT:",
            "ERROR:",
            "[error]",
            "[warn]",
        )
    ):
        return True
    if txt.startswith("FFMPEG_PROGRESS:"):
        return False
    if re.match(r"^[a-zA-Z0-9_]+=.+$", txt):
        # Drop ffmpeg -progress key/value chatter by default.
        return False
    low = txt.lower()
    if ("error" in low) or ("failed" in low) or ("exception" in low):
        return True
    # Drop encoder banner spam and other verbose non-critical lines.
    return False

def _status_line_is_critical(line):
    txt = str(line).strip()
    return txt.startswith(("JOB_START:", "TOTAL_FRAMES:", "PROCESSED:", "JOB_DONE:", "BATCH_DONE", "ERROR:", "[error]"))

def _enqueue_status_line(q, line):
    if q is None:
        return
    txt = str(line).rstrip("\n")
    if not _status_line_should_forward(txt):
        return
    try:
        q.put_nowait(txt)
        return
    except queue.Full:
        pass
    if _status_line_is_critical(txt):
        try:
            q.put(txt, timeout=0.05)
        except Exception:
            pass

def _enqueue_status_event(q, event):
    if q is None:
        return
    try:
        q.put_nowait(event)
        return
    except queue.Full:
        pass
    # Prefer delivering terminal control events over older log lines.
    try:
        q.get_nowait()
    except Exception:
        pass
    try:
        q.put_nowait(event)
    except Exception:
        pass

def _conv_status_log_keep(kind, line, state):
    if _conv_status_debug:
        return True
    if kind in {"JOB_START", "TOTAL_FRAMES", "JOB_DONE", "BATCH_DONE"}:
        return True
    if kind == "PROCESSED":
        now = time.monotonic()
        try:
            frame = int(line.split(":", 1)[1].strip())
        except Exception:
            frame = 0
        last_f = int(state.get("last_processed_log_frame", 0))
        last_t = float(state.get("last_processed_log_ts", 0.0))
        if (frame - last_f) >= int(_conv_status_log_every_frames) or (now - last_t) >= float(_conv_status_log_min_interval_s):
            state["last_processed_log_frame"] = frame
            state["last_processed_log_ts"] = now
            return True
        return False
    if line.startswith("ERROR:") or line.startswith("[error]"):
        return True
    if line.startswith(
        (
            "BACKEND_REQUESTED:",
            "FFMPEG:",
            "FPS:",
            "WIDTH:",
            "HEIGHT:",
            "INPUT_",
            "OUTPUT_",
            "COMPUTE_BACKEND_ACTIVE:",
        )
    ):
        return True
    return False

def _append_status_log_locked(state, line):
    txt = str(line).strip()
    if not txt:
        return
    state["log_lines"].append(txt)
    state["log_dirty"] = True

def _set_status_text_locked(state, text):
    t = str(text)
    if state.get("status_text") != t:
        state["status_text"] = t

def _current_status_progress_locked(state):
    total_jobs = int(state.get("total_jobs") or 0)
    done_jobs = int(state.get("done_jobs") or 0)
    current = float(state.get("current_job_display_frac") or 0.0)
    current = max(0.0, min(1.0, current))
    if total_jobs <= 0:
        return current
    if not bool(state.get("job_active", False)):
        current = 0.0
    return max(0.0, min(1.0, (float(done_jobs) + current) / float(total_jobs)))

def _update_frame_progress_locked(state):
    total = state.get("current_job_total_frames")
    processed = int(state.get("current_job_processed_frames") or 0)
    if total is not None:
        try:
            total_i = max(0, int(total))
        except Exception:
            total_i = 0
        if total_i > 0:
            frac = max(0.0, min(1.0, float(processed) / float(total_i)))
            state["current_job_frame_frac"] = max(float(state.get("current_job_frame_frac") or 0.0), frac)
        else:
            prev = float(state.get("current_job_frame_frac") or 0.0)
            # Fallback when total frames are unknown: monotonic asymptotic progress from processed count.
            pseudo = 1.0 - math.exp(-float(max(0, processed)) / 240.0) if processed > 0 else prev
            state["current_job_frame_frac"] = max(prev, min(0.985, pseudo))
    else:
        prev = float(state.get("current_job_frame_frac") or 0.0)
        pseudo = 1.0 - math.exp(-float(max(0, processed)) / 240.0) if processed > 0 else prev
        state["current_job_frame_frac"] = max(prev, min(0.985, pseudo))
    display = float(state.get("current_job_frame_frac") or 0.0)
    if bool(state.get("job_active", False)):
        display = min(0.99, display)
    state["current_job_display_frac"] = max(0.0, min(1.0, display))

def _set_live_status_from_frames_locked(state):
    total_jobs = int(state.get("total_jobs") or 0)
    idx = state.get("current_job_index")
    if idx is None:
        job_no = min(total_jobs, int(state.get("done_jobs") or 0) + 1) if total_jobs > 0 else "?"
    else:
        try:
            job_no = int(idx) + 1
        except Exception:
            job_no = "?"
    total_label = total_jobs if total_jobs > 0 else "?"
    processed = int(state.get("current_job_processed_frames") or 0)
    total = state.get("current_job_total_frames")
    if total is None:
        frame_part = str(processed)
    else:
        frame_part = f"{processed}/{int(total)}"
    _set_status_text_locked(state, f"Job {job_no}/{total_label}: frames {frame_part}")

def _finish_status_state_locked(state, code=0, error_message=None):
    if code == 0 and not error_message:
        state["active"] = False
        state["done"] = True
        state["success"] = True
        state["exit_code"] = 0
        total_jobs = int(state.get("total_jobs") or 0)
        state["done_jobs"] = max(int(state.get("done_jobs") or 0), total_jobs)
        state["job_active"] = False
        state["current_job_frame_frac"] = 1.0
        state["current_job_display_frac"] = 1.0
        state["preview_rebuild_pending"] = True
        _set_status_text_locked(state, "Conversion complete. Closing dialog...")
        _append_status_log_locked(state, "[done] conversion complete")
        state["autoclose_pending"] = True
    else:
        msg = error_message or f"Conversion finished with exit code {code}"
        state["active"] = False
        state["done"] = True
        state["success"] = False
        state["exit_code"] = int(code) if code is not None else 1
        state["error_message"] = str(msg)
        state["job_active"] = False
        state["preview_rebuild_pending"] = True
        _set_status_text_locked(state, "Conversion error")
        _append_status_log_locked(state, "[error] " + str(msg))

def _init_conv_state(total_jobs):
    global _conv_state, _conv_state_rev, _conv_ui_applied_rev, _conv_ui_last_log_push_ts
    now = time.monotonic()
    with _conv_state_lock:
        _conv_state = {
            "active": True,
            "done": False,
            "success": False,
            "error_message": None,
            "exit_code": None,
            "total_jobs": int(total_jobs),
            "done_jobs": 0,
            "job_active": False,
            "current_job_index": None,
            "current_job_total_frames": None,
            "current_job_processed_frames": 0,
            "current_job_frame_frac": 0.0,
            "current_job_display_frac": 0.0,
            "status_text": f"Starting batch conversion ({int(total_jobs)} jobs)...",
            "log_lines": deque(maxlen=_conv_status_log_max_lines),
            "log_dirty": True,
            "last_event_ts": now,
            "stall_warned": False,
            "autoclose_pending": False,
            "preview_rebuild_pending": False,
            "last_processed_log_frame": 0,
            "last_processed_log_ts": 0.0,
        }
        _conv_state_rev = 1
        _conv_ui_applied_rev = -1
        _conv_ui_last_log_push_ts = 0.0

def _status_apply_event(event):
    global _conv_state_rev
    with _conv_state_lock:
        state = _conv_state
        if state is None:
            return

        now = time.monotonic()
        state["last_event_ts"] = now
        state["stall_warned"] = False
        changed = False

        if isinstance(event, dict):
            etype = str(event.get("type") or "").upper()
            if etype == "DONE":
                code = int(event.get("code", 1))
                _finish_status_state_locked(state, code=code, error_message=None if code == 0 else f"Conversion finished with exit code {code}")
                changed = True
            elif etype == "ERROR":
                _finish_status_state_locked(state, code=1, error_message=event.get("message", "Unknown error"))
                changed = True
        else:
            line = str(event).strip()
            if line:
                kind, payload = _parse_progress_from_line_batch(line)
                keep_line = _conv_status_log_keep(kind, line, state)
                if keep_line:
                    _append_status_log_locked(state, line)
                    changed = True

                if kind == "JOB_START":
                    idx = payload.get("index")
                    state["job_active"] = True
                    state["current_job_index"] = int(idx) if idx is not None else None
                    state["current_job_total_frames"] = None
                    state["current_job_processed_frames"] = 0
                    state["current_job_frame_frac"] = 0.0
                    state["current_job_display_frac"] = 0.0
                    total_jobs = int(state.get("total_jobs") or 0)
                    job_no = (int(idx) + 1) if idx is not None else "?"
                    total_label = total_jobs if total_jobs > 0 else "?"
                    _set_status_text_locked(state, f"Job {job_no}/{total_label} started")
                    changed = True
                elif kind == "TOTAL_FRAMES":
                    try:
                        state["current_job_total_frames"] = max(0, int(payload))
                    except Exception:
                        state["current_job_total_frames"] = None
                    _set_live_status_from_frames_locked(state)
                    changed = True
                elif kind == "PROCESSED":
                    try:
                        p_now = max(0, int(payload))
                    except Exception:
                        p_now = None
                    if p_now is not None:
                        state["job_active"] = True
                        state["current_job_processed_frames"] = max(int(state.get("current_job_processed_frames") or 0), p_now)
                        _update_frame_progress_locked(state)
                        _set_live_status_from_frames_locked(state)
                        changed = True
                elif kind == "JOB_DONE":
                    idx = payload.get("index")
                    code = int(payload.get("code", 0))
                    if idx is not None:
                        state["done_jobs"] = max(int(state.get("done_jobs") or 0), int(idx) + 1)
                    else:
                        state["done_jobs"] = min(int(state.get("total_jobs") or 0), int(state.get("done_jobs") or 0) + 1)
                    state["job_active"] = False
                    if code == 0:
                        state["current_job_frame_frac"] = 1.0
                    state["current_job_display_frac"] = 0.0
                    _set_status_text_locked(state, f"Job {idx} done (exit {code})")
                    changed = True
                elif kind == "BATCH_DONE":
                    state["done_jobs"] = max(int(state.get("done_jobs") or 0), int(state.get("total_jobs") or 0))
                    state["job_active"] = False
                    state["current_job_display_frac"] = 0.0
                    _set_status_text_locked(state, "Batch finished")
                    changed = True
                elif kind == "DURATION_MS":
                    changed = changed or False
                elif kind in {"FFMPEG_PROGRESS", "FFMPEG_KV"}:
                    changed = changed or False
                else:
                    # Fallback parser for legacy textual progress lines.
                    m = re.search(r"Processed\s+(\d+)\s*/\s*(\d+)\s*frames", line, re.I)
                    if m:
                        try:
                            got = max(0, int(m.group(1)))
                            tot = max(0, int(m.group(2)))
                        except Exception:
                            got = None
                            tot = None
                        if got is not None:
                            state["job_active"] = True
                            state["current_job_processed_frames"] = max(int(state.get("current_job_processed_frames") or 0), got)
                            if tot is not None and tot > 0:
                                cur_tot = state.get("current_job_total_frames")
                                if cur_tot is None or int(cur_tot) <= 0:
                                    state["current_job_total_frames"] = int(tot)
                            _update_frame_progress_locked(state)
                            _set_live_status_from_frames_locked(state)
                            changed = True

        if changed:
            _conv_state_rev += 1

def _status_worker_loop():
    global _conv_raw_queue, _conv_worker_thread, _conv_state_rev
    stop_after_item = False
    while True:
        q = _conv_raw_queue
        if q is None:
            return
        try:
            item = q.get(timeout=0.1)
        except queue.Empty:
            with _conv_state_lock:
                st = _conv_state
                if st is not None and bool(st.get("done")) and not bool(st.get("active")):
                    break
            continue
        batch = [item]
        for _ in range(127):
            try:
                batch.append(q.get_nowait())
            except queue.Empty:
                break
        for item in batch:
            try:
                _status_apply_event(item)
            except Exception as e:
                with _conv_state_lock:
                    st = _conv_state
                    if st is not None:
                        _append_status_log_locked(st, f"[warn] status worker parse error: {e}")
                        _conv_state_rev += 1
            if isinstance(item, dict) and str(item.get("type") or "").upper() in {"DONE", "ERROR"}:
                stop_after_item = True
                break
        if stop_after_item:
            break
    _cleanup_convert_tmpdir()
    _conv_raw_queue = None
    _conv_worker_thread = None

def _schedule_ui_heartbeat():
    try:
        dpg.set_frame_callback(dpg.get_frame_count() + int(_ui_heartbeat_frame_step), _ui_heartbeat)
        return True
    except Exception:
        return False

def _status_snapshot():
    global _conv_ui_last_log_push_ts, _conv_state_rev
    with _conv_state_lock:
        st = _conv_state
        if st is None:
            return None

        now = time.monotonic()
        if bool(st.get("active")) and not bool(st.get("done")):
            if (now - float(st.get("last_event_ts") or 0.0)) > float(_conv_stall_warn_after_s) and not bool(st.get("stall_warned")):
                st["stall_warned"] = True
                _append_status_log_locked(st, "[warn] status stream stalled; waiting for converter output...")
                _conv_state_rev += 1

        rev = int(_conv_state_rev)
        progress_val = _current_status_progress_locked(st)
        log_text = None
        if bool(st.get("log_dirty")) and (now - float(_conv_ui_last_log_push_ts)) >= float(_conv_status_log_ui_interval_s):
            log_text = "\n".join(st.get("log_lines", []))
            st["log_dirty"] = False
            _conv_ui_last_log_push_ts = now

        return {
            "rev": rev,
            "active": bool(st.get("active")),
            "done": bool(st.get("done")),
            "success": bool(st.get("success")),
            "status_text": str(st.get("status_text") or ""),
            "progress": float(progress_val),
            "log_text": log_text,
            "autoclose_pending": bool(st.get("autoclose_pending")),
            "preview_rebuild_pending": bool(st.get("preview_rebuild_pending")),
        }

def _clear_status_autoclose_pending():
    with _conv_state_lock:
        st = _conv_state
        if st is not None:
            st["autoclose_pending"] = False

def _clear_status_preview_rebuild_pending():
    with _conv_state_lock:
        st = _conv_state
        if st is not None:
            st["preview_rebuild_pending"] = False

def _apply_status_snapshot_to_ui(snapshot):
    global _conv_success_autoclose_queued
    if snapshot is None:
        return
    try:
        if dpg.does_item_exist("conv_status_text") and dpg.get_value("conv_status_text") != snapshot["status_text"]:
            dpg.set_value("conv_status_text", snapshot["status_text"])
    except Exception:
        pass
    try:
        if dpg.does_item_exist("conv_progress"):
            dpg.set_value("conv_progress", max(0.0, min(1.0, float(snapshot.get("progress", 0.0)))))
    except Exception:
        pass
    if snapshot.get("log_text") is not None:
        try:
            if dpg.does_item_exist("conv_log"):
                dpg.set_value("conv_log", snapshot["log_text"])
        except Exception:
            pass
    try:
        if snapshot.get("active") and not snapshot.get("done"):
            dpg.configure_item("conv_cancel_button", show=True)
            dpg.configure_item("conv_close_button", show=False)
        else:
            dpg.configure_item("conv_cancel_button", show=False)
            dpg.configure_item("conv_close_button", show=True)
    except Exception:
        pass

    if snapshot.get("preview_rebuild_pending"):
        rebuild_preview_work_image()
        request_preview_update()
        _clear_status_preview_rebuild_pending()

    if snapshot.get("success") and snapshot.get("done") and snapshot.get("autoclose_pending") and not _conv_success_autoclose_queued:
        _conv_success_autoclose_queued = True
        _schedule_convert_modal_autoclose()
        _clear_status_autoclose_pending()

# ----------------------------
# Converter queue & parsing (progress extraction, extended for batch)
# ----------------------------
def _append_to_conv_log(line):
    _conv_log_lines.append(str(line))
    _flush_conv_log(force=False)

def _flush_conv_log(force=False):
    global _conv_log_last_ui_push
    now = time.monotonic()
    if not force and (now - _conv_log_last_ui_push) < 0.08:
        return
    try:
        dpg.set_value("conv_log", "\n".join(_conv_log_lines))
        _conv_log_last_ui_push = now
    except Exception:
        pass

def _parse_progress_from_line_batch(line):
    """
    Very small, robust parser for expected batch lines:
      JOB_START: <index> <input> -> <output>
      TOTAL_FRAMES: N
      PROCESSED: i
      JOB_DONE: <index> <exitcode>
      BATCH_DONE
    Also parses key=value lines from ffmpeg -progress and attempts to map out_time_ms.
    Returns a tuple: (type, payload)
    """
    line = line.strip()
    if line.startswith("JOB_START:"):
        # JOB_START: 0 /path/in.mov -> /path/out.mp4
        try:
            rest = line[len("JOB_START:"):].strip()
            parts = rest.split(None, 1)
            idx = int(parts[0])
            rest2 = parts[1] if len(parts) > 1 else ""
            # keep whole rest as payload
            return ("JOB_START", {"index": idx, "desc": rest2})
        except Exception:
            return ("LOG", line)
    if line.startswith("TOTAL_FRAMES:"):
        try:
            n = int(line.split(":",1)[1].strip())
            return ("TOTAL_FRAMES", n)
        except Exception:
            return ("LOG", line)
    if line.startswith("PROCESSED:"):
        try:
            got = int(line.split(":",1)[1].strip())
            return ("PROCESSED", got)
        except Exception:
            return ("LOG", line)
    if line.startswith("DURATION_MS:"):
        try:
            ms = int(line.split(":",1)[1].strip())
            return ("DURATION_MS", ms)
        except Exception:
            return ("LOG", line)
    if line.startswith("FFMPEG_PROGRESS:"):
        try:
            frac = float(line.split(":",1)[1].strip())
            return ("FFMPEG_PROGRESS", frac)
        except Exception:
            return ("LOG", line)
    if line.startswith("JOB_DONE:"):
        try:
            rest = line[len("JOB_DONE:"):].strip()
            parts = rest.split()
            idx = int(parts[0]) if parts else None
            code = int(parts[1]) if len(parts) > 1 else 0
            return ("JOB_DONE", {"index": idx, "code": code})
        except Exception:
            return ("LOG", line)
    if line.startswith("BATCH_DONE"):
        return ("BATCH_DONE", None)
    # ffmpeg progress pattern: out_time_ms=12345 etc.
    kv = None
    if "=" in line and re.match(r"^[a-zA-Z0-9_]+=.+$", line):
        k,v = line.split("=",1)
        kv = (k.strip(), v.strip())
        return ("FFMPEG_KV", kv)
    # fallback
    return ("LOG", line)

def _store_conv_duration_ms(v):
    global _conv_duration_ms
    try:
        _conv_duration_ms = int(v)
    except Exception:
        _conv_duration_ms = None

def _batch_progress_value():
    if _conv_total_jobs and _conv_total_jobs > 0:
        return max(0.0, min(1.0, float(_conv_done_jobs + _conv_current_job_stage) / float(_conv_total_jobs)))
    return max(0.0, min(1.0, float(_conv_current_job_stage)))

def _set_batch_progress():
    try:
        dpg.set_value("conv_progress", _batch_progress_value())
    except Exception:
        pass

def _set_conv_status(text, force=False, min_interval=0.12):
    global _conv_last_status_update
    now = time.monotonic()
    if not force and (now - _conv_last_status_update) < float(min_interval):
        return
    try:
        if force or dpg.get_value("conv_status_text") != text:
            dpg.set_value("conv_status_text", text)
            _conv_last_status_update = now
    except Exception:
        pass

def _refresh_current_job_stage():
    global _conv_current_job_stage
    # Status bar is frame-driven; encoding progress remains in logs only.
    combined = float(_conv_job_frame_frac)
    combined = max(0.0, min(1.0, combined))
    _conv_current_job_stage = max(float(_conv_current_job_stage), combined)

def _set_live_job_status():
    job_no = int(_conv_current_job_index) + 1 if _conv_current_job_index is not None else "?"
    total = _conv_total_jobs if _conv_total_jobs else "?"
    if _conv_current_job_total:
        frame_part = f"{_conv_current_job_processed}/{_conv_current_job_total}"
    else:
        frame_part = str(_conv_current_job_processed)
    text = f"Job {job_no}/{total}: frames {frame_part}"
    _set_conv_status(text, force=False)

def _schedule_convert_modal_autoclose(delay_frames=75):
    def _hide_modal(sender=None, app_data=None, user_data=None):
        try:
            dpg.hide_item("convert_modal")
        except Exception:
            pass
    try:
        dpg.set_frame_callback(dpg.get_frame_count() + int(delay_frames), _hide_modal)
    except Exception:
        pass

def _finish_conversion_ui(code, error_message=None):
    global _conv_queue, _conv_raw_queue, _conv_worker_thread, _conv_current_job_index, _conv_current_job_total, _conv_current_job_processed
    global _conv_total_jobs, _conv_done_jobs, _conv_current_job_stage, _convert_tmpdir, _conv_duration_ms
    global _conv_job_frame_frac, _conv_job_encode_frac, _conv_last_status_update, _conv_last_logged_processed
    global _conv_state, _conv_state_rev, _conv_ui_applied_rev, _conv_success_autoclose_queued, _conv_render_last_heartbeat_ts
    if code == 0 and not error_message:
        _conv_done_jobs = max(_conv_done_jobs, _conv_total_jobs)
        _conv_current_job_stage = 1.0
        dpg.set_value("conv_progress", 1.0)
        _set_conv_status("Conversion complete. Closing dialog...", force=True)
        _append_to_conv_log("[done] conversion complete")
        _flush_conv_log(force=True)
        try:
            dpg.configure_item("conv_cancel_button", show=False)
            dpg.configure_item("conv_close_button", show=True)
        except Exception:
            pass
        _schedule_convert_modal_autoclose()
    else:
        msg = error_message or f"Conversion finished with exit code {code}"
        _set_conv_status("Conversion error", force=True)
        _append_to_conv_log("[error] " + msg)
        _flush_conv_log(force=True)
        try:
            dpg.configure_item("conv_cancel_button", show=False)
            dpg.configure_item("conv_close_button", show=True)
        except Exception:
            pass
    _cleanup_convert_tmpdir()
    _conv_queue = None
    _conv_raw_queue = None
    _conv_worker_thread = None
    _conv_current_job_index = None
    _conv_current_job_total = None
    _conv_current_job_processed = None
    _conv_total_jobs = 0
    _conv_done_jobs = 0
    _conv_current_job_stage = 0.0
    _conv_duration_ms = None
    _conv_job_frame_frac = 0.0
    _conv_job_encode_frac = 0.0
    _conv_last_status_update = 0.0
    _conv_last_logged_processed = 0
    _conv_render_last_heartbeat_ts = 0.0
    _conv_success_autoclose_queued = False
    with _conv_state_lock:
        _conv_state = None
        _conv_state_rev = 0
        _conv_ui_applied_rev = -1
    rebuild_preview_work_image()
    request_preview_update()

def _poll_converter_queue(user_data=None):
    global _conv_queue, _conv_current_job_index, _conv_current_job_total, _conv_current_job_processed
    global _conv_total_jobs, _conv_done_jobs, _conv_current_job_stage, _conv_duration_ms
    global _conv_job_frame_frac, _conv_job_encode_frac, _conv_last_logged_processed, _conv_poll_last_ts
    global _conv_use_render_tick
    if _conv_queue is None:
        return
    _conv_poll_last_ts = time.monotonic()
    drained = 0
    latest_processed = None
    try:
        while drained < _conv_poll_max_items:
            try:
                item = _conv_queue.get_nowait()
            except queue.Empty:
                break
            drained += 1

            try:
                # structured items from thread
                if isinstance(item, dict) and item.get("type") == "DONE":
                    code = item.get("code", 0)
                    _finish_conversion_ui(code=int(code) if code is not None else 1)
                    return
                if isinstance(item, dict) and item.get("type") == "ERROR":
                    msg = item.get("message", "Unknown error")
                    _finish_conversion_ui(code=1, error_message=msg)
                    return
                # otherwise item is a raw log line (string)
                line = str(item)
                # parse
                kind, payload = _parse_progress_from_line_batch(line)
                should_log_line = True
                if kind == "PROCESSED":
                    try:
                        processed_now = int(payload)
                    except Exception:
                        processed_now = None
                    if processed_now is not None:
                        if (_conv_log_processed_step > 1) and (processed_now % _conv_log_processed_step != 0):
                            should_log_line = False
                        if processed_now > int(_conv_last_logged_processed or 0):
                            _conv_last_logged_processed = processed_now
                elif kind in {"FFMPEG_PROGRESS", "FFMPEG_KV"}:
                    # Keep ffmpeg internals out of live log/status to avoid UI stalls.
                    should_log_line = False

                if should_log_line:
                    _append_to_conv_log(line)

                if kind == "JOB_START":
                    _conv_current_job_index = payload.get("index")
                    _conv_current_job_total = None
                    _conv_current_job_processed = 0
                    _conv_current_job_stage = 0.0
                    _conv_job_frame_frac = 0.0
                    _conv_job_encode_frac = 0.0
                    _conv_last_logged_processed = 0
                    latest_processed = None
                    job_no = int(_conv_current_job_index) + 1 if _conv_current_job_index is not None else "?"
                    total = _conv_total_jobs if _conv_total_jobs else "?"
                    _set_conv_status(f"Job {job_no}/{total} started", force=True)
                    _set_batch_progress()
                elif kind == "TOTAL_FRAMES":
                    _conv_current_job_total = int(payload)
                    _set_live_job_status()
                elif kind == "DURATION_MS":
                    _store_conv_duration_ms(payload)
                elif kind == "PROCESSED":
                    try:
                        p_now = int(payload)
                    except Exception:
                        p_now = None
                    if p_now is not None:
                        latest_processed = p_now if latest_processed is None else max(int(latest_processed), p_now)
                elif kind == "FFMPEG_PROGRESS":
                    # Ignored for status/progress; frame progress drives the UI.
                    pass
                elif kind == "JOB_DONE":
                    idx = payload.get("index")
                    code = payload.get("code")
                    _conv_job_frame_frac = 1.0
                    _conv_job_encode_frac = 1.0 if int(code or 0) == 0 else _conv_job_encode_frac
                    _conv_current_job_stage = 1.0
                    latest_processed = None
                    if idx is not None:
                        _conv_done_jobs = max(_conv_done_jobs, int(idx) + 1)
                    elif _conv_total_jobs:
                        _conv_done_jobs = min(_conv_total_jobs, _conv_done_jobs + 1)
                    _conv_current_job_stage = 0.0
                    _set_batch_progress()
                    _set_conv_status(f"Job {idx} done (exit {code})", force=True)
                    # continue waiting for next job or final DONE
                elif kind == "BATCH_DONE":
                    _conv_done_jobs = max(_conv_done_jobs, _conv_total_jobs)
                    _conv_current_job_stage = 0.0
                    _set_conv_status("Batch finished", force=True)
                    dpg.set_value("conv_progress", 1.0)
                    _append_to_conv_log("[batch done]")
                    _flush_conv_log(force=True)
                    # Hide cancel, show close
                    try:
                        dpg.configure_item("conv_cancel_button", show=False)
                        dpg.configure_item("conv_close_button", show=True)
                    except Exception:
                        pass
                elif kind == "FFMPEG_KV":
                    k, v = payload
                    if k == "duration_ms":
                        _store_conv_duration_ms(v)
                else:
                    # generic log line — if it contains "Processed X/Y" try to parse and set progress
                    m = re.search(r"Processed\s+(\d+)\s*/\s*(\d+)\s*frames", line, re.I)
                    if m:
                        got = int(m.group(1)); tot = int(m.group(2))
                        if tot > 0:
                            pv = max(0.0, min(1.0, float(got) / float(tot)))
                            _conv_job_frame_frac = max(_conv_job_frame_frac, pv)
                            _refresh_current_job_stage()
                            _set_batch_progress()
                            _set_live_job_status()
            except Exception as e:
                _append_to_conv_log(f"[warn] progress UI parse error: {e}")
                continue

        # Apply latest progress updates once per poll tick to avoid UI overload.
        updated = False
        if latest_processed is not None:
            _conv_current_job_processed = int(latest_processed)
            if _conv_current_job_total:
                _conv_job_frame_frac = max(0.0, min(1.0, float(_conv_current_job_processed) / float(_conv_current_job_total)))
            else:
                _conv_job_frame_frac = min(1.0, _conv_job_frame_frac + 0.01)
            updated = True
        if updated:
            _refresh_current_job_stage()
            _set_batch_progress()
            _set_live_job_status()
    except Exception as e:
        _append_to_conv_log(f"[warn] converter poll loop error: {e}")
    finally:
        if (not _conv_use_render_tick) and (_conv_queue is not None):
            try:
                dpg.set_frame_callback(dpg.get_frame_count() + _conv_poll_frame_step, _poll_converter_queue)
            except Exception:
                pass

def _converter_render_tick(sender=None, app_data=None):
    global _conv_render_last_heartbeat_ts, _conv_ui_applied_rev
    now = time.monotonic()
    if _conv_status_pipeline == "v2":
        if (now - float(_conv_render_last_heartbeat_ts)) >= float(_conv_render_heartbeat_min_interval_s):
            snap = _status_snapshot()
            if snap is not None:
                if int(snap.get("rev", -1)) != int(_conv_ui_applied_rev) or (snap.get("log_text") is not None):
                    _apply_status_snapshot_to_ui(snap)
                _conv_ui_applied_rev = max(int(_conv_ui_applied_rev), int(snap.get("rev", -1)))
            _conv_render_last_heartbeat_ts = now
        return
    if _conv_queue is None:
        return
    if (now - float(_conv_poll_last_ts)) >= float(_conv_render_poll_min_interval_s):
        _poll_converter_queue()

def _ui_heartbeat(sender=None, app_data=None):
    global _conv_ui_heartbeat_running, _conv_ui_applied_rev
    if _conv_ui_heartbeat_running:
        try:
            dpg.set_frame_callback(dpg.get_frame_count() + int(_ui_heartbeat_frame_step), _ui_heartbeat)
        except Exception:
            pass
        return
    _conv_ui_heartbeat_running = True
    try:
        if _conv_status_pipeline == "v2":
            snapshot = _status_snapshot()
            if snapshot is not None:
                if int(snapshot.get("rev", -1)) != int(_conv_ui_applied_rev) or (snapshot.get("log_text") is not None):
                    _apply_status_snapshot_to_ui(snapshot)
                _conv_ui_applied_rev = max(int(_conv_ui_applied_rev), int(snapshot.get("rev", -1)))
        else:
            # Legacy v1 watchdog path.
            if _conv_queue is not None:
                now = time.monotonic()
                if (now - float(_conv_poll_last_ts)) >= float(_ui_heartbeat_stale_poll_s):
                    _poll_converter_queue()
    except Exception:
        pass
    finally:
        _conv_ui_heartbeat_running = False
        try:
            dpg.set_frame_callback(dpg.get_frame_count() + int(_ui_heartbeat_frame_step), _ui_heartbeat)
        except Exception:
            pass

# ----------------------------
# Cancel conversion
# ----------------------------
def _cancel_conversion_callback(sender=None, app_data=None):
    global _convert_proc, _convert_cancel_requested, _conv_state_rev
    _convert_cancel_requested = True
    if _conv_status_pipeline == "v2":
        with _conv_state_lock:
            st = _conv_state
            if st is not None:
                _set_status_text_locked(st, "Cancel requested...")
                _append_status_log_locked(st, "[user] cancel requested")
                _conv_state_rev += 1
    else:
        try:
            dpg.set_value("conv_status_text", "Cancel requested...")
            _append_to_conv_log("[user] cancel requested")
            dpg.configure_item("conv_cancel_button", show=False)
        except Exception:
            pass
    try:
        dpg.configure_item("conv_cancel_button", show=False)
    except Exception:
        pass
    if _convert_proc:
        try:
            _convert_proc.terminate()
        except Exception:
            try:
                _convert_proc.kill()
            except Exception:
                pass
    def _force_stop():
        time.sleep(5.0)
        if _conv_status_pipeline == "v2":
            q = _conv_raw_queue
        else:
            q = _conv_queue
        if q is not None:
            _enqueue_status_event(q, {"type": "ERROR", "message": "Cancelled by user (timeout forced)."})
    threading.Thread(target=_force_stop, daemon=True).start()

# ----------------------------
# Spawn convert_clip.py with --batch and stream stdout -> modal via queue
# ----------------------------
def process_queue_and_spawn_batch(sender=None, app_data=None):
    """
    Writes a batch.json describing the queued jobs and launches convert_clip.py --batch <batch.json>
    Streams output into the conversion modal.
    """
    global queue_items, _conv_queue, _conv_raw_queue, _conv_worker_thread, _convert_thread, _convert_proc, _convert_tmpdir, _convert_cancel_requested
    global _conv_total_jobs, _conv_done_jobs, _conv_current_job_stage, _conv_duration_ms
    global _conv_job_frame_frac, _conv_job_encode_frac, _conv_last_status_update, _conv_log_last_ui_push
    global _conv_last_logged_processed, _conv_poll_last_ts, _conv_use_render_tick, _conv_render_last_heartbeat_ts
    global _conv_state, _conv_state_rev, _conv_ui_applied_rev, _conv_success_autoclose_queued
    if not queue_items:
        print("Queue empty — nothing to process.")
        return

    ffmpeg_path = _find_ffmpeg_candidates()
    # create temporary batch JSON
    tmpdir = tempfile.mkdtemp(prefix="aerochrome_batch_")
    _convert_tmpdir = tmpdir
    batch_path = os.path.join(tmpdir, "batch.json")
    # build job list: ensure absolute paths and a "preset" snapshot
    jobs = []
    for job in queue_items:
        j = {
            "input": os.path.abspath(job["input"]),
            "output": os.path.abspath(job["output"]),
            "preset": job["preset"]
        }
        jobs.append(j)
    batch_payload = {"jobs": jobs}
    try:
        mode = dpg.get_value("compute_backend") if dpg.does_item_exist("compute_backend") else "Auto"
    except Exception:
        mode = "Auto"
    if mode == "CPU":
        batch_payload["backend"] = "cpu_ext"
    elif mode == "GPU":
        batch_payload["backend"] = "gpu"
    else:
        batch_payload["backend"] = "auto"
    if ffmpeg_path:
        batch_payload["ffmpeg_path"] = ffmpeg_path
    # write
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(batch_payload, f, indent=2)

    # build command
    cmd, cmd_err = _resolve_converter_command(batch_path, ffmpeg_path=ffmpeg_path)
    if not cmd:
        print(cmd_err or "Error: converter command unavailable.")
        dpg.set_value("conv_status_text", cmd_err or "Error: converter command unavailable.")
        dpg.show_item("convert_modal")
        dpg.set_value("conv_log", cmd_err or "Converter command unavailable.")
        dpg.configure_item("conv_cancel_button", show=False)
        dpg.configure_item("conv_close_button", show=True)
        return

    print("Starting batch conversion:", " ".join(_shlex_quote(x) for x in cmd))

    _convert_cancel_requested = False
    _conv_success_autoclose_queued = False
    _conv_poll_last_ts = time.monotonic()
    _conv_render_last_heartbeat_ts = 0.0
    rebuild_preview_work_image()
    request_preview_update()
    dpg.set_value("conv_log", "")
    dpg.set_value("conv_progress", 0.0)
    if dpg.does_item_exist("conv_total_frames"):
        try: dpg.delete_item("conv_total_frames")
        except Exception: pass
    dpg.show_item("convert_modal")
    try:
        dpg.configure_item("conv_cancel_button", show=True)
        dpg.configure_item("conv_close_button", show=False)
    except Exception:
        pass

    if _conv_status_pipeline == "v2":
        # Initialize new status state pipeline.
        _conv_queue = None
        _conv_raw_queue = queue.Queue(maxsize=int(_conv_raw_queue_maxsize))
        _init_conv_state(len(jobs))
        snap = _status_snapshot()
        if snap is not None:
            _apply_status_snapshot_to_ui(snap)
            _conv_ui_applied_rev = int(snap.get("rev", -1))
        _conv_worker_thread = threading.Thread(target=_status_worker_loop, daemon=True)
        _conv_worker_thread.start()

        def _run_converter_v2():
            global _convert_proc, _conv_raw_queue, _convert_cancel_requested
            q = _conv_raw_queue
            try:
                _convert_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                )
            except FileNotFoundError as e:
                if q is not None:
                    _enqueue_status_event(q, {"type": "ERROR", "message": f"Execution failed: {e}"})
                return
            except Exception as e:
                if q is not None:
                    _enqueue_status_event(q, {"type": "ERROR", "message": f"Failed to spawn converter: {e}"})
                return

            try:
                if _convert_proc.stdout is not None:
                    for raw in _convert_proc.stdout:
                        if raw is None:
                            continue
                        _enqueue_status_line(q, raw)
                        if _convert_cancel_requested:
                            try:
                                _convert_proc.terminate()
                            except Exception:
                                pass
                code = _convert_proc.wait()
                if q is not None:
                    _enqueue_status_event(q, {"type": "DONE", "code": int(code)})
            except Exception as e:
                if q is not None:
                    _enqueue_status_event(q, {"type": "ERROR", "message": str(e)})

        _convert_thread = threading.Thread(target=_run_converter_v2, daemon=True)
        _convert_thread.start()
        _ui_heartbeat()
        return

    # Legacy v1 pipeline (rollback path)
    _conv_queue = queue.Queue()
    _conv_raw_queue = None
    _conv_total_jobs = len(jobs)
    _conv_done_jobs = 0
    _conv_current_job_stage = 0.0
    _conv_duration_ms = None
    _conv_job_frame_frac = 0.0
    _conv_job_encode_frac = 0.0
    _conv_last_status_update = 0.0
    _conv_last_logged_processed = 0
    _conv_poll_last_ts = time.monotonic()
    _conv_log_lines.clear()
    _conv_log_last_ui_push = 0.0
    _set_conv_status(f"Starting batch conversion ({_conv_total_jobs} jobs)...", force=True)

    def _run_converter_v1():
        global _convert_proc, _conv_queue, _convert_cancel_requested
        try:
            _convert_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
        except FileNotFoundError as e:
            try:
                _conv_queue.put({"type": "ERROR", "message": f"Execution failed: {e}"})
            except Exception:
                pass
            return
        except Exception as e:
            try:
                _conv_queue.put({"type": "ERROR", "message": f"Failed to spawn converter: {e}"})
            except Exception:
                pass
            return
        try:
            if _convert_proc.stdout is not None:
                last_ffmpeg_progress = None
                for raw in _convert_proc.stdout:
                    if raw is None:
                        continue
                    line = raw.rstrip("\n")
                    if line.startswith("JOB_START:"):
                        last_ffmpeg_progress = None
                    if line.startswith("FFMPEG_PROGRESS:"):
                        try:
                            frac = float(line.split(":", 1)[1].strip())
                            if last_ffmpeg_progress is not None and frac <= (last_ffmpeg_progress + 1e-6):
                                continue
                            last_ffmpeg_progress = frac
                        except Exception:
                            pass
                    try:
                        _conv_queue.put(line)
                    except Exception:
                        pass
                    if _convert_cancel_requested:
                        try:
                            _convert_proc.terminate()
                        except Exception:
                            pass
            code = _convert_proc.wait()
            try:
                _conv_queue.put({"type": "DONE", "code": int(code)})
            except Exception:
                pass
        except Exception as e:
            try:
                _conv_queue.put({"type": "ERROR", "message": str(e)})
            except Exception:
                pass

    _convert_thread = threading.Thread(target=_run_converter_v1, daemon=True)
    _convert_thread.start()
    if _conv_use_render_tick:
        _poll_converter_queue()
    else:
        try:
            dpg.set_frame_callback(dpg.get_frame_count() + _conv_poll_frame_step, _poll_converter_queue)
        except Exception:
            pass

def _shlex_quote(s):
    return '"' + str(s).replace('"', '\\"') + '"'

# ----------------------------
# UI Construction (video-only) with Queue controls next to Convert Clip
# ----------------------------
dpg.create_context()

with dpg.texture_registry():
    dpg.add_dynamic_texture(width=MAX_PREVIEW_W, height=MAX_PREVIEW_H,
                            default_value=[0.0] * (MAX_PREVIEW_W * MAX_PREVIEW_H * 4),
                            tag="texture")
    dpg.add_dynamic_texture(width=HIST_W, height=HIST_H,
                            default_value=[0.0] * (HIST_W * HIST_H * 4),
                            tag="hist_texture")
    dpg.add_dynamic_texture(width=WAVE_W, height=WAVE_H,
                            default_value=[0.0] * (WAVE_W * WAVE_H * 4),
                            tag="wave_texture")
    for ch in CHANNEL_PREVIEWS:
        dpg.add_dynamic_texture(width=MAX_CH_W, height=MAX_CH_H,
                                default_value=[0.0] * (MAX_CH_W * MAX_CH_H * 4),
                                tag=f"tex_{ch}")
    for ch in CHANNEL_PREVIEWS:
        dpg.add_dynamic_texture(width=MAX_CH_W, height=MAX_CH_H,
                                default_value=[0.0] * (MAX_CH_W * MAX_CH_H * 4),
                                tag=f"tex_before_{ch}")

with dpg.window(label="Py-Chrome (Video)", width=1300, height=850):
    with dpg.group(horizontal=True):
        # Left controls
        with dpg.child_window(tag="left_panel", resizable_x=True, autosize_y=True, border=True,
                              horizontal_scrollbar=True, width=500):
            dpg.add_input_text(label="Video Path", tag="file", width=400, readonly=True)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Open File...", callback=lambda: dpg.show_item("file_dialog"))
                # old single convert replaced by Process Queue (will process queued jobs)
                dpg.add_button(label="Convert Clip (Process Queue)", callback=lambda: process_queue_and_spawn_batch())
            dpg.add_spacer(height=6)
            dpg.add_text("OpenCV not available — video load disabled", tag="cv2_missing_text", show=not CV2_AVAILABLE, color=(200,0,0))
            
            dpg.add_spacer(height=6)

            # Queue listbox
            with dpg.group(horizontal=True):
                dpg.add_text("Queue")
                dpg.add_listbox(items=[], tag="queue_list", width=150, num_items=8)
                
                # queue control buttons
                with dpg.group():
                    dpg.add_button(label="Add to Queue", tag="add_queue_button", callback=add_current_to_queue)
                    dpg.add_button(label="Remove Queue", callback=remove_selected_from_queue)
                    dpg.add_button(label="Move Up", callback=move_selected_up)
                    dpg.add_button(label="Move Down", callback=move_selected_down)
            with dpg.item_handler_registry(tag="queue_list_handlers"):
                dpg.add_item_double_clicked_handler(callback=on_queue_list_double_click)
            dpg.bind_item_handler_registry("queue_list", "queue_list_handlers")

            dpg.add_spacer(height=6)

            # Presets
            with dpg.collapsing_header(label="Presets (folder)", default_open=False):
                with dpg.group(horizontal=True):
                    dpg.add_input_text(label="Preset name", tag="preset_name_input", width=90, default_value="")
                    dpg.add_button(label="Save Preset to folder", callback=lambda s,a: save_preset_to_folder(dpg.get_value("preset_name_input")))
                dpg.add_spacer(height=6)
                with dpg.group(horizontal=True):
                    dpg.add_combo(items=[], label="Presets", tag="preset_combo", width=90, callback=lambda s,a: load_preset_from_folder(dpg.get_value("preset_combo")))
                    dpg.add_button(label="Delete Preset", callback=show_delete_preset_confirm)
                    dpg.add_button(label="Refresh", callback=lambda s,a: refresh_presets_dropdown())

            dpg.add_spacer(height=6)
            with dpg.collapsing_header(label="Graph Scopes", default_open=False, tag="scope_header"):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Histogram", callback=lambda s,a: set_scope_mode("Histogram"))
                    dpg.add_button(label="Waveform", callback=lambda s,a: set_scope_mode("Waveform"))
                    dpg.add_button(label="Vectorscope", callback=lambda s,a: set_scope_mode("Vectorscope"))
                dpg.add_text("Active graph: Histogram", tag="scope_mode_display")
                dpg.add_checkbox(label="Use Converted Data (All Graphs)", tag="analysis_use_converted", default_value=True, callback=request_preview_update)
                dpg.add_slider_int(label="Graph point size", tag="scope_point_size",
                                   default_value=DEFAULT_SCOPE_POINT_SIZE, min_value=1, max_value=6,
                                   callback=on_scope_point_size_changed)
                dpg.add_spacer(height=4)

                with dpg.group(tag="scope_histogram_group", show=True):
                    dpg.add_text(
                        "Solid histogram with overlap blending: IR is RED, Red is GREEN, Green is BLUE.",
                        wrap=430
                    )
                    dpg.add_slider_float(label="Histogram visual gain", tag="hist_gain",
                                         default_value=10.0, min_value=1.0, max_value=200.0,
                                         callback=request_preview_update)
                    dpg.add_image("hist_texture", width=HIST_W, height=HIST_H)

                with dpg.group(tag="scope_waveform_group", show=False):
                    dpg.add_text("Waveform uses additive RGB blending (overlap trends toward white).", wrap=430)
                    dpg.add_image("wave_texture", width=WAVE_W, height=WAVE_H)

                with dpg.group(tag="scope_vectorscope_group", show=False):
                    dpg.add_checkbox(label="Vectorscope x2", tag="vectorscope_x2", default_value=False, callback=on_vectorscope_zoom_changed)
                    with dpg.plot(label="Vectorscope", height=240, width=-1):
                        dpg.add_plot_axis(dpg.mvXAxis, label="Cb", tag="vectorscope_x_axis")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Cr", tag="vectorscope_y_axis")
                        dpg.set_axis_limits("vectorscope_x_axis", -128.0, 128.0)
                        dpg.set_axis_limits("vectorscope_y_axis", -128.0, 128.0)
                        dpg.add_scatter_series([], [], parent="vectorscope_y_axis", tag="vectorscope_series_r")
                        dpg.add_scatter_series([], [], parent="vectorscope_y_axis", tag="vectorscope_series_y")
                        dpg.add_scatter_series([], [], parent="vectorscope_y_axis", tag="vectorscope_series_g")
                        dpg.add_scatter_series([], [], parent="vectorscope_y_axis", tag="vectorscope_series_c")
                        dpg.add_scatter_series([], [], parent="vectorscope_y_axis", tag="vectorscope_series_b")
                        dpg.add_scatter_series([], [], parent="vectorscope_y_axis", tag="vectorscope_series_m")

            dpg.add_spacer(height=6)
            with dpg.collapsing_header(label="LUT Preview (Non-Destructive)", default_open=False):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Load LUT (.cube)", callback=lambda: dpg.show_item("lut_file_dialog"))
                    dpg.add_button(label="Clear LUT", callback=clear_lut_preview)
                dpg.add_checkbox(label="Enable LUT Preview", tag="lut_enable", default_value=False, callback=request_preview_update)
                dpg.add_text("No LUT loaded.", tag="lut_status_text", wrap=430)

            dpg.add_spacer(height=6)
            dpg.add_button(label="Reset to Default", callback=reset_to_defaults)
            dpg.add_spacer(height=6)

            with dpg.collapsing_header(label="White Balance (Temperature + Tint)", default_open=False):
                dpg.add_button(label="Set WB Reference", tag="wb_dropper_btn", callback=toggle_wb_dropper)
                dpg.add_text("Click the button, then click a neutral area in Main Preview to auto-set WB.", wrap=400)
                dpg.add_slider_int(label="WB Temperature (K)", tag="wb_temp", default_value=DEFAULT_PRESET["wb_temp"], min_value=2000, max_value=12000, callback=request_preview_update)
                dpg.add_slider_int(label="WB Tint", tag="wb_tint", default_value=DEFAULT_PRESET["wb_tint"], min_value=-100, max_value=100, callback=request_preview_update)
                dpg.add_text("Temperature = Kelvin. Tint = green<->magenta.", wrap=400)

            dpg.add_spacer(height=6)
            with dpg.collapsing_header(label="Fraction sliders", default_open=False):
                for label, tag, default, minv, maxv in [
                    ("Red Vis Fraction", "fracRx", DEFAULT_PRESET["fracRx"], 0.0, 1.0),
                    ("Green Vis Fraction", "fracGx", DEFAULT_PRESET["fracGx"], 0.0, 1.0),
                    ("IR Fraction", "fracBY", DEFAULT_PRESET["fracBY"], 0.0, 1.0)
                ]:
                    dpg.add_slider_float(label=label, tag=tag, default_value=default, min_value=minv, max_value=maxv, callback=request_preview_update)

            dpg.add_spacer(height=6)
            with dpg.collapsing_header(label="Gamma & Exposure", default_open=False):
                for label, tag, default, minv, maxv in [
                    ("Gamma Red Visible","gammaRx",DEFAULT_PRESET["gammaRx"],0.1,5.0),
                    ("Gamma Red IR","gammaRy",DEFAULT_PRESET["gammaRy"],0.1,5.0),
                    ("Gamma Green Visible","gammaGx",DEFAULT_PRESET["gammaGx"],0.1,5.0),
                    ("Gamma Green IR","gammaGy",DEFAULT_PRESET["gammaGy"],0.1,5.0),
                    ("Gamma IR","gammaBY",DEFAULT_PRESET["gammaBY"],0.1,5.0),
                    ("Exposure","exposure",DEFAULT_PRESET["exposure"],0.1,5.0)
                ]:
                    dpg.add_slider_float(label=label, tag=tag, default_value=default, min_value=minv, max_value=maxv, callback=request_preview_update)

            dpg.add_text("", tag="warning_text", color=(255, 0, 0))
            dpg.add_spacer(height=6)

            with dpg.collapsing_header(label="Scatterplots (Separate)", default_open=False, tag="scatter_header"):
                dpg.add_slider_int(label="Scatter marker size", tag="scatter_marker_size",
                                   default_value=DEFAULT_MARKER_SIZE, min_value=1, max_value=8,
                                   callback=on_marker_size_changed)
                with dpg.group(horizontal=False):
                    with dpg.plot(label="IR vs Red", height=165, width=-1):
                        dpg.add_plot_axis(dpg.mvXAxis, label="IR (0-255)", tag="axis_rg_x")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Red (0-255)", tag="axis_rg_y")
                        dpg.set_axis_limits("axis_rg_x", 0.0, 255.0)
                        dpg.set_axis_limits("axis_rg_y", 0.0, 255.0)
                        dpg.add_scatter_series([], [], parent="axis_rg_y", tag="series_rg")
                    with dpg.plot(label="IR vs Green", height=165, width=-1):
                        dpg.add_plot_axis(dpg.mvXAxis, label="IR (0-255)", tag="axis_rb_x")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Green (0-255)", tag="axis_rb_y")
                        dpg.set_axis_limits("axis_rb_x", 0.0, 255.0)
                        dpg.set_axis_limits("axis_rb_y", 0.0, 255.0)
                        dpg.add_scatter_series([], [], parent="axis_rb_y", tag="series_rb")
                    with dpg.plot(label="Red vs Green", height=165, width=-1):
                        dpg.add_plot_axis(dpg.mvXAxis, label="Red (0-255)", tag="axis_gb_x")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Green (0-255)", tag="axis_gb_y")
                        dpg.set_axis_limits("axis_gb_x", 0.0, 255.0)
                        dpg.set_axis_limits("axis_gb_y", 0.0, 255.0)
                        dpg.add_scatter_series([], [], parent="axis_gb_y", tag="series_gb")

        # file dialog
        with dpg.file_dialog(directory_selector=False, show=False, callback=open_file_callback,
                             tag="file_dialog", width=800, height=500):
            for ext in SUPPORTED_VIDEO_EXTENSIONS:
                dpg.add_file_extension(ext)
        with dpg.file_dialog(directory_selector=False, show=False, callback=open_lut_file_callback,
                             tag="lut_file_dialog", width=700, height=400):
            dpg.add_file_extension(".cube")
            dpg.add_file_extension(".CUBE")

        # Right preview: main preview + timeline container under it
        with dpg.child_window(tag="right_panel", autosize_x=True, autosize_y=True):
            dpg.add_text("Main Preview")
            dpg.add_image("texture", tag="preview_image")
            # timeline holder — slider gets added here (parent="right_preview_controls")
            with dpg.group(tag="right_preview_controls", horizontal=False):
                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        items=list(PREVIEW_QUALITY_MODES),
                        label="Preview Quality",
                        tag="preview_quality",
                        default_value="Balanced",
                        callback=on_preview_quality_changed,
                        width=180
                    )
                    dpg.add_combo(
                        items=list(COMPUTE_BACKEND_MODES),
                        label="Compute",
                        tag="compute_backend",
                        default_value="Auto",
                        callback=on_compute_backend_changed,
                        width=140
                    )
                    dpg.add_combo(
                        items=list(PREVIEW_BIT_DEPTH_MODES),
                        label="Preview Depth",
                        tag="preview_bit_depth",
                        default_value="8-bit",
                        callback=on_preview_bit_depth_changed,
                        width=140
                    )
                dpg.add_text("", tag="compute_status_text")
                dpg.add_text("", tag="right_preview_controls_placeholder", show=False)

            dpg.add_spacer(height=8)
            dpg.add_text("Before Conversion Original / Channels")
            with dpg.group(horizontal=True):
                for ch in CHANNEL_PREVIEWS:
                    with dpg.group():
                        dpg.add_text(f"Before: {ch}")
                        dpg.add_image(f"tex_before_{ch}")

            dpg.add_separator()
            dpg.add_text("After Conversion Converted Channels")
            with dpg.group(horizontal=True):
                for ch in CHANNEL_PREVIEWS:
                    with dpg.group():
                        dpg.add_text(f"After: {ch}")
                        dpg.add_image(f"tex_{ch}")

# Delete preset modal
with dpg.window(label="Confirm delete preset", modal=True, show=False, tag="delete_preset_modal", no_title_bar=False, width=400, height=120):
    dpg.add_text("Are you sure you want to permanently delete this preset?")
    dpg.add_spacer(height=6)
    dpg.add_text("", tag="delete_preset_name_display")
    dpg.add_spacer(height=6)
    dpg.add_text("", tag="preset_to_delete", show=False)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Confirm Delete", callback=confirm_delete_preset)
        dpg.add_button(label="Cancel", callback=lambda s,a: dpg.hide_item("delete_preset_modal"))

# Preset overwrite confirmation modal
with dpg.window(label="Preset exists", modal=True, show=False, tag="overwrite_preset_modal", no_title_bar=False, width=420, height=140):
    dpg.add_text("", tag="overwrite_preset_name_display", wrap=390)
    dpg.add_text("", tag="preset_to_overwrite_name", show=False)
    dpg.add_text("", tag="preset_to_overwrite_path", show=False)
    dpg.add_spacer(height=8)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Overwrite", callback=confirm_overwrite_preset)
        dpg.add_button(label="Cancel", callback=cancel_overwrite_preset)

# Conversion modal: progress + log + cancel
with dpg.window(label="Conversion progress", modal=True, show=False, tag="convert_modal", no_title_bar=False, width=720, height=520):
    dpg.add_text("Conversion status", tag="conv_status_text")
    dpg.add_spacer(height=6)
    dpg.add_progress_bar(tag="conv_progress", default_value=0.0, width=-1)
    dpg.add_spacer(height=6)
    dpg.add_input_text(tag="conv_log", default_value="", multiline=True, readonly=True, height=380, width=-1)
    dpg.add_spacer(height=6)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Cancel", tag="conv_cancel_button", callback=_cancel_conversion_callback)
        dpg.add_button(label="Close", tag="conv_close_button", show=False, callback=lambda s,a: dpg.hide_item("convert_modal"))

# Setup viewport and run
dpg.create_viewport(title="Py-Chrome (Video)", width=1300, height=850)
dpg.setup_dearpygui()
try:
    _conv_use_render_tick = True
    dpg.set_render_callback(_converter_render_tick)
except Exception:
    _conv_use_render_tick = False
_schedule_ui_heartbeat()

with dpg.handler_registry():
    dpg.add_mouse_click_handler(callback=on_preview_image_click)

refresh_presets_dropdown()
refresh_queue_listbox()
rebuild_scatter_themes(dpg.get_value("scatter_marker_size") if dpg.does_item_exist("scatter_marker_size") else DEFAULT_MARKER_SIZE)
rebuild_scope_point_themes(dpg.get_value("scope_point_size") if dpg.does_item_exist("scope_point_size") else DEFAULT_SCOPE_POINT_SIZE)
_apply_vectorscope_zoom()
_set_lut_status()
_set_compute_status()
set_scope_mode(_scope_mode)
dpg.show_viewport()
dpg.start_dearpygui()
_reset_preview_processor()
dpg.destroy_context()
