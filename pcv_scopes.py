from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def sample_points(xs, ys, limit):
    xs = np.asarray(xs, dtype=np.float32).reshape(-1)
    ys = np.asarray(ys, dtype=np.float32).reshape(-1)
    n = int(min(xs.size, ys.size))
    if n <= 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    if n <= int(limit):
        return xs[:n], ys[:n]
    idx = np.random.choice(n, int(limit), replace=False)
    return xs[idx], ys[idx]


def _hist_counts_bincount(values, bins):
    vals = np.asarray(values).ravel()
    if vals.size == 0:
        return np.zeros((bins,), dtype=np.int64)
    vals = np.clip(vals, 0.0, 1.0)
    inds = np.floor(vals * bins).astype(np.int64)
    inds[inds >= bins] = bins - 1
    return np.bincount(inds, minlength=bins)[:bins]


def build_histogram_texture(src_img, *, bins, height, gain):
    if src_img is None:
        return np.zeros((height, bins, 4), dtype=np.float32)

    arr = np.asarray(src_img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return np.zeros((height, bins, 4), dtype=np.float32)

    flat = np.clip(arr.reshape(-1, arr.shape[2])[:, :3], 0.0, 1.0)
    if flat.size == 0:
        return np.zeros((height, bins, 4), dtype=np.float32)

    total_pixels = float(flat.shape[0])
    if total_pixels <= 0:
        return np.zeros((height, bins, 4), dtype=np.float32)

    c_r = _hist_counts_bincount(flat[:, 0], bins)
    c_g = _hist_counts_bincount(flat[:, 1], bins)
    c_b = _hist_counts_bincount(flat[:, 2], bins)

    h_r = c_r.astype(np.float32) / total_pixels
    h_g = c_g.astype(np.float32) / total_pixels
    h_b = c_b.astype(np.float32) / total_pixels

    gain = float(gain) if float(gain) > 0.0 else 1.0

    rh = np.rint(h_r * gain * (height - 1)).astype(np.int32)
    gh = np.rint(h_g * gain * (height - 1)).astype(np.int32)
    bh = np.rint(h_b * gain * (height - 1)).astype(np.int32)

    rh[(c_r > 0) & (rh == 0)] = 1
    gh[(c_g > 0) & (gh == 0)] = 1
    bh[(c_b > 0) & (bh == 0)] = 1

    h_r = np.clip(rh, 0, height - 1)
    h_g = np.clip(gh, 0, height - 1)
    h_b = np.clip(bh, 0, height - 1)

    img = np.zeros((height, bins, 4), dtype=np.float32)
    rows = np.arange(height, dtype=np.int32)[:, None]
    img[:, :, 0] = (rows >= (height - h_r[None, :])).astype(np.float32)
    img[:, :, 1] = (rows >= (height - h_g[None, :])).astype(np.float32)
    img[:, :, 2] = (rows >= (height - h_b[None, :])).astype(np.float32)
    img[:, :, 3] = np.max(img[:, :, :3], axis=2)
    return img


def build_waveform_texture(src_img, *, width, height, point_size):
    if src_img is None:
        return np.zeros((height, width, 4), dtype=np.float32)

    arr = np.asarray(src_img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return np.zeros((height, width, 4), dtype=np.float32)

    h, w = arr.shape[:2]
    step_x = max(1, int(w / width))
    step_y = max(1, int(h / max(120, int(height / 2))))
    small = np.clip(arr[::step_y, ::step_x, :], 0.0, 1.0)
    sh, sw = small.shape[:2]
    if sh <= 0 or sw <= 0:
        return np.zeros((height, width, 4), dtype=np.float32)

    x_idx = np.linspace(0, width - 1, sw, dtype=np.int32)
    x_grid = np.broadcast_to(x_idx[None, :], (sh, sw))

    tex = np.zeros((height, width, 4), dtype=np.float32)
    ps = max(1, int(point_size))
    blur_k = max(1, (ps * 2) - 1)
    if blur_k % 2 == 0:
        blur_k += 1

    for ch in range(3):
        y = np.clip(np.rint((1.0 - small[:, :, ch]) * (height - 1)).astype(np.int32), 0, height - 1)
        plane = np.zeros((height, width), dtype=np.float32)
        np.add.at(plane, (y.ravel(), x_grid.ravel()), 1.0)
        if blur_k > 1 and cv2 is not None:
            plane = cv2.GaussianBlur(plane, (blur_k, blur_k), 0.0)
        mx = float(np.max(plane))
        if mx > 0.0:
            plane = np.log1p(plane) / math.log1p(mx)
        tex[:, :, ch] = np.clip(plane, 0.0, 1.0)

    tex[:, :, :3] = np.clip(np.power(tex[:, :, :3], 0.78) * 1.25, 0.0, 1.0)
    tex[:, :, 3] = np.max(tex[:, :, :3], axis=2)
    return tex


def compute_vectorscope_series(src_img, *, max_scope_points):
    series = {
        "vectorscope_series_r": (np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)),
        "vectorscope_series_y": (np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)),
        "vectorscope_series_g": (np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)),
        "vectorscope_series_c": (np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)),
        "vectorscope_series_b": (np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)),
        "vectorscope_series_m": (np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)),
    }

    if src_img is None or cv2 is None:
        return series

    arr = np.asarray(src_img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return series

    h, w = arr.shape[:2]
    step_x = max(1, int(w / 280))
    step_y = max(1, int(h / 160))
    s = np.clip(arr[::step_y, ::step_x, :], 0.0, 1.0)

    r = s[:, :, 0]
    g = s[:, :, 1]
    b = s[:, :, 2]

    cb = np.clip(((-0.168736 * r) - (0.331264 * g) + (0.5 * b)) * 255.0, -128.0, 128.0)
    cr = np.clip(((0.5 * r) - (0.418688 * g) - (0.081312 * b)) * 255.0, -128.0, 128.0)

    hsv = cv2.cvtColor((s * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].reshape(-1)
    xs = cb.reshape(-1)
    ys = cr.reshape(-1)

    sector_specs = [
        ("vectorscope_series_r", (hue < 15) | (hue >= 165)),
        ("vectorscope_series_y", (hue >= 15) & (hue < 45)),
        ("vectorscope_series_g", (hue >= 45) & (hue < 75)),
        ("vectorscope_series_c", (hue >= 75) & (hue < 105)),
        ("vectorscope_series_b", (hue >= 105) & (hue < 135)),
        ("vectorscope_series_m", (hue >= 135) & (hue < 165)),
    ]

    per_sector_limit = max(1000, int(max_scope_points / 6))
    for tag, mask in sector_specs:
        sx = xs[mask]
        sy = ys[mask]
        series[tag] = sample_points(sx, sy, limit=per_sector_limit)

    return series
