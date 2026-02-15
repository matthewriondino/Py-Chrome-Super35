from __future__ import annotations

import re

import numpy as np


def parse_cube_lut_file(path):
    title = ""
    size = None
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    entries = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            up = line.upper()
            if up.startswith("TITLE"):
                m = re.search(r'"(.*)"', line)
                title = m.group(1) if m else line.split(None, 1)[-1]
                continue
            if up.startswith("LUT_1D_SIZE"):
                continue
            if up.startswith("LUT_3D_SIZE"):
                parts = line.split()
                if len(parts) >= 2:
                    size = int(parts[1])
                continue
            if up.startswith("DOMAIN_MIN"):
                parts = line.split()
                if len(parts) >= 4:
                    domain_min = np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])],
                        dtype=np.float32,
                    )
                continue
            if up.startswith("DOMAIN_MAX"):
                parts = line.split()
                if len(parts) >= 4:
                    domain_max = np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])],
                        dtype=np.float32,
                    )
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    entries.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except Exception:
                    pass

    if size is None or size < 2:
        raise ValueError("LUT_3D_SIZE missing or invalid in .cube file.")

    expected = size * size * size
    if len(entries) < expected:
        raise ValueError(f"LUT data incomplete: expected {expected}, got {len(entries)}.")

    flat = np.asarray(entries[:expected], dtype=np.float32)
    return title, size, flat, domain_min, domain_max


def _lut_flat_lookup_indices(r, g, b, size):
    return (r + (g * size) + (b * size * size)).astype(np.int64)


def apply_lut_preview(img, *, enabled, lut_flat, lut_size, domain_min, domain_max):
    if img is None or lut_flat is None or int(lut_size) <= 1:
        return img
    if not bool(enabled):
        return img

    arr = np.asarray(img, dtype=np.float32)
    size = int(lut_size)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr

    dom_min = np.asarray(domain_min, dtype=np.float32).reshape((1, 1, 3))
    dom_max = np.asarray(domain_max, dtype=np.float32).reshape((1, 1, 3))
    denom = np.maximum(dom_max - dom_min, 1e-6)

    norm = np.clip((arr[:, :, :3] - dom_min) / denom, 0.0, 1.0)
    p = norm * float(size - 1)
    i0 = np.floor(p).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, size - 1)
    f = p - i0.astype(np.float32)

    fr = f[:, :, 0]
    fg = f[:, :, 1]
    fb = f[:, :, 2]

    r0, g0, b0 = i0[:, :, 0], i0[:, :, 1], i0[:, :, 2]
    r1, g1, b1 = i1[:, :, 0], i1[:, :, 1], i1[:, :, 2]

    flat = np.asarray(lut_flat, dtype=np.float32)

    c000 = flat[_lut_flat_lookup_indices(r0, g0, b0, size)]
    c100 = flat[_lut_flat_lookup_indices(r1, g0, b0, size)]
    c010 = flat[_lut_flat_lookup_indices(r0, g1, b0, size)]
    c110 = flat[_lut_flat_lookup_indices(r1, g1, b0, size)]
    c001 = flat[_lut_flat_lookup_indices(r0, g0, b1, size)]
    c101 = flat[_lut_flat_lookup_indices(r1, g0, b1, size)]
    c011 = flat[_lut_flat_lookup_indices(r0, g1, b1, size)]
    c111 = flat[_lut_flat_lookup_indices(r1, g1, b1, size)]

    c00 = c000 * (1.0 - fr)[..., None] + c100 * fr[..., None]
    c10 = c010 * (1.0 - fr)[..., None] + c110 * fr[..., None]
    c01 = c001 * (1.0 - fr)[..., None] + c101 * fr[..., None]
    c11 = c011 * (1.0 - fr)[..., None] + c111 * fr[..., None]
    c0 = c00 * (1.0 - fg)[..., None] + c10 * fg[..., None]
    c1 = c01 * (1.0 - fg)[..., None] + c11 * fg[..., None]
    lut_rgb = c0 * (1.0 - fb)[..., None] + c1 * fb[..., None]

    out = arr.copy()
    out[:, :, :3] = np.clip(lut_rgb, 0.0, 1.0)
    return out
