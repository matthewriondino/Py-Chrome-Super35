from __future__ import annotations

import math

import numpy as np

NEUTRAL_KELVIN = 6500.0


def clamp(x):
    return np.clip(x, 0.0, 1.0)


def kelvin_to_rgb(kelvin):
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

    def _cl(v):
        return float(np.clip(v, 0.0, 255.0) / 255.0)

    return np.array([_cl(red), _cl(green), _cl(blue)], dtype=np.float32)


def apply_white_balance(img, temp_kelvin, tint_value, neutral_kelvin=NEUTRAL_KELVIN):
    if img is None:
        return None
    src_rgb = kelvin_to_rgb(temp_kelvin)
    ref_rgb = kelvin_to_rgb(neutral_kelvin)
    eps = 1e-8
    gains = ref_rgb / (src_rgb + eps)
    tint_norm = float(np.clip(tint_value, -100.0, 100.0)) / 100.0
    g_tint_multiplier = 1.0 - 0.15 * tint_norm
    gains[1] *= g_tint_multiplier
    out = np.asarray(img, dtype=np.float32) * gains.reshape((1, 1, 3))
    return clamp(out)


def find_wb_from_gray_sample(rgb_sample, neutral_kelvin=NEUTRAL_KELVIN):
    avg = (rgb_sample[0] + rgb_sample[1] + rgb_sample[2]) / 3.0
    if avg < 0.001:
        return 6500, 0
    desired_gains = np.array(
        [
            avg / (rgb_sample[0] + 1e-8),
            avg / (rgb_sample[1] + 1e-8),
            avg / (rgb_sample[2] + 1e-8),
        ],
        dtype=np.float32,
    )
    ref_rgb = kelvin_to_rgb(neutral_kelvin)
    target_rgb = ref_rgb / (desired_gains + 1e-8)

    best_temp = 6500
    min_error = float("inf")
    for temp in range(2000, 12001, 200):
        color = kelvin_to_rgb(temp)
        error = np.linalg.norm(color - target_rgb)
        if error < min_error:
            min_error = error
            best_temp = temp

    for temp in range(max(2000, best_temp - 200), min(12001, best_temp + 200), 20):
        color = kelvin_to_rgb(temp)
        error = np.linalg.norm(color - target_rgb)
        if error < min_error:
            min_error = error
            best_temp = temp

    temp_gains = ref_rgb / (kelvin_to_rgb(best_temp) + 1e-8)
    green_adjustment_needed = desired_gains[1] / (temp_gains[1] + 1e-8)
    tint_norm = (1.0 - green_adjustment_needed) / 0.15
    tint_value = int(np.clip(tint_norm * 100, -100, 100))
    return best_temp, tint_value


def scientific_irg_transform(
    img,
    *,
    fracRx,
    fracGx,
    fracBy,
    gammaRx,
    gammaRy,
    gammaGx,
    gammaGy,
    gammaBy,
    exposure,
):
    arr = np.asarray(img, dtype=np.float32)
    Z1, Z2, Z3 = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    fracRy = 1.0 - float(fracRx)
    fracGy = 1.0 - float(fracGx)

    eps = 1e-6
    fracBy = max(float(fracBy), eps)
    fracRx = max(float(fracRx), eps)
    fracGx = max(float(fracGx), eps)

    innerY = np.clip(1.0 - (Z3 / fracBy), 0.0, 1.0)
    Y = 1.0 - innerY ** (1.0 / float(gammaBy))

    tmp1 = (1.0 - Y) ** float(gammaRy)
    termR = fracRy * (1.0 - tmp1)
    innerX1 = np.clip(1.0 - ((Z1 - termR) / fracRx), 0.0, 1.0)
    X1 = 1.0 - innerX1 ** (1.0 / float(gammaRx))

    tmp2 = (1.0 - Y) ** float(gammaGy)
    termG = fracGy * (1.0 - tmp2)
    innerX2 = np.clip(1.0 - ((Z2 - termG) / fracGx), 0.0, 1.0)
    X2 = 1.0 - innerX2 ** (1.0 / float(gammaGx))

    out = np.dstack([clamp(Y), clamp(X1), clamp(X2)])
    return clamp(out * float(exposure))
