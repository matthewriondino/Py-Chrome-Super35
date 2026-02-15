#!/usr/bin/env python3
"""Quick backend parity + throughput check for Py-Chrome native processors."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np

try:
    from pychrome_native import FrameProcessor, params_from_preset
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pychrome_native import FrameProcessor, params_from_preset


def run_once(proc: FrameProcessor, frame: np.ndarray, params: dict, iters: int) -> tuple[np.ndarray, float]:
    out = None
    t0 = time.perf_counter()
    for _ in range(max(1, int(iters))):
        out = proc.process(frame, params)
    dt = max(1e-9, time.perf_counter() - t0)
    return out, dt


def main():
    ap = argparse.ArgumentParser(description="Benchmark and verify Py-Chrome backends")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backends", default="numpy,cpu_ext,metal,auto", help="Comma-separated backends")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    frame = rng.random((args.height, args.width, 3), dtype=np.float32)
    params = params_from_preset({
        "wb_temp": 7300,
        "wb_tint": 14,
        "fracRx": 0.66,
        "fracGx": 0.79,
        "fracBY": 0.94,
        "gammaRx": 1.15,
        "gammaRy": 0.95,
        "gammaGx": 1.08,
        "gammaGy": 1.18,
        "gammaBY": 0.92,
        "exposure": 1.12,
    })

    names = [b.strip() for b in args.backends.split(",") if b.strip()]
    if not names:
        names = ["numpy", "cpu_ext", "metal", "auto"]

    baseline = None

    print(f"Frame: {args.width}x{args.height}, iters={args.iters}")
    for name in names:
        proc = FrameProcessor(name)
        try:
            out, dt = run_once(proc, frame, params, args.iters)
            active = proc.active_backend
            fps = float(args.iters) / dt
            if baseline is None:
                baseline = out
                err = 0.0
            else:
                err = float(np.max(np.abs(out - baseline)))
            print(f"{name:>7} -> active={active:>7}  fps={fps:8.2f}  max_abs_err={err:.8f}")
        except Exception as exc:
            print(f"{name:>7} -> ERROR: {exc}")
        finally:
            proc.close()


if __name__ == "__main__":
    main()
