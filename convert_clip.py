#!/usr/bin/env python3
"""
convert_clip.py - improved progress reporting for Aerochrome converter

Key improvements:
- prints DURATION_MS in job metadata so GUI can compute ffmpeg progress
- during ffmpeg reassembly can emit optional "FFMPEG_PROGRESS" lines for debugging
- still prints PROCESSED: <i> frequently (throttled to reduce UI log pressure)
"""
import os
import sys
import argparse
import shutil
import subprocess
import json
import time
import threading
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

try:
    from pychrome_native import (
        FrameProcessor,
        params_from_preset,
        normalize_backend_request,
    )
except Exception as e:
    print(f"[warn] pychrome_native import failed; using NumPy fallback only: {e}", file=sys.stderr, flush=True)

    def _clamp(x):
        return np.clip(x, 0.0, 1.0)

    def _kelvin_to_rgb(kelvin):
        kelvin = float(np.clip(kelvin, 1000.0, 40000.0))
        tmp = kelvin / 100.0
        if tmp <= 66.0:
            red = 255.0
        else:
            red = 329.698727446 * ((tmp - 60.0) ** -0.1332047592)
        if tmp <= 66.0:
            green = 99.4708025861 * np.log(tmp) - 161.1195681661
        else:
            green = 288.1221695283 * ((tmp - 60.0) ** -0.0755148492)
        if tmp >= 66.0:
            blue = 255.0
        elif tmp <= 19.0:
            blue = 0.0
        else:
            blue = 138.5177312231 * np.log(tmp - 10.0) - 305.0447927307
        return np.array(
            [
                float(np.clip(red, 0.0, 255.0) / 255.0),
                float(np.clip(green, 0.0, 255.0) / 255.0),
                float(np.clip(blue, 0.0, 255.0) / 255.0),
            ],
            dtype=np.float32,
        )

    def _apply_white_balance_numpy(img, temp_kelvin, tint_value):
        src_rgb = _kelvin_to_rgb(temp_kelvin)
        ref_rgb = _kelvin_to_rgb(6500.0)
        gains = ref_rgb / (src_rgb + 1e-8)
        tint_norm = float(np.clip(tint_value, -100.0, 100.0)) / 100.0
        gains[1] *= 1.0 - 0.15 * tint_norm
        return _clamp(np.asarray(img, dtype=np.float32) * gains.reshape((1, 1, 3)))

    def _scientific_irg_transform_numpy(img, fracRx, fracGx, fracBY, gammaRx, gammaRy, gammaGx, gammaGy, gammaBY, exposure):
        arr = np.asarray(img, dtype=np.float32)
        Z1 = arr[:, :, 0]
        Z2 = arr[:, :, 1]
        Z3 = arr[:, :, 2]
        eps = 1e-6
        fracBY = max(float(fracBY), eps)
        fracRx = max(float(fracRx), eps)
        fracGx = max(float(fracGx), eps)
        fracRy = 1.0 - fracRx
        fracGy = 1.0 - fracGx
        innerY = np.clip(1.0 - (Z3 / fracBY), 0.0, 1.0)
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
        return _clamp(out * float(exposure))

    def params_from_preset(preset):
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

    def normalize_backend_request(backend):
        b = str(backend or "auto").strip().lower()
        if b in {"auto", "numpy", "cpu_ext", "metal", "gpu"}:
            return b
        if b == "cupy":
            return "numpy"
        return "auto"

    class FrameProcessor:
        def __init__(self, requested_backend="auto"):
            self.requested_backend = normalize_backend_request(requested_backend)
            self.active_backend = "numpy"

        def close(self):
            return None

        def process(self, rgb, params):
            wb = _apply_white_balance_numpy(rgb, params["wb_temp"], params["wb_tint"])
            out = _scientific_irg_transform_numpy(
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
            self.active_backend = "numpy"
            return np.asarray(out, dtype=np.float32)

PROCESSED_LOG_EVERY_FRAMES = max(1, int(os.getenv("PYCHROME_PROCESSED_LOG_STEP", "8")))
PROCESSED_LOG_MIN_INTERVAL_S = max(0.03, float(os.getenv("PYCHROME_PROCESSED_LOG_MIN_INTERVAL", "0.12")))
EMIT_FFMPEG_PROGRESS = os.getenv("PYCHROME_EMIT_FFMPEG_PROGRESS", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
CONVERTER_TRACE_MODE = str(os.getenv("PYCHROME_CONVERTER_TRACE_MODE", "")).strip().lower()
if CONVERTER_TRACE_MODE not in {"off", "sample", "full"}:
    _legacy_trace = os.getenv("PYCHROME_CONVERTER_TRACE", "0").strip().lower() in {"1", "true", "yes", "on"}
    CONVERTER_TRACE_MODE = "full" if _legacy_trace else "off"
CONVERTER_TRACE_ENABLED = CONVERTER_TRACE_MODE != "off"
CONVERTER_TRACE_PATH = (os.getenv("PYCHROME_CONVERTER_TRACE_PATH", "") or "").strip() or None
CONVERTER_TRACE_PROCESSED_STEP = max(1, int(os.getenv("PYCHROME_CONVERTER_TRACE_PROCESSED_STEP", "200")))
_TRACE_LOCK = threading.Lock()
_TRACE_STARTED = False


def _conv_trace(msg):
    global _TRACE_STARTED
    if not CONVERTER_TRACE_ENABLED or not CONVERTER_TRACE_PATH:
        return
    try:
        p = Path(CONVERTER_TRACE_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        wall = time.time()
        mono = time.monotonic()
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall))
        ms = int((wall - int(wall)) * 1000.0)
        tid = threading.current_thread().name
        text = str(msg)
        if CONVERTER_TRACE_MODE == "sample":
            low = text.lower()
            if text.startswith("processed="):
                pass
            elif not any(
                k in low
                for k in (
                    "start",
                    "done",
                    "error",
                    "exit",
                    "active_backend",
                    "job",
                    "batch",
                    "main args",
                    "main exit",
                )
            ):
                return
        with _TRACE_LOCK:
            with open(p, "a", encoding="utf-8") as f:
                if not _TRACE_STARTED:
                    f.write("PyChromeSuper35 converter trace\n")
                    f.write(f"pid: {os.getpid()}\n")
                    f.write(f"python: {sys.executable}\n")
                    f.write(f"cwd: {os.getcwd()}\n")
                    f.write(f"trace_mode: {CONVERTER_TRACE_MODE}\n")
                    f.write("\n")
                    _TRACE_STARTED = True
                f.write(f"{ts}.{ms:03d} mono={mono:12.3f} th={tid:<18} {text}\n")
    except Exception:
        pass

def _emit_processed_progress(i, state, force=False):
    now = time.monotonic()
    last_i = int(state.get("i", 0))
    last_t = float(state.get("t", 0.0))
    if force or last_i <= 0 or (i - last_i) >= PROCESSED_LOG_EVERY_FRAMES or (now - last_t) >= PROCESSED_LOG_MIN_INTERVAL_S:
        print(f"PROCESSED: {i}", flush=True)
        if force or (i % CONVERTER_TRACE_PROCESSED_STEP == 0):
            _conv_trace(f"processed={i}")
        state["i"] = int(i)
        state["t"] = now

def _find_ffmpeg_candidates(provided_path=None):
    if provided_path:
        if os.path.exists(provided_path) and os.access(provided_path, os.X_OK):
            return provided_path
    p = shutil.which("ffmpeg")
    if p: return p
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

def _find_ffprobe_candidates(ffmpeg_path=None):
    if ffmpeg_path:
        p = Path(ffmpeg_path)
        sibling = p.with_name("ffprobe")
        try:
            if sibling.exists() and os.access(str(sibling), os.X_OK):
                return str(sibling)
        except Exception:
            pass
    p = shutil.which("ffprobe")
    if p:
        return p
    candidates = [
        "/opt/homebrew/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        "/usr/bin/ffprobe",
        "/bin/ffprobe",
        "/snap/bin/ffprobe",
        "/usr/local/opt/ffmpeg/bin/ffprobe",
    ]
    for c in candidates:
        try:
            if c and os.path.exists(c) and os.access(c, os.X_OK):
                return c
        except Exception:
            pass
    return None

def _parse_fraction_to_float(v):
    if not v:
        return None
    if "/" in str(v):
        a, b = str(v).split("/", 1)
        try:
            num = float(a)
            den = float(b)
            if den != 0.0:
                return num / den
            return None
        except Exception:
            return None
    try:
        return float(v)
    except Exception:
        return None

def _infer_bit_depth(pix_fmt, bits_per_raw_sample=None, profile=None):
    try:
        if bits_per_raw_sample is not None and str(bits_per_raw_sample).isdigit():
            b = int(bits_per_raw_sample)
            if b > 0:
                return b
    except Exception:
        pass
    s = (pix_fmt or "").lower()
    if "p16" in s or "16le" in s or "16be" in s:
        return 16
    if "p14" in s:
        return 14
    if "p12" in s:
        return 12
    if "p10" in s or "10le" in s or "10be" in s:
        return 10
    if "rgb48" in s:
        return 16
    p = (profile or "").lower()
    if "10" in p:
        return 10
    return 8

def probe_input_metadata(input_path, ffmpeg_path=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video: " + input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    meta = {
        "frame_count": frame_count,
        "fps": fps if fps > 0 else 25.0,
        "fps_rate": None,
        "width": width,
        "height": height,
        "pix_fmt": None,
        "codec_name": None,
        "profile": None,
        "bit_depth": 8,
        "video_bit_rate": None,
        "color_range": None,
        "color_space": None,
        "color_transfer": None,
        "color_primaries": None,
    }

    ffprobe = _find_ffprobe_candidates(ffmpeg_path=ffmpeg_path)
    if not ffprobe:
        return meta

    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,pix_fmt,bits_per_raw_sample,avg_frame_rate,r_frame_rate,codec_name,profile,bit_rate,color_range,color_space,color_transfer,color_primaries",
        "-of", "json",
        input_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return meta
        payload = json.loads(proc.stdout or "{}")
        streams = payload.get("streams", [])
        if not streams:
            return meta
        s0 = streams[0]
        if s0.get("width"):
            meta["width"] = int(s0["width"])
        if s0.get("height"):
            meta["height"] = int(s0["height"])
        pix_fmt = s0.get("pix_fmt")
        codec_name = s0.get("codec_name")
        profile = s0.get("profile")
        bits_per_raw_sample = s0.get("bits_per_raw_sample")
        vid_br = s0.get("bit_rate")
        meta["pix_fmt"] = pix_fmt
        meta["codec_name"] = codec_name
        meta["profile"] = profile
        try:
            meta["video_bit_rate"] = int(vid_br) if vid_br is not None else None
        except Exception:
            meta["video_bit_rate"] = None
        meta["color_range"] = s0.get("color_range")
        meta["color_space"] = s0.get("color_space")
        meta["color_transfer"] = s0.get("color_transfer")
        meta["color_primaries"] = s0.get("color_primaries")
        rate = s0.get("avg_frame_rate") or s0.get("r_frame_rate")
        rate_f = _parse_fraction_to_float(rate)
        if rate and rate != "0/0":
            meta["fps_rate"] = str(rate)
        if rate_f and rate_f > 0:
            meta["fps"] = float(rate_f)
        meta["bit_depth"] = _infer_bit_depth(pix_fmt, bits_per_raw_sample=bits_per_raw_sample, profile=profile)
    except Exception:
        return meta

    return meta

def _choose_output_encode_settings(meta):
    bit_depth = int(meta.get("bit_depth") or 8)
    src_br = int(meta.get("video_bit_rate") or 0)
    # Prefer visually higher quality than previous defaults.
    # For 4K 10-bit sources this reduces over-compression artifacts.
    if bit_depth >= 10:
        codec_args = ["-preset", "medium", "-crf", "10"]
        if src_br > 0:
            # Keep a sane VBV ceiling near source bitrate for better detail retention.
            maxrate = max(50000000, int(src_br))
            bufsize = maxrate * 2
            codec_args += ["-maxrate", str(maxrate), "-bufsize", str(bufsize)]
        return {
            "input_pix_fmt": "rgb48le",
            "output_pix_fmt": "yuv420p10le",
            "codec": "libx265",
            "codec_args": codec_args,
            "target_bit_depth": 10,
        }
    codec_args = ["-preset", "slow", "-crf", "12"]
    if src_br > 0:
        maxrate = max(20000000, int(src_br))
        bufsize = maxrate * 2
        codec_args += ["-maxrate", str(maxrate), "-bufsize", str(bufsize)]
    return {
        "input_pix_fmt": "rgb24",
        "output_pix_fmt": "yuv420p",
        "codec": "libx264",
        "codec_args": codec_args,
        "target_bit_depth": 8,
    }

def _output_codec_tag(codec_name):
    c = (codec_name or "").lower()
    if c == "libx265" or c == "hevc":
        return "hvc1"
    if c == "libx264" or c == "h264":
        return "avc1"
    return None

def _build_color_args(meta):
    args = []
    cr = meta.get("color_range")
    cs = meta.get("color_space")
    ct = meta.get("color_transfer")
    cp = meta.get("color_primaries")
    if cr in ("tv", "pc"):
        args += ["-color_range", cr]
    if cs:
        args += ["-colorspace", str(cs)]
    if ct:
        args += ["-color_trc", str(ct)]
    if cp:
        args += ["-color_primaries", str(cp)]
    return args

def _read_exact_binary(stream, size):
    if stream is None:
        return None
    remain = int(size)
    if remain <= 0:
        return b""
    parts = []
    while remain > 0:
        chunk = stream.read(remain)
        if not chunk:
            break
        parts.append(chunk)
        remain -= len(chunk)
    if not parts:
        return None
    return b"".join(parts)

def process_and_encode_with_ffmpeg(input_path, preset, output_path, ffmpeg_cmd_path, backend="auto"):
    _conv_trace(f"process_and_encode start input={input_path} output={output_path} backend_req={backend}")
    meta = probe_input_metadata(input_path, ffmpeg_path=ffmpeg_cmd_path)
    frame_count = int(meta.get("frame_count") or 0)
    fps = float(meta.get("fps") or 25.0)
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    fps_rate = meta.get("fps_rate")
    duration_ms = int(round((frame_count / fps) * 1000.0)) if fps > 0 else 0

    print(f"TOTAL_FRAMES: {frame_count}", flush=True)
    print(f"FPS: {fps}", flush=True)
    print(f"WIDTH: {width}", flush=True)
    print(f"HEIGHT: {height}", flush=True)
    print(f"DURATION_MS: {duration_ms}", flush=True)
    print(f"INPUT_BIT_DEPTH: {int(meta.get('bit_depth', 8))}", flush=True)
    if meta.get("pix_fmt"):
        print(f"INPUT_PIX_FMT: {meta['pix_fmt']}", flush=True)

    encode = _choose_output_encode_settings(meta)
    _conv_trace(
        "meta "
        + f"frames={frame_count} fps={fps:.6f} size={width}x{height} "
        + f"input_bit_depth={int(meta.get('bit_depth') or 8)} input_pix_fmt={meta.get('pix_fmt')} "
        + f"output_bit_depth={encode['target_bit_depth']} output_pix_fmt={encode['output_pix_fmt']} codec={encode['codec']}"
    )
    print(f"OUTPUT_BIT_DEPTH: {encode['target_bit_depth']}", flush=True)
    print(f"OUTPUT_PIX_FMT: {encode['output_pix_fmt']}", flush=True)
    out_tag = _output_codec_tag(encode["codec"])
    color_args = _build_color_args(meta)

    if width <= 0 or height <= 0:
        raise RuntimeError("Could not determine input resolution.")

    rate_for_ffmpeg = fps_rate if fps_rate else f"{fps:.6f}"
    enc_cmd = [
        ffmpeg_cmd_path,
        "-y",
        "-hide_banner",
        "-loglevel", "info",
        "-nostdin",
        "-f", "rawvideo",
        "-pix_fmt", encode["input_pix_fmt"],
        "-s:v", f"{width}x{height}",
        "-r", rate_for_ffmpeg,
        "-i", "-",
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", encode["codec"],
        "-pix_fmt", encode["output_pix_fmt"],
    ]
    if out_tag:
        enc_cmd += ["-tag:v", out_tag]
    if color_args:
        enc_cmd += color_args
    enc_cmd += encode["codec_args"] + [
        "-c:a", "copy",
        "-movflags", "+faststart+write_colr",
        "-progress", "pipe:1",
        output_path,
    ]

    dec_cmd = [
        ffmpeg_cmd_path,
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", input_path,
        "-map", "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vsync", "0",
        "-f", "rawvideo",
        "-pix_fmt", encode["input_pix_fmt"],
        "pipe:1",
    ]

    decoder = None
    decoder_stderr = ""

    try:
        proc = subprocess.Popen(
            enc_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
    except Exception:
        raise
    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("ffmpeg did not expose stdin/stdout pipes.")

    def _read_progress():
        ff_out_time_ms = None
        last_frac = -1.0
        for raw in proc.stdout:
            if raw is None:
                continue
            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                line = str(raw).strip()
            if not line:
                continue
            is_kv = ("=" in line)
            if not is_kv:
                print(line, flush=True)
            if line.startswith("out_time_ms="):
                try:
                    ff_out_time_ms = int(line.split("=", 1)[1])
                except Exception:
                    ff_out_time_ms = None
            if EMIT_FFMPEG_PROGRESS and ff_out_time_ms is not None and duration_ms > 0:
                try:
                    frac = min(1.0, max(0.0, float(ff_out_time_ms) / float(duration_ms)))
                    if last_frac < 0.0 or (frac - last_frac) >= 0.002:
                        print(f"FFMPEG_PROGRESS: {frac:.6f}", flush=True)
                        last_frac = frac
                except Exception:
                    pass
            if EMIT_FFMPEG_PROGRESS and line.startswith("progress="):
                progval = line.split("=", 1)[1].strip()
                if progval == "end":
                    print("FFMPEG_PROGRESS: 1.0", flush=True)

    reader = threading.Thread(target=_read_progress, daemon=True)
    reader.start()

    try:
        decoder = subprocess.Popen(
            dec_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except Exception:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=2)
        raise
    if decoder.stdout is None:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=2)
        raise RuntimeError("ffmpeg decoder did not expose stdout pipe.")

    params = params_from_preset(preset)
    backend_req = normalize_backend_request(backend)

    bytes_per_channel = 2 if encode["target_bit_depth"] >= 10 else 1
    frame_bytes = int(width) * int(height) * 3 * bytes_per_channel
    if frame_bytes <= 0:
        raise RuntimeError("Invalid frame size for decode.")
    processor = FrameProcessor(requested_backend=backend_req)
    backend_logged = False

    i = 0
    progress_state = {"i": 0, "t": 0.0}
    loop_error = None
    try:
        while True:
            raw = _read_exact_binary(decoder.stdout, frame_bytes)
            if raw is None:
                break
            if len(raw) != frame_bytes:
                loop_error = RuntimeError(
                    f"Incomplete decoded frame: got {len(raw)} bytes, expected {frame_bytes}."
                )
                break
            if bytes_per_channel == 2:
                rgb = np.frombuffer(raw, dtype="<u2").reshape((height, width, 3)).astype(np.float32) / 65535.0
            else:
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).astype(np.float32) / 255.0
            out = processor.process(rgb, params)
            if not backend_logged:
                print(f"COMPUTE_BACKEND_ACTIVE: {processor.active_backend}", flush=True)
                _conv_trace(f"active_backend={processor.active_backend}")
                backend_logged = True

            if encode["target_bit_depth"] >= 10:
                arr16 = np.round(np.clip(out, 0.0, 1.0) * 65535.0).astype(np.uint16)
                proc.stdin.write(arr16.astype("<u2", copy=False).tobytes())
            else:
                arr8 = np.round(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
                proc.stdin.write(arr8.tobytes())

            i += 1
            _emit_processed_progress(i, progress_state)
    except BrokenPipeError:
        loop_error = RuntimeError("ffmpeg pipe closed unexpectedly during encode.")
    except Exception as e:
        loop_error = e
    finally:
        try:
            if decoder and decoder.stdout:
                decoder.stdout.close()
        except Exception:
            pass
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            processor.close()
        except Exception:
            pass
    decoder_code = 0
    if decoder is not None:
        try:
            decoder_code = decoder.wait()
        except Exception:
            decoder_code = 1
        try:
            if decoder.stderr is not None:
                err_raw = decoder.stderr.read()
                if isinstance(err_raw, bytes):
                    decoder_stderr = err_raw.decode("utf-8", errors="replace")
                elif err_raw is not None:
                    decoder_stderr = str(err_raw)
        except Exception:
            decoder_stderr = ""

    code = proc.wait()
    reader.join(timeout=2.0)
    if loop_error is not None:
        raise loop_error
    if i <= 0:
        err = decoder_stderr.strip()
        if err:
            raise RuntimeError("No frames decoded from input video. ffmpeg decode error: " + err)
        raise RuntimeError("No frames decoded from input video.")
    if decoder_code != 0:
        err = decoder_stderr.strip()
        if err:
            raise RuntimeError(f"ffmpeg decode failed with exit code {decoder_code}: {err}")
        raise RuntimeError(f"ffmpeg decode failed with exit code {decoder_code}")
    if code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {code}")
    if i > 0:
        _emit_processed_progress(i, progress_state, force=True)
    if not backend_logged:
        print(f"COMPUTE_BACKEND_ACTIVE: {processor.active_backend}", flush=True)
        _conv_trace(f"active_backend={processor.active_backend}")
    print(f"Finished processing frames: {i}", flush=True)
    _conv_trace(f"process_and_encode done frames={i} decoder_code={decoder_code} encoder_code={code}")
    return fps, i, (width, height)

def process_job_frames(input_path, preset, tmp_out_dir, backend='auto'):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video: " + input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    # emit metadata the GUI expects (TOTAL_FRAMES + DURATION_MS)
    duration_ms = int(round((frame_count / fps) * 1000.0)) if fps > 0 else 0
    print(f"TOTAL_FRAMES: {frame_count}", flush=True)
    print(f"FPS: {fps}", flush=True)
    print(f"WIDTH: {width}", flush=True)
    print(f"HEIGHT: {height}", flush=True)
    print(f"DURATION_MS: {duration_ms}", flush=True)   # important: GUI can use this for ffmpeg stage

    params = params_from_preset(preset)
    backend_req = normalize_backend_request(backend)
    processor = FrameProcessor(requested_backend=backend_req)
    backend_logged = False

    i = 0
    progress_state = {"i": 0, "t": 0.0}
    start_t = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out = processor.process(rgb, params)
            if not backend_logged:
                print(f"COMPUTE_BACKEND_ACTIVE: {processor.active_backend}", flush=True)
                backend_logged = True
            arr8 = np.round(out * 255.0).astype(np.uint8)
            out_path = os.path.join(tmp_out_dir, f"frame_{i:06d}.png")
            Image.fromarray(arr8).save(out_path)
            i += 1
            _emit_processed_progress(i, progress_state)
            if (i % 50) == 0:
                elapsed = time.time() - start_t
                print(f"Processed {i}/{frame_count} frames (elapsed {elapsed:.1f}s)", flush=True)
    finally:
        cap.release()
        try:
            processor.close()
        except Exception:
            pass
    if i > 0:
        _emit_processed_progress(i, progress_state, force=True)
    if not backend_logged:
        print(f"COMPUTE_BACKEND_ACTIVE: {processor.active_backend}", flush=True)
    print(f"Finished processing frames: {i}", flush=True)
    return fps, i, (width, height)

def reassemble_with_ffmpeg(frames_dir, input_video, fps, output_path, ffmpeg_cmd_path="ffmpeg", duration_ms=None, total_frames=None):
    frames_pattern = os.path.join(frames_dir, "frame_%06d.png")
    cmd = [
        ffmpeg_cmd_path,
        "-y",
        "-hide_banner",
        "-loglevel", "info",
        "-framerate", str(int(round(fps))),
        "-i", frames_pattern,
        "-i", input_video,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "slow",
        "-crf", "18",
        "-c:a", "copy",
        "-progress", "pipe:1",
        output_path
    ]
    # spawn ffmpeg and read its pipe:1 key=value progress output
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
    if proc.stdout is None:
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited {code}")
        return
    # ffmpeg emits key=value lines via -progress; keep stdout drained and logs readable.
    ff_out_time_ms = None
    last_frac = -1.0
    for raw in proc.stdout:
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            print(line, flush=True)
        # parse out_time_ms or frame counts
        if line.startswith("out_time_ms="):
            try:
                ff_out_time_ms = int(line.split("=",1)[1])
            except Exception:
                ff_out_time_ms = None
        if EMIT_FFMPEG_PROGRESS and line.startswith("progress="):
            progval = line.split("=",1)[1].strip()
            if progval == "end":
                print("FFMPEG_PROGRESS: 1.0", flush=True)
        if EMIT_FFMPEG_PROGRESS and ff_out_time_ms is not None and duration_ms:
            try:
                frac = min(1.0, max(0.0, float(ff_out_time_ms) / float(duration_ms)))
                if last_frac < 0.0 or (frac - last_frac) >= 0.002:
                    print(f"FFMPEG_PROGRESS: {frac:.6f}", flush=True)
                    last_frac = frac
            except Exception:
                pass
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {code}")

def write_video_with_opencv(input_path, preset, output_path, backend='numpy'):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video for fallback: " + input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, max(1.0, fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV VideoWriter failed to open.")
    params = params_from_preset(preset)
    backend_req = normalize_backend_request(backend)
    processor = FrameProcessor(requested_backend=backend_req)
    backend_logged = False
    i = 0
    progress_state = {"i": 0, "t": 0.0}
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out = processor.process(rgb, params)
            if not backend_logged:
                print(f"COMPUTE_BACKEND_ACTIVE: {processor.active_backend}", flush=True)
                backend_logged = True
            arr8 = np.round(out * 255.0).astype(np.uint8)
            bgr = cv2.cvtColor(arr8, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            i += 1
            _emit_processed_progress(i, progress_state)
    finally:
        writer.release()
        cap.release()
        try:
            processor.close()
        except Exception:
            pass
    if i > 0:
        _emit_processed_progress(i, progress_state, force=True)
    if not backend_logged:
        print(f"COMPUTE_BACKEND_ACTIVE: {processor.active_backend}", flush=True)
    return fps, i

def run_job(job_index, inp, outp, preset, ffmpeg_path=None, backend='auto'):
    _conv_trace(f"run_job start index={job_index} input={inp} output={outp} ffmpeg_path={ffmpeg_path} backend={backend}")
    print(f"JOB_START: {job_index} {inp} -> {outp}", flush=True)
    if ffmpeg_path:
        try:
            fps, frames_written, size = process_and_encode_with_ffmpeg(
                inp, preset, outp, ffmpeg_cmd_path=ffmpeg_path, backend=backend
            )
            print(f"Frames written: {frames_written}, fps: {fps}, size: {size}", flush=True)
            print(f"JOB_DONE: {job_index} 0", flush=True)
            _conv_trace(f"run_job done index={job_index} ok frames={frames_written}")
        except Exception as e:
            print(f"JOB_DONE: {job_index} 1", flush=True)
            print(f"ERROR: ffmpeg encode path failed: {e}", flush=True)
            _conv_trace(f"run_job error index={job_index} ffmpeg_path error={e}")
            raise
    else:
        try:
            fps_w, frames_written_w = write_video_with_opencv(inp, preset, outp, backend=backend)
            print(f"Frames written: {frames_written_w}, fps: {fps_w}", flush=True)
            print(f"JOB_DONE: {job_index} 0", flush=True)
            _conv_trace(f"run_job done index={job_index} fallback_ok frames={frames_written_w}")
        except Exception as e:
            print(f"JOB_DONE: {job_index} 1", flush=True)
            print(f"ERROR: OpenCV fallback failed: {e}", flush=True)
            _conv_trace(f"run_job error index={job_index} fallback error={e}")
            raise

def run_batch(batch_path, ffmpeg_path=None, backend='auto'):
    _conv_trace(f"run_batch start batch_path={batch_path} ffmpeg_path={ffmpeg_path} backend={backend}")
    with open(batch_path, "r", encoding="utf-8") as f:
        batch = json.load(f)
    jobs = batch.get("jobs", [])
    if not jobs:
        print("No jobs in batch.", flush=True); return 1
    if ffmpeg_path is None and batch.get("ffmpeg_path"):
        ffmpeg_path = batch.get("ffmpeg_path")
    if backend == 'auto' and batch.get("backend"):
        backend = batch.get("backend")
    backend = normalize_backend_request(backend)
    print("BACKEND_REQUESTED: " + backend, flush=True)
    ffmpeg_resolved = _find_ffmpeg_candidates(ffmpeg_path)
    _conv_trace(f"run_batch resolved ffmpeg={ffmpeg_resolved} jobs={len(jobs)} backend={backend}")
    if ffmpeg_resolved: print("FFMPEG: found -> " + str(ffmpeg_resolved), flush=True)
    else: print("FFMPEG: not found; will fallback to OpenCV (audio dropped)", flush=True)
    for idx, j in enumerate(jobs):
        inp = j.get("input"); outp = j.get("output"); preset = j.get("preset", {})
        if not inp or not outp:
            print(f"JOB_DONE: {idx} 1", flush=True)
            print(f"ERROR: job {idx} missing input/output", flush=True)
            continue
        try:
            run_job(idx, inp, outp, preset, ffmpeg_path=ffmpeg_resolved, backend=backend)
        except Exception as e:
            print(f"JOB_DONE: {idx} 1", flush=True)
            print(f"ERROR: job {idx} failed: {e}", flush=True)
            _conv_trace(f"run_batch job_error index={idx} error={e}")
            continue
    print("BATCH_DONE", flush=True)
    _conv_trace("run_batch done")
    return 0

def run_single_mode(input_path, preset_path, output_path, ffmpeg_path=None, backend='auto'):
    _conv_trace(
        "run_single start "
        + f"input={input_path} preset={preset_path} output={output_path} ffmpeg_path={ffmpeg_path} backend={backend}"
    )
    if not os.path.exists(input_path): print("Input not found: " + input_path, flush=True); return 2
    if not os.path.exists(preset_path): print("Preset not found: " + preset_path, flush=True); return 2
    with open(preset_path, "r", encoding="utf-8") as f: preset = json.load(f)
    ffmpeg_resolved = _find_ffmpeg_candidates(ffmpeg_path)
    backend = normalize_backend_request(backend)
    print("BACKEND_REQUESTED: " + backend, flush=True)
    try:
        run_job(0, input_path, output_path, preset, ffmpeg_path=ffmpeg_resolved, backend=backend)
        _conv_trace("run_single done")
        return 0
    except Exception as e:
        _conv_trace(f"run_single error={e}")
        print("Conversion failed: " + str(e), flush=True); return 1

def main():
    ap = argparse.ArgumentParser(description="convert_clip.py - improved progress reporting")
    ap.add_argument("--batch", required=False, help="Batch JSON file")
    ap.add_argument("--input", required=False, help="Single input video")
    ap.add_argument("--preset", required=False, help="Single preset JSON for single job")
    ap.add_argument("--output", required=False, help="Single job output file")
    ap.add_argument("--ffmpeg-path", required=False, help="Explicit path to ffmpeg binary (optional)")
    ap.add_argument(
        "--backend",
        required=False,
        default="auto",
        choices=["auto", "gpu", "numpy", "cpu_ext", "metal", "cupy"],
        help="Processing backend (gpu prefers Metal then CuPy; cupy is explicit CuPy request)",
    )
    args = ap.parse_args()
    _conv_trace(f"main args={sys.argv[1:]}")

    if args.batch:
        if not os.path.exists(args.batch): print("Batch JSON not found: " + args.batch, flush=True); sys.exit(2)
        code = run_batch(args.batch, ffmpeg_path=args.ffmpeg_path, backend=args.backend)
        _conv_trace(f"main exit code={code}")
        sys.exit(code)
    else:
        if not (args.input and args.preset and args.output):
            print("Single-mode requires --input --preset --output (or use --batch)", flush=True); ap.print_help(); sys.exit(2)
        code = run_single_mode(args.input, args.preset, args.output, ffmpeg_path=args.ffmpeg_path, backend=args.backend)
        _conv_trace(f"main exit code={code}")
        sys.exit(code)

if __name__ == "__main__":
    main()
