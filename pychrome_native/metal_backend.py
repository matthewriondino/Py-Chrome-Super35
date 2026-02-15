"""Native Metal backend via a long-lived Swift worker subprocess."""

from __future__ import annotations

import json
import os
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class MetalBackendError(RuntimeError):
    pass


def _readable_error(stderr_tail: str) -> str:
    msg = stderr_tail.strip()
    if not msg:
        return ""
    if len(msg) > 800:
        return msg[-800:]
    return msg


def metal_backend_runtime_available() -> bool:
    if sys.platform != "darwin":
        return False
    if shutil.which("swiftc") is None:
        return False
    src = Path(__file__).resolve().parent.parent / "native" / "metal_worker.swift"
    return src.exists()


class MetalBackend:
    def __init__(self, source_path: Optional[str] = None, frame_timeout_s: float = 8.0, control_timeout_s: float = 4.0):
        if source_path is None:
            source_path = str(Path(__file__).resolve().parent.parent / "native" / "metal_worker.swift")
        self.source_path = Path(source_path)
        self.frame_timeout_s = float(frame_timeout_s)
        self.control_timeout_s = float(control_timeout_s)

        self._binary_path: Optional[Path] = None
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_lines = []
        self._stderr_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None
        self._width = None
        self._height = None
        self._last_params: Optional[Dict[str, float]] = None

    def _compile_worker_if_needed(self) -> Path:
        if not metal_backend_runtime_available():
            raise MetalBackendError("Metal runtime prerequisites missing (swiftc/source).")
        if not self.source_path.exists():
            raise MetalBackendError(f"Metal worker source missing: {self.source_path}")

        swiftc = shutil.which("swiftc")
        if not swiftc:
            raise MetalBackendError("swiftc not found in PATH.")

        cache_dir = Path(tempfile.gettempdir()) / "pychrome_native"
        cache_dir.mkdir(parents=True, exist_ok=True)
        module_cache_dir = cache_dir / "swift_module_cache"
        module_cache_dir.mkdir(parents=True, exist_ok=True)
        mtime = int(self.source_path.stat().st_mtime_ns)
        binary = cache_dir / f"metal_worker_{mtime}"

        if not binary.exists():
            # Remove stale worker binaries to keep cache tidy.
            for stale in cache_dir.glob("metal_worker_*"):
                if stale != binary:
                    try:
                        stale.unlink()
                    except Exception:
                        pass

            cmd = [
                swiftc,
                "-O",
                str(self.source_path),
                "-module-cache-path",
                str(module_cache_dir),
                "-o",
                str(binary),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                err = (proc.stderr or "").strip()
                out = (proc.stdout or "").strip()
                detail = err or out or f"swiftc exit {proc.returncode}"
                raise MetalBackendError(f"Failed to compile Metal worker: {detail}")
            try:
                os.chmod(binary, 0o755)
            except Exception:
                pass
        return binary

    def _stderr_reader(self):
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for raw in proc.stderr:
                if raw is None:
                    continue
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
                line = line.rstrip("\n")
                with self._stderr_lock:
                    self._stderr_lines.append(line)
                    if len(self._stderr_lines) > 120:
                        self._stderr_lines = self._stderr_lines[-120:]
        except Exception:
            return

    def _stderr_tail(self) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr_lines[-20:])

    def _read_exact(self, count: int, timeout_s: float) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            raise MetalBackendError("Metal worker is not running.")

        fd = self._proc.stdout.fileno()
        remain = int(count)
        parts = []
        deadline = time.monotonic() + float(timeout_s)

        while remain > 0:
            wait = deadline - time.monotonic()
            if wait <= 0.0:
                raise TimeoutError("Timed out reading from Metal worker.")
            ready, _, _ = select.select([fd], [], [], wait)
            if not ready:
                raise TimeoutError("Timed out reading from Metal worker.")
            chunk = os.read(fd, remain)
            if not chunk:
                tail = _readable_error(self._stderr_tail())
                if tail:
                    raise MetalBackendError(f"Metal worker closed pipe.\n{tail}")
                raise MetalBackendError("Metal worker closed pipe unexpectedly.")
            parts.append(chunk)
            remain -= len(chunk)

        return b"".join(parts)

    def _send_message(self, header: Dict, payload: Optional[bytes] = None):
        if self._proc is None or self._proc.stdin is None:
            raise MetalBackendError("Metal worker is not running.")
        body = payload or b""
        msg = dict(header)
        msg["payload_bytes"] = len(body)
        raw_header = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        if len(raw_header) > (16 * 1024 * 1024):
            raise MetalBackendError("Metal control message too large.")
        try:
            self._proc.stdin.write(struct.pack("<I", len(raw_header)))
            self._proc.stdin.write(raw_header)
            if body:
                self._proc.stdin.write(body)
            self._proc.stdin.flush()
        except BrokenPipeError:
            tail = _readable_error(self._stderr_tail())
            if tail:
                raise MetalBackendError(f"Metal worker broken pipe.\n{tail}")
            raise MetalBackendError("Metal worker broken pipe.")

    def _recv_message(self, timeout_s: float) -> Tuple[Dict, bytes]:
        raw_len = self._read_exact(4, timeout_s=timeout_s)
        header_len = struct.unpack("<I", raw_len)[0]
        if header_len <= 0 or header_len > (16 * 1024 * 1024):
            raise MetalBackendError(f"Invalid Metal message length: {header_len}")
        raw_header = self._read_exact(int(header_len), timeout_s=timeout_s)
        try:
            header = json.loads(raw_header.decode("utf-8"))
        except Exception as exc:
            raise MetalBackendError(f"Invalid Metal response JSON: {exc}") from exc

        payload_len = int(header.get("payload_bytes", 0) or 0)
        if payload_len < 0:
            payload_len = 0
        payload = self._read_exact(payload_len, timeout_s=timeout_s) if payload_len else b""

        if not bool(header.get("ok", False)):
            err = header.get("error") or "Unknown Metal worker error"
            tail = _readable_error(self._stderr_tail())
            if tail:
                err = f"{err}\n{tail}"
            raise MetalBackendError(str(err))

        return header, payload

    def _request(self, cmd: str, timeout_s: float, payload: Optional[bytes] = None, **fields) -> Tuple[Dict, bytes]:
        msg = {"cmd": cmd}
        msg.update(fields)
        self._send_message(msg, payload=payload)
        return self._recv_message(timeout_s=timeout_s)

    def start(self, width: int, height: int):
        if self._proc is not None:
            return
        self._binary_path = self._compile_worker_if_needed()
        self._proc = subprocess.Popen(
            [str(self._binary_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
        self._stderr_thread.start()

        try:
            self._request("init", timeout_s=self.control_timeout_s, width=int(width), height=int(height))
        except Exception:
            self.close(force=True)
            raise

        self._width = int(width)
        self._height = int(height)

    def ensure_dimensions(self, width: int, height: int):
        width = int(width)
        height = int(height)
        if self._proc is None:
            self.start(width=width, height=height)
            return
        if self._width == width and self._height == height:
            return
        self._request("init", timeout_s=self.control_timeout_s, width=width, height=height)
        self._width = width
        self._height = height
        if self._last_params:
            self._request("set_params", timeout_s=self.control_timeout_s, **self._last_params)

    def set_params(
        self,
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
    ):
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
        if self._proc is None:
            raise MetalBackendError("Metal worker is not running.")
        if self._last_params == params:
            return
        self._request("set_params", timeout_s=self.control_timeout_s, **params)
        self._last_params = params

    def process_frame(self, frame_rgb_float32: np.ndarray) -> np.ndarray:
        if self._proc is None:
            raise MetalBackendError("Metal worker is not running.")
        arr = np.asarray(frame_rgb_float32, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise MetalBackendError("Expected frame shape [H, W, 3].")
        arr = arr[:, :, :3]
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr, dtype=np.float32)

        h, w = int(arr.shape[0]), int(arr.shape[1])
        self.ensure_dimensions(width=w, height=h)

        payload = arr.tobytes(order="C")
        _, out_payload = self._request(
            "process_frame",
            timeout_s=self.frame_timeout_s,
            payload=payload,
            width=w,
            height=h,
        )
        expected = h * w * 3 * 4
        if len(out_payload) != expected:
            raise MetalBackendError(f"Metal output size mismatch: got {len(out_payload)}, expected {expected}")
        out = np.frombuffer(out_payload, dtype=np.float32).reshape((h, w, 3)).copy()
        return out

    def close(self, force: bool = False):
        proc = self._proc
        self._proc = None
        self._width = None
        self._height = None
        self._last_params = None
        if proc is None:
            return

        try:
            if not force and proc.poll() is None:
                try:
                    self._send_message({"cmd": "close"}, payload=None)
                    self._recv_message(timeout_s=1.5)
                except Exception:
                    pass
        finally:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
