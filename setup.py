#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

RUNTIME_PACKAGES = [
    "dearpygui",
    "numpy",
    "pillow",
    "opencv-python",
]

BUILD_PACKAGES = [
    "setuptools",
    "wheel",
]

REQUIRED_IMPORTS = [
    ("dearpygui.dearpygui", "dearpygui"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("cv2", "opencv-python"),
]

OPTIONAL_IMPORTS = [
    ("cupy", "cupy"),
]


def _print_step(text: str) -> None:
    print(f"\n==> {text}", flush=True)


def _run(cmd, *, check: bool = True, shell: bool = False) -> int:
    if isinstance(cmd, str):
        shown = cmd
    else:
        shown = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"$ {shown}", flush=True)
    result = subprocess.run(cmd, shell=shell, cwd=str(ROOT))
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {shown}")
    return int(result.returncode)


def _run_optional_sudo(cmd: list[str], *, check: bool = False) -> int:
    try:
        needs_sudo = (os.name != "nt") and hasattr(os, "geteuid") and os.geteuid() != 0 and shutil.which("sudo")
    except Exception:
        needs_sudo = False
    full_cmd = (["sudo"] + cmd) if needs_sudo else cmd
    return _run(full_cmd, check=check)


def _which_any(candidates: list[str]) -> str | None:
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    return None


def _pip_install(packages: list[str], *, extra_args: list[str] | None = None) -> int:
    cmd = [PYTHON_EXE, "-m", "pip", "install", "--upgrade"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    return _run(cmd, check=False)


def ensure_python_packages() -> None:
    _print_step("Installing Python packages")
    _run([PYTHON_EXE, "-m", "ensurepip", "--upgrade"], check=False)
    _run([PYTHON_EXE, "-m", "pip", "install", "--upgrade", "pip"], check=False)
    packages = [*BUILD_PACKAGES, *RUNTIME_PACKAGES]

    if _pip_install(packages) == 0:
        return

    print("Default pip install failed; retrying with --user.", flush=True)
    if _pip_install(packages, extra_args=["--user"]) == 0:
        return

    print("User install failed; retrying with --break-system-packages (if supported).", flush=True)
    if _pip_install(packages, extra_args=["--break-system-packages"]) == 0:
        return

    raise RuntimeError("Unable to install Python runtime dependencies via pip.")


def _cupy_candidates() -> list[str]:
    raw = os.getenv("PYCHROME_CUPY_PACKAGES", "").strip()
    if raw:
        vals = [x.strip() for x in raw.split(",") if x.strip()]
        if vals:
            return vals
    # Try binary CUDA wheels first to avoid source builds on fresh systems.
    return ["cupy-cuda12x", "cupy-cuda11x", "cupy"]


def install_optional_cupy() -> tuple[bool, str]:
    try:
        importlib.import_module("cupy")
        return True, "already installed"
    except Exception:
        pass

    if platform.system() == "Darwin":
        return False, "skipped on macOS (CuPy usually requires CUDA)"

    _print_step("Installing optional CuPy (GPU acceleration)")
    for pkg in _cupy_candidates():
        print(f"Trying CuPy package: {pkg}", flush=True)
        code = _pip_install([pkg], extra_args=["--only-binary=:all:"])
        if code == 0:
            try:
                importlib.import_module("cupy")
                return True, pkg
            except Exception:
                pass
        # Retry with --user for locked environments.
        code = _pip_install([pkg], extra_args=["--only-binary=:all:", "--user"])
        if code == 0:
            try:
                importlib.import_module("cupy")
                return True, pkg
            except Exception:
                pass
    return False, "no compatible CuPy wheel found"


def _discover_brew() -> str | None:
    brew = shutil.which("brew")
    if brew:
        return brew
    for path in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew"):
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    return None


def _install_homebrew_macos() -> str | None:
    brew = _discover_brew()
    if brew:
        return brew
    _print_step("Homebrew not found. Installing Homebrew (macOS)")
    install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    code = _run(install_cmd, check=False, shell=True)
    if code != 0:
        print("Homebrew install failed.", flush=True)
        return None
    return _discover_brew()


def _ffmpeg_present() -> bool:
    if shutil.which("ffmpeg"):
        return True
    if platform.system() == "Darwin":
        return os.path.exists("/opt/homebrew/bin/ffmpeg") or os.path.exists("/usr/local/bin/ffmpeg")
    return False


def _ffprobe_present() -> bool:
    if shutil.which("ffprobe"):
        return True
    if platform.system() == "Darwin":
        return os.path.exists("/opt/homebrew/bin/ffprobe") or os.path.exists("/usr/local/bin/ffprobe")
    return False


def _install_ffmpeg_macos() -> bool:
    brew = _install_homebrew_macos()
    if not brew:
        return False
    _print_step("Installing ffmpeg via Homebrew")
    code = _run([brew, "install", "ffmpeg"], check=False)
    if code != 0:
        _run([brew, "upgrade", "ffmpeg"], check=False)
    return _ffmpeg_present()


def _install_ffmpeg_linux() -> bool:
    _print_step("Installing ffmpeg on Linux")
    if shutil.which("apt-get"):
        _run_optional_sudo(["apt-get", "update"], check=False)
        _run_optional_sudo(["apt-get", "install", "-y", "ffmpeg"], check=False)
        return _ffmpeg_present()
    if shutil.which("dnf"):
        _run_optional_sudo(["dnf", "install", "-y", "ffmpeg"], check=False)
        return _ffmpeg_present()
    if shutil.which("yum"):
        _run_optional_sudo(["yum", "install", "-y", "ffmpeg"], check=False)
        return _ffmpeg_present()
    if shutil.which("pacman"):
        _run_optional_sudo(["pacman", "-Sy", "--noconfirm", "ffmpeg"], check=False)
        return _ffmpeg_present()
    return False


def _install_ffmpeg_windows() -> bool:
    _print_step("Installing ffmpeg on Windows")
    if shutil.which("winget"):
        _run(
            [
                "winget",
                "install",
                "-e",
                "--id",
                "Gyan.FFmpeg",
                "--accept-source-agreements",
                "--accept-package-agreements",
            ],
            check=False,
        )
        if _ffmpeg_present():
            return True
    if shutil.which("choco"):
        _run(["choco", "install", "-y", "ffmpeg"], check=False)
    return _ffmpeg_present()


def ensure_ffmpeg() -> bool:
    if _ffmpeg_present():
        _print_step("ffmpeg already installed")
        return True
    system = platform.system()
    if system == "Darwin":
        return _install_ffmpeg_macos()
    if system == "Linux":
        return _install_ffmpeg_linux()
    if system == "Windows":
        return _install_ffmpeg_windows()
    return False


def build_native_extension() -> bool:
    _print_step("Building native CPU extension (optional, for better performance)")
    setup_path = ROOT / "CPU_setup.py"
    if not setup_path.exists():
        print("CPU_setup.py not found; skipping native extension build.", flush=True)
        return False
    code = _run([PYTHON_EXE, str(setup_path), "build_ext", "--inplace"], check=False)
    if code == 0:
        return True
    print("Native extension build failed. The app can still run with NumPy fallback.", flush=True)
    return False


def verify_python_imports() -> tuple[bool, list[str]]:
    missing = []
    for module_name, package_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(f"{package_name} (import '{module_name}')")
    return (len(missing) == 0), missing


def verify_optional_imports() -> list[str]:
    missing_optional = []
    for module_name, package_name in OPTIONAL_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing_optional.append(f"{package_name} (optional)")
    return missing_optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize Py-Chrome Super35: install Python deps, ffmpeg, and optional native extension."
    )
    parser.add_argument("--skip-python-packages", action="store_true", help="Skip pip package installation.")
    parser.add_argument("--skip-ffmpeg", action="store_true", help="Skip ffmpeg/Homebrew setup.")
    parser.add_argument("--skip-native-build", action="store_true", help="Skip native CPU extension build.")
    parser.add_argument("--skip-cupy", action="store_true", help="Skip optional CuPy install attempt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Py-Chrome Super35 initializer")
    print(f"Python: {PYTHON_EXE}")
    print(f"Project: {ROOT}")

    had_error = False
    python_ok = True
    cupy_ok = False
    cupy_note = "not attempted"

    try:
        if not args.skip_python_packages:
            ensure_python_packages()
    except Exception as exc:
        had_error = True
        python_ok = False
        print(f"[error] Python dependency install failed: {exc}", flush=True)

    ffmpeg_ok = True
    if not args.skip_ffmpeg:
        ffmpeg_ok = ensure_ffmpeg()
        if not ffmpeg_ok:
            had_error = True
            print("[error] ffmpeg installation failed or ffmpeg not found in expected paths.", flush=True)

    native_ok = True
    if not args.skip_native_build:
        native_ok = build_native_extension()

    if not args.skip_cupy:
        cupy_ok, cupy_note = install_optional_cupy()
    else:
        cupy_note = "skipped by flag"

    _print_step("Validating installed dependencies")
    imports_ok, missing_imports = verify_python_imports()
    if not imports_ok:
        had_error = True
        print("[error] Missing required Python imports:", flush=True)
        for item in missing_imports:
            print(f"  - {item}", flush=True)
    optional_missing = verify_optional_imports()
    ffprobe_ok = _ffprobe_present()
    if not ffprobe_ok:
        print("[warn] ffprobe was not found. App can run, but metadata probing may be reduced.", flush=True)

    print("\nSummary:")
    print(f"- Python deps: {'skipped' if args.skip_python_packages else ('ok' if (python_ok and imports_ok) else 'failed')}")
    print(f"- ffmpeg: {'ok' if (args.skip_ffmpeg or ffmpeg_ok) else 'failed'}")
    print(f"- ffprobe: {'ok' if ffprobe_ok else 'warn'}")
    print(f"- Native extension: {'ok' if (args.skip_native_build or native_ok) else 'fallback to NumPy'}")
    print(f"- CuPy (optional): {'ok' if cupy_ok else cupy_note}")
    if optional_missing:
        print(f"- Optional GPU deps: missing {', '.join(optional_missing)}")
    print("\nRun app with:")
    print("python3 PyChromeSuper35.py")

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
