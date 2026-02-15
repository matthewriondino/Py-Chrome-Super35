#!/usr/bin/env python3
"""
Build a shareable desktop release package for non-Python end users.

This script uses PyInstaller to build:
1) a converter binary (convert_clip_bin)
2) the GUI app bundle (PyChromeSuper35)

The converter binary is bundled inside the GUI app so end users do not need
Python installed.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENTRY_SCRIPT = ROOT / "PyChromeSuper35.py"
CONVERTER_SCRIPT = ROOT / "convert_clip.py"
NATIVE_DIR = ROOT / "native"
PRESETS_DIR = ROOT / "presets"
README_FILE = ROOT / "README.md"

APP_NAME = "PyChromeSuper35"
CONVERTER_BIN_NAME = "convert_clip_bin"

PYI_BUILD_ROOT = ROOT / ".release_build"
PYI_DIST_ROOT = ROOT / ".release_dist"
RELEASE_DIR = ROOT / "release"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build non-Python user release package.")
    parser.add_argument("--install-deps", action="store_true", help="Install/upgrade build and runtime dependencies before building.")
    parser.add_argument("--no-zip", action="store_true", help="Build release folder but skip creating zip archive.")
    parser.add_argument("--clean-only", action="store_true", help="Delete build/release artifacts and exit.")
    parser.add_argument("--keep-build-artifacts", action="store_true", help="Keep temporary PyInstaller work/dist folders.")
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("Running:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def _pyi_bundle_arg(src: Path, dest: str) -> str:
    return f"{src}{os.pathsep}{dest}"


def _rm_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"Removed: {path}")


def _rm_file(path: Path) -> None:
    if path.exists():
        path.unlink()
        print(f"Removed: {path}")


def clean_outputs() -> None:
    _rm_tree(PYI_BUILD_ROOT)
    _rm_tree(PYI_DIST_ROOT)
    _rm_tree(RELEASE_DIR)
    _rm_file(ROOT / f"{APP_NAME}.spec")
    _rm_file(ROOT / f"{CONVERTER_BIN_NAME}.spec")


def install_deps() -> None:
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pyinstaller",
            "dearpygui",
            "numpy",
            "pillow",
            "opencv-python",
        ]
    )


def build_converter_binary() -> Path:
    out_dir = PYI_DIST_ROOT / "converter"
    work_dir = PYI_BUILD_ROOT / "converter"
    spec_dir = PYI_BUILD_ROOT / "spec"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--console",
        "--name",
        CONVERTER_BIN_NAME,
        "--workpath",
        str(work_dir),
        "--distpath",
        str(out_dir),
        "--specpath",
        str(spec_dir),
        "--collect-all",
        "pychrome_native",
        "--add-data",
        _pyi_bundle_arg(NATIVE_DIR, "native"),
        str(CONVERTER_SCRIPT),
    ]
    _run(cmd)

    exe = ".exe" if os.name == "nt" else ""
    converter_bin = out_dir / f"{CONVERTER_BIN_NAME}{exe}"
    if not converter_bin.exists():
        raise RuntimeError(f"Expected converter binary not found: {converter_bin}")
    return converter_bin


def build_gui_app(converter_bin: Path) -> Path:
    out_dir = PYI_DIST_ROOT / "gui"
    work_dir = PYI_BUILD_ROOT / "gui"
    spec_dir = PYI_BUILD_ROOT / "spec"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        APP_NAME,
        "--workpath",
        str(work_dir),
        "--distpath",
        str(out_dir),
        "--specpath",
        str(spec_dir),
        "--collect-all",
        "pychrome_native",
        "--add-data",
        _pyi_bundle_arg(PRESETS_DIR, "presets"),
        "--add-data",
        _pyi_bundle_arg(NATIVE_DIR, "native"),
        "--add-binary",
        _pyi_bundle_arg(converter_bin, "."),
        str(ENTRY_SCRIPT),
    ]
    _run(cmd)

    system = platform.system().lower()
    if system == "darwin":
        artifact = out_dir / f"{APP_NAME}.app"
    else:
        artifact = out_dir / APP_NAME

    if not artifact.exists():
        raise RuntimeError(f"Expected GUI artifact not found: {artifact}")
    return artifact


def stage_release(gui_artifact: Path, *, no_zip: bool) -> Path | None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    target_app = RELEASE_DIR / gui_artifact.name
    if target_app.exists():
        if target_app.is_dir():
            shutil.rmtree(target_app)
        else:
            target_app.unlink()

    if gui_artifact.is_dir():
        shutil.copytree(gui_artifact, target_app)
    else:
        shutil.copy2(gui_artifact, target_app)

    if README_FILE.exists():
        shutil.copy2(README_FILE, RELEASE_DIR / README_FILE.name)

    quickstart = RELEASE_DIR / "QUICKSTART.txt"
    quickstart.write_text(
        "\n".join(
            [
                "PyChromeSuper35 - Quick Start",
                "",
                "1. Open the app bundle/application inside this folder.",
                "2. Load a video and add one or more jobs to queue.",
                "3. Click Convert Clip (Process Queue).",
                "",
                "This release is packaged for non-Python users.",
            ]
        ),
        encoding="utf-8",
    )

    if no_zip:
        return None

    system = platform.system().lower()
    arch = platform.machine().lower().replace(" ", "")
    zip_base = ROOT / f"{APP_NAME}_{system}_{arch}"
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=RELEASE_DIR))
    return zip_path


def validate_inputs() -> None:
    missing = []
    for p in (ENTRY_SCRIPT, CONVERTER_SCRIPT, NATIVE_DIR, PRESETS_DIR):
        if not p.exists():
            missing.append(str(p))
    if missing:
        msg = "Missing required files/folders:\n- " + "\n- ".join(missing)
        raise FileNotFoundError(msg)


def main() -> int:
    args = parse_args()

    try:
        validate_inputs()

        if args.clean_only:
            clean_outputs()
            print("Clean finished.")
            return 0

        if args.install_deps:
            install_deps()

        clean_outputs()

        converter_bin = build_converter_binary()
        gui_artifact = build_gui_app(converter_bin)
        zip_path = stage_release(gui_artifact, no_zip=args.no_zip)

        if not args.keep_build_artifacts:
            _rm_tree(PYI_BUILD_ROOT)
            _rm_tree(PYI_DIST_ROOT)

        print("\nRelease ready:")
        print(f"- Folder: {RELEASE_DIR}")
        if zip_path:
            print(f"- Zip:    {zip_path}")
        return 0

    except subprocess.CalledProcessError as exc:
        print(f"Build failed (exit {exc.returncode}).")
        return exc.returncode
    except Exception as exc:
        print(f"Build failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
