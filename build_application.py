#!/usr/bin/env python3
"""
One-command application builder for distribution to non-Python users.

Usage:
  python3 build_application.py
  python3 build_application.py --no-zip
  python3 build_application.py --clean-only
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RELEASE_SCRIPT = ROOT / "build_release.py"


REQUIRED_MODULES = [
    ("PyInstaller", "pyinstaller"),
    ("dearpygui.dearpygui", "dearpygui"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("cv2", "opencv-python"),
]


def _missing_python_deps() -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for module_name, package_name in REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append((module_name, package_name))
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install dependencies and build a shareable PyChromeSuper35 application package."
    )
    parser.add_argument("--no-zip", action="store_true", help="Build app/package but skip creating the release zip archive.")
    parser.add_argument("--clean-only", action="store_true", help="Delete build outputs and exit (no install/build).")
    parser.add_argument("--keep-build-artifacts", action="store_true", help="Keep temporary PyInstaller build folders after successful export.")
    parser.add_argument("--force-install-deps", action="store_true", help="Force dependency installation before build.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not RELEASE_SCRIPT.exists():
        print(f"Missing required script: {RELEASE_SCRIPT}")
        return 1

    cmd = [sys.executable, str(RELEASE_SCRIPT)]

    install_deps = False
    if not args.clean_only:
        missing = _missing_python_deps()
        if missing:
            install_deps = True
            print("Missing Python dependencies detected:")
            for module_name, package_name in missing:
                print(f"- {package_name} (import: {module_name})")
            print("Will install/upgrade dependencies before building.")
        elif args.force_install_deps:
            install_deps = True
            print("Force install requested. Will install/upgrade dependencies before building.")
        else:
            print("Dependency check passed. Building without reinstalling dependencies.")

    if install_deps:
        cmd.append("--install-deps")
    if args.no_zip:
        cmd.append("--no-zip")
    if args.clean_only:
        cmd.append("--clean-only")
    if args.keep_build_artifacts:
        cmd.append("--keep-build-artifacts")

    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Build failed with exit code {exc.returncode}.")
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
