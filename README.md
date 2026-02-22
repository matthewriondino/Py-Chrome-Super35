# Py-Chrome Super35 (Video CIR / Aerochrome Emulation)

Py-Chrome Super35 is a desktop video tool that emulates Color Infrared (CIR) aesthetics, inspired by the look historically associated with Kodak Aerochrome.

## What It Is

Py-Chrome Super35 lets you:
- Load video clips and scrub a timeline.
- Apply white balance + CIR transform controls.
- Analyze output with Histogram, Waveform, Vectorscope, and Scatterplots.
- Save/load presets.
- Queue multiple clips and batch convert.
- Preview LUTs non-destructively.
- Export while preserving practical source characteristics (resolution, frame rate, audio, and bit-depth class when supported).

## Why It Was Made

Aerochrome film stock is discontinued, rare, often expired, and expensive.

This project was built as a practical digital alternative for creators who want that CIR-style visual language with a repeatable, editable, modern video workflow.

## Installation

## Option A: Run From Source (Simple)

For users who do not code and just want to run the app from source:

1. Install Python from [python.org](https://www.python.org/downloads/macos/).
2. Open this folder in Finder.
3. Right-click `setup.py` and choose `Open With` -> `Python Launcher`.
4. Wait for setup to complete (Terminal will show progress).
5. Close Terminal.
6. Right-click `PyChromeSuper35.py` and choose `Open With` -> `Python Launcher`.

What `setup.py` does:
- Installs required Python packages.
- Installs ffmpeg (and Homebrew first on macOS if needed).
- Attempts optional native CPU extension build.
- Attempts optional CuPy install when compatible.

## Option B: Build Executable Application

To build a standalone application package:

1. Install Python from [python.org](https://www.python.org/downloads/macos/).
2. Open this folder in Finder.
3. Right-click `build_application.py` and choose `Open With` -> `Python Launcher`.
4. Wait for build to complete (Terminal will show progress).
5. Close Terminal.
6. A folder will be created called "Release", the application "PyChromeSuper35" will be made ready to run

What `setup.py` does:
- Installs required Python packages.
- Installs ffmpeg (and Homebrew first on macOS if needed).
- Attempts optional native CPU extension build.
- Attempts optional CuPy install when compatible.
- Compiles all source code into a single application to run

## How It Works

## High-Level Pipeline

Each frame follows this order:
1. Decode frame from input.
2. Pre-White balance (temperature + tint).
3. Scientific IRG transform (fraction + gamma + exposure controls).
4. Optional non-destructive LUT preview (GUI preview path).
5. Display in preview and scopes.
6. For export: encode processed frames to output file.

## Core Transform Model

The transform takes white-balanced channels and computes CIR-like output channels:
- `Y` (IR-derived channel)
- `X1` (red-recovered channel)
- `X2` (green-recovered channel)

Control groups:
- Fractions: `fracRx`, `fracGx`, `fracBY`
- Gamma: `gammaRx`, `gammaRy`, `gammaGx`, `gammaGy`, `gammaBY`
- Exposure: `exposure`

This gives fine control over foliage response, channel separation, saturation behavior, and highlight rolloff.

## Preview Bit-Depth

Preview has a depth selector:
- `8-bit` preview: OpenCV decode path.
- `10-bit` preview: ffmpeg-based `rgb48le` preview decode path (falls back automatically if unavailable).

This affects preview only. Export settings are independent.

## Compute Backends

Compute selector:
- `Auto`
- `CPU`
- `GPU`

Backend behavior:
- Auto: prefers accelerated backend, then safe fallback.
- CPU: prefers native CPU extension when available.
- GPU: prefers GPU (Metal (macOS)), or CuPy (Windows) if available.
- Fallback remains functional with NumPy.

## How To Use (Detailed)

## Recommended Workflow

1. Open a clip.
2. Pick `Preview Quality`, `Compute`, and `Preview Depth`.
3. Optional: load a LUT for visual preview.
4. Set white balance with WB dropper first. 
5. Adjust fraction sliders.
6. Adjust gamma + exposure.
7. Check scopes for clipping/separation.
8. Save preset.
9. Add clip to queue.
10. Repeat for more clips.
11. Convert queue.

## Queue Editing Behavior

Queue includes:
- Add to Queue
- Update to Queue (appears when current clip already exists in queue)
- Remove Queue
- Move Up / Move Down

Double-click behavior:
- Double-click a queue item to reload that clip and its saved settings for re-editing.

## Practical Examples

## Example 1: Balanced CIR Look

Goal: natural scene with strong CIR feel but controlled highlights.

Suggested approach:
1. WB dropper on neutral object.
2. Keep `fracBY` conservative to avoid clipping in bright foliage.
3. Increase `Gamma Red IR` and `Gamma Green IR`, while decreasing 'Gamma Red Visible' and 'Gamma Green Visible' for better tonal separation in IR-affected regions.
4. Fine tune exposure at the end.
5. Confirm with Histogram + Waveform.

## Example 2: Strong Crimson Foliage

Goal: aggressive false-color style.

Suggested approach:
1. WB first.
2. Lower visible fractions (`fracRx`, `fracGx`) with care.
3. Raise IR-region gamma controls (`Gamma Red IR` and `Gamma Green IR`) to shape foliage response.
4. Watch histogram for blown highlights.
5. Save preset before export.

## Technical: Import and Export (Decode/Encode)

This section is for video editors and technical users.

## Input / Import

- Metadata is probed from source (frame count, fps, dimensions, pixel format, bit depth class, color metadata when available).
- Preview timeline uses frame decoding for interactive viewing.
- Preview can use 8-bit or 10-bit decode mode as selected in UI.

## Export / Encode

Preferred export path:
- ffmpeg decode -> process in float32 -> ffmpeg encode.

Fallback path:
- OpenCV encode path (functional fallback when ffmpeg unavailable).

Default export behavior:
- Source 10-bit class -> 10-bit HEVC (`libx265`, `yuv420p10le`, `hvc1`).
- Source 8-bit class -> 8-bit H.264 (`libx264`, `yuv420p`, `avc1`).

Stream mapping behavior:
- Processed video stream becomes output video.
- Source audio is copied when available.

Practical output intent:
- Match source resolution and frame rate.
- Preserve audio when possible.
- Preserve source bit-depth class when supported by selected encode path.

## Important Note

Preview depth does not force export depth.
Export depth is decided by conversion/export pipeline rules.

## Troubleshooting

## Setup Script Opens Then Fails

- Ensure Python from python.org is installed.
- Re-run `setup.py` with Python Launcher.
- If prompted for password during package/system installs on macOS, this is normal.

## ffmpeg Not Found

- Run `setup.py` again.
- Verify `ffmpeg` exists in Terminal:

```bash
ffmpeg -version
ffprobe -version
```

## Preview Looks Blocky On High-Bit-Depth Footage

- Switch `Preview Depth` to `10-bit`.
- If ffmpeg is missing, install it (setup script handles this).
- Set `Preview Quality` to `Balanced` or `Full`.

## Conversion Progress Seems Stuck

- Let conversion continue for a moment under heavy system load.
- The status/log path is asynchronous and may briefly lag on busy systems.
- Check output file growth and final completion status.

Note: Do not panic if the Progress Bar freezes, conversion continues

## Output Has Audio But No Video (or unreadable in some players)

- Re-run using ffmpeg path (not fallback).
- Verify source codec compatibility and try another container/target codec if needed.
- Test output in multiple players (QuickTime, VLC, Resolve) to isolate player-specific behavior.

## GPU Mode Issues

- On macOS, `GPU` mode uses Metal path when available.
- If GPU backend is unstable on a system, switch to `CPU` or `Auto`.

## Project Files (Current Folder)

Main runtime files:
- `PyChromeSuper35.py` (GUI entry + orchestration)
- `convert_clip.py` (batch conversion/export pipeline)
- `pcv_color.py` (color math + transform)
- `pcv_lut.py` (LUT parsing/apply)
- `pcv_scopes.py` (scope generation)
- `pychrome_native/` (backend manager + native acceleration glue)
- `native/metal_worker.swift` (Metal helper)
- `setup.py` (simple installer/bootstrap)
- `CPU_setup.py` (native CPU extension build)
- `build_application.py` (one-command app build)
- `build_release.py` (packaging pipeline)

## License

MIT
