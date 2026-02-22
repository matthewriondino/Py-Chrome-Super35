# SOURCE_CODE_EXPLANATION

This document explains what the source code does, how major modules interact, and how frame data moves through the system.

## 1) Top-Level Architecture

The project is intentionally split into UI orchestration, processing functions, conversion/export logic, and optional acceleration backends.

Main files:
- `PyChromeSuper35.py`: main GUI application, state, controls, preview rendering, queue/status orchestration.
- `convert_clip.py`: CLI conversion engine used by GUI batch mode and direct CLI usage.
- `pcv_color.py`: white balance and scientific IRG transform functions.
- `pcv_lut.py`: `.cube` LUT parser + LUT apply logic.
- `pcv_scopes.py`: histogram/waveform/vectorscope generation.
- `pychrome_native/backend_manager.py`: backend selection/fallback (`metal`, `cupy`, `cpu_ext`, `numpy`).
- `pychrome_native/cpu_core.c`: compiled CPU extension for faster frame processing.
- `pychrome_native/metal_backend.py` + `native/metal_worker.swift`: Metal-based GPU processing bridge on macOS.

## 2) Application Entry and UI Construction

`PyChromeSuper35.py` builds the entire DearPyGui interface in one run:
1. Creates textures for main preview, scopes, and channel previews.
2. Builds left panel controls (file open, queue, presets, WB, transform sliders, scopes, LUT).
3. Builds right panel preview/timeline/compute controls.
4. Builds modal windows (delete preset, overwrite preset, conversion status).
5. Starts viewport and main event loop.

Most GUI callbacks mutate shared state and call `request_preview_update()` rather than rendering immediately.
This is a key performance design: render requests are coalesced.

## 3) State and Render Scheduling

Important runtime state in `PyChromeSuper35.py`:
- `preview_img`: current frame in float32 [0..1].
- `preview_work_img`: downscaled working image for faster preview.
- `video_capture`: OpenCV capture handle for timeline playback.
- queue state: `queue_items` list with input/output/preset snapshots.
- conversion state: `_conv_state` and worker queues for status pipeline.

Render throttling model:
- UI changes call `request_preview_update()`.
- Dirty updates are coalesced via frame callback heartbeat.
- Preview/scope/scatter/channel updates are throttled at different intervals.
- During conversion, preview workload is reduced to keep UI responsive.

## 4) Preview Pipeline (What Happens Per Frame)

In `update_main_preview()` the pipeline is:
1. Select source frame (`preview_img` or downscaled `preview_work_img`).
2. Apply white balance.
3. Apply IRG transform via backend manager (`auto/cpu/gpu`).
4. Apply optional LUT preview (non-destructive).
5. Upload main texture + channel textures.
6. Update scopes/scatter if their throttles allow.

### Preview bit depth behavior
- `8-bit`: frame decode via OpenCV path.
- `10-bit`: frame decode request through ffmpeg (`rgb48le`) for high-bit-depth preview path.
- Fallback to 8-bit if ffmpeg decode is unavailable.

This affects preview only, not export policy.

## 5) Color/Transform Functions

`pcv_color.py` handles:
- Kelvin -> RGB white balance gain conversion.
- Tint adjustment behavior.
- Scientific IRG equations.
- Clamping and normalization helpers.

The transform uses user parameters:
- `fracRx`, `fracGx`, `fracBY`
- `gammaRx`, `gammaRy`, `gammaGx`, `gammaGy`, `gammaBY`
- `exposure`

The output channels are stored as `[IR, R, G]` internally, then mapped for display/export behavior.

## 6) Scope and Analysis System

`pcv_scopes.py` provides vectorized scope generation:
- Histogram texture generation with overlap/additive channel color behavior.
- Waveform texture generation.
- Vectorscope scatter series generation.
- Sampling limits for performance.

In GUI:
- Only one main scope view is shown at a time.
- Scatterplots are separate and collapsible.
- Scope update frequency is throttled more aggressively during conversion.

## 7) Presets and Queue Model

Presets:
- Stored as JSON in `presets/`.
- UI supports save/load/delete/overwrite confirmation.

Queue items contain:
- input path
- output path
- full preset snapshot

Key behavior:
- Queue is keyed by input path.
- Existing queue item changes button label to `Update to Queue`.
- Double-click queue item reloads clip + settings into UI for re-editing.

## 8) Conversion Pipeline (GUI -> CLI)

When the user converts queue:
1. GUI writes a temporary batch JSON file.
2. GUI launches converter command (`convert_clip.py --batch ...` in source mode, `convert_clip_bin` in packaged mode).
3. GUI reads converter stdout asynchronously.
4. A dedicated status worker parses progress events and updates a shared status snapshot.
5. UI heartbeat applies snapshot to modal text/progress/log with throttling.

This worker/snapshot model prevents UI freezes from log flood.

## 9) `convert_clip.py` Processing Details

`convert_clip.py` runs per job:
1. Probe input metadata (`ffprobe` + OpenCV fallback).
2. Determine output strategy (bit-depth class, codec args, pixel format).
3. Decode source frames.
4. Process each frame using selected backend.
5. Encode output with ffmpeg and map/copy audio when available.
6. Emit structured status lines for GUI (`JOB_START`, `TOTAL_FRAMES`, `PROCESSED`, `JOB_DONE`, `BATCH_DONE`).

Bit-depth policy:
- 10-bit class source -> 10-bit HEVC output path.
- 8-bit class source -> 8-bit H.264 output path.

Progress lines are intentionally compact/throttled to reduce status UI pressure.

## 10) Backend Manager and Fallback Behavior

`pychrome_native/backend_manager.py` selects processing backend in this order (depending on request):
- Metal (if available and healthy)
- CuPy (if available)
- compiled CPU extension (`_cpu_core`)
- NumPy fallback

Design goals:
- Never crash entire app when optional backend fails.
- Degrade gracefully to safe fallback.
- Keep output behavior functionally consistent.

## 11) Native CPU Extension

`CPU_setup.py` builds `pychrome_native/_cpu_core...so` from `pychrome_native/cpu_core.c`.

Purpose:
- Speed up CPU transform path by replacing heavy NumPy sections with compiled C loops.

If build fails, app still runs using NumPy fallback.

## 12) Metal Worker Model

The Metal backend uses a helper worker process (`metal_worker.swift`) instead of embedding GPU kernels directly in Python.

Flow:
1. Python spawns helper.
2. Parameters are sent to worker.
3. Frames are sent/received as buffers.
4. On failure, backend manager marks backend degraded and falls back.

## 13) Packaging and Build Scripts

- `setup.py`: user-friendly installer/bootstrap (pip deps + ffmpeg + optional extras).
- `build_application.py`: dependency preflight + release builder wrapper.
- `build_release.py`: PyInstaller packaging pipeline.

Packaged app includes converter binary and runtime resources so non-Python users can run it directly.

## 14) Troubleshooting Source-Level Issues

If something breaks while editing:
1. Run syntax checks:
   - `python3 -m py_compile PyChromeSuper35.py convert_clip.py pcv_color.py pcv_lut.py pcv_scopes.py`
2. Test source run first.
3. Then test packaged run.
4. Compare behavior differences (environment/path/backend availability).

Common source-vs-packaged difference:
- Packaged app is usually more stable due to a controlled runtime environment and fixed dependency bundle.

