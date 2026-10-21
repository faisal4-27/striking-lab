"""Video normalisation (ffmpeg) and frame decoding (OpenCV).

Every upload is transcoded once to constant-frame-rate H.264 (long side ≤ 1280,
≤ 60 fps, rotation baked in). The pipeline analyses that file and the browser
plays the same file, so frame index ``i`` is exactly time ``i / fps`` for both,
which keeps the overlay in sync. It also makes .mov/HEVC and MediaRecorder
WebM uploads (often with bogus frame-rate headers) behave consistently.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 1280
MAX_FPS = 60
MAX_SECONDS = 12


def _tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    # winget installs; the server may have started before PATH was refreshed
    winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet"
    for cand in [winget / "Links" / f"{name}.exe", *winget.glob(f"Packages/*FFmpeg*/*/bin/{name}.exe")]:
        if cand.exists():
            return str(cand)
    raise RuntimeError(f"{name} not found - install ffmpeg (winget install Gyan.FFmpeg)")


def measure_fps(path: Path) -> float:
    """Median packet interval; robust to MediaRecorder's 1000/1 headers."""
    out = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=pts_time", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    pts = sorted(float(p["pts_time"]) for p in json.loads(out).get("packets", []) if "pts_time" in p)
    if len(pts) < 3:
        return 30.0
    dt = np.median(np.diff(pts))
    fps = 1.0 / dt if dt > 0 else 30.0
    # snap to common rates
    for common in (24, 25, 30, 48, 50, 60, 90, 120, 240):
        if abs(fps - common) / common < 0.06:
            return float(common)
    return float(np.clip(fps, 10, 240))


def normalize(src: Path, dst: Path) -> dict:
    src_fps = measure_fps(src)
    fps = min(src_fps, MAX_FPS)
    scale = (f"scale=w='if(gte(iw,ih),min({MAX_SIDE},iw),-2)':"
             f"h='if(gte(iw,ih),-2,min({MAX_SIDE},ih))'")
    cmd = [_tool("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-t", str(MAX_SECONDS),
           "-vf", scale, "-r", str(fps), "-fps_mode", "cfr",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-an", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)
    cap = cv2.VideoCapture(str(dst))
    info = {
        "fps": fps, "source_fps": src_fps,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def iter_frames(path: Path, only: set[int] | None = None):
    """Yield (index, BGR frame). ``only`` skips decoding work for other frames' outputs."""
    cap = cv2.VideoCapture(str(path))
    i = 0
    try:
        while True:
            if only is not None and i not in only:
                if not cap.grab():
                    break
                i += 1
                if i > max(only):
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()
