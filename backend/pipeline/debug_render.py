"""Render the tracked skeleton, ball and detected events onto the clip.

    python -m pipeline.debug_render data/cli/<name>     -> data/cli/<name>/debug.mp4
    python -m pipeline.debug_render data/cli/<name> --sheet  -> also a contact sheet around contact
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .skeleton import EDGES
from .tracks import Tracks
from .video import iter_frames


def draw(frame, tr: Tracks, t: int, ev: dict) -> np.ndarray:
    img = frame.copy()
    kp = tr.kp[t]
    for a, b in EDGES:
        if not np.any(np.isnan(kp[[a, b]])):
            cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (0, 255, 255), 2)
    for p in kp:
        if not np.any(np.isnan(p)):
            cv2.circle(img, tuple(p.astype(int)), 3, (0, 128, 255), -1)
    if not np.isnan(tr.ball[t, 0]):
        r = int(tr.ball_r[t]) if not np.isnan(tr.ball_r[t]) else 8
        cv2.circle(img, tuple(tr.ball[t].astype(int)), max(r, 4), (255, 0, 255), 2)
    labels = [k for k in ("approach_start", "plant", "backswing", "contact", "follow_through") if ev.get(k) == t]
    cv2.putText(img, f"{t} {' '.join(labels)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--sheet", action="store_true")
    args = ap.parse_args()
    tr = Tracks.load(args.dir / "tracks.npz")
    ev = json.loads((args.dir / "analysis.json").read_text())["events"]
    out = cv2.VideoWriter(str(args.dir / "debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), tr.fps, (tr.width, tr.height))
    sheet = []
    c = ev["contact"]
    picks = set(np.linspace(max(0, c - int(0.5 * tr.fps)), min(tr.T - 1, c + int(0.4 * tr.fps)), 12).astype(int))
    for t, frame in iter_frames(args.dir / "playback.mp4"):
        img = draw(frame, tr, t, ev)
        out.write(img)
        if t in picks:
            sheet.append(cv2.resize(img, (tr.width // 3, tr.height // 3)))
    out.release()
    if args.sheet and sheet:
        while len(sheet) % 4:
            sheet.append(np.zeros_like(sheet[0]))
        rows = [np.hstack(sheet[i:i + 4]) for i in range(0, len(sheet), 4)]
        cv2.imwrite(str(args.dir / "sheet.jpg"), np.vstack(rows))


if __name__ == "__main__":
    main()
