"""Ball detection (YOLO11) and offline trajectory reconstruction.

Detection runs per frame at a low confidence threshold so that motion-blurred
balls survive; the tracker then decides which detections form the real ball
path. Tracking happens in *stabilised* coordinates so a panning handheld
camera does not look like ball motion.

The kick is the hard case for a normal Kalman tracker: the ball sits still and
then jumps to >25 m/s in one frame. Instead we build short tracklets with
loose gating and stitch them by positional continuity, which naturally
handles that impulse.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import interpolate_nans

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
COCO_SPORTS_BALL = 32


@dataclass
class BallDet:
    xy: np.ndarray   # (2,) image px
    r: float         # radius px
    conf: float


class BallDetector:
    """Uses fine-tuned weights at models/ball.pt if present, else COCO YOLO11m."""

    def __init__(self, imgsz: int = 960, conf: float = 0.08):
        from ultralytics import YOLO
        custom = MODELS_DIR / "ball.pt"
        if custom.exists():
            self.model, self.classes = YOLO(str(custom)), None
            log.info("Using fine-tuned ball detector %s", custom)
        else:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.model, self.classes = YOLO(str(MODELS_DIR / "yolo11m.pt")), [COCO_SPORTS_BALL]
        self.imgsz, self.conf = imgsz, conf

    def __call__(self, frame_bgr: np.ndarray) -> list[BallDet]:
        res = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, classes=self.classes,
                                 verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            w, h = x2 - x1, y2 - y1
            # A ball is round; very elongated boxes are blur streaks — keep but use the short side.
            out.append(BallDet(np.array([(x1 + x2) / 2, (y1 + y2) / 2]), float(min(w, h)) / 2, float(c)))
        return out


@dataclass
class Tracklet:
    frames: list[int]
    xy: list[np.ndarray]
    r: list[float]
    conf: list[float]
    det_idx: list[int]

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    def velocity(self) -> np.ndarray:
        if len(self.frames) < 2:
            return np.zeros(2)
        n = min(4, len(self.frames))
        dt = self.frames[-1] - self.frames[-n]
        return (self.xy[-1] - self.xy[-n]) / max(dt, 1)

    def score(self) -> float:
        return len(self.frames) * float(np.mean(self.conf))


def build_tracklets(dets: list[list[BallDet]], max_gap: int = 3) -> list[Tracklet]:
    """dets[t] must already be in stabilised coordinates."""
    active: list[Tracklet] = []
    done: list[Tracklet] = []
    for t, frame_dets in enumerate(dets):
        still = []
        for tr in active:
            if t - tr.end > max_gap:
                done.append(tr)
            else:
                still.append(tr)
        active = still
        claimed: set[int] = set()
        for k in sorted(range(len(frame_dets)), key=lambda k: -frame_dets[k].conf):
            d = frame_dets[k]
            best, best_cost = None, np.inf
            for i, tr in enumerate(active):
                if i in claimed:
                    continue
                gap = t - tr.end
                pred = tr.xy[-1] + tr.velocity() * gap
                r = max(np.median(tr.r), d.r, 2.0)
                gate = 4 * r + 0.6 * np.linalg.norm(tr.velocity()) * gap
                dist = np.linalg.norm(d.xy - pred)
                if dist < gate and dist < best_cost:
                    best, best_cost = i, dist
            if best is None:
                active.append(Tracklet([t], [d.xy], [d.r], [d.conf], [k]))
                claimed.add(len(active) - 1)
            else:
                tr = active[best]
                tr.frames.append(t); tr.xy.append(d.xy); tr.r.append(d.r)
                tr.conf.append(d.conf); tr.det_idx.append(k)
                claimed.add(best)
    return done + active


def stitch(tracklets: list[Tracklet], fps: float, max_speed_px: float) -> list[Tracklet]:
    """Chain tracklets forward/backward from the strongest one by continuity.

    ``max_speed_px`` bounds how far (px/s) the ball can plausibly travel between
    the end of one tracklet and the start of the next.
    """
    cands = [t for t in tracklets if len(t.frames) >= 2 or t.conf[0] > 0.4]
    if not cands:
        return []
    seed = max(cands, key=Tracklet.score)
    chain = [seed]

    def gap_ok(a: Tracklet, b: Tracklet) -> bool:
        gap = b.start - a.end
        if gap <= 0 or gap > int(0.4 * fps):
            return False
        dist = np.linalg.norm(b.xy[0] - a.xy[-1])
        r = max(np.median(a.r), np.median(b.r), 2.0)
        return dist <= 6 * r + max_speed_px * gap / fps

    # forward
    while True:
        nxt = [t for t in cands if t not in chain and gap_ok(chain[-1], t)]
        if not nxt:
            break
        chain.append(max(nxt, key=lambda t: t.score() - 0.5 * (t.start - chain[-1].end)))
    # backward
    while True:
        prv = [t for t in cands if t not in chain and gap_ok(t, chain[0])]
        if not prv:
            break
        chain.insert(0, max(prv, key=lambda t: t.score() - 0.5 * (chain[0].start - t.end)))
    return chain


def reconstruct(dets_img: list[list[BallDet]], transforms: np.ndarray, fps: float, body_px: float):
    """Return per-frame ball (T,2) image px, radius (T,), conf (T,).

    Uses stabilised coordinates internally; outputs image coordinates so the
    overlay matches the video.
    """
    from .geometry import apply_transform
    T = len(dets_img)
    stab = []
    for t, ds in enumerate(dets_img):
        stab.append([BallDet(apply_transform(d.xy, transforms[t]), d.r, d.conf) for d in ds])
    # ~40 m/s ball, body ~1.75 m tall → px/s ceiling
    max_speed_px = 40.0 * (body_px / 1.75)
    chain = stitch(build_tracklets(stab), fps, max_speed_px)

    ball = np.full((T, 2), np.nan)
    r = np.full(T, np.nan)
    conf = np.zeros(T)
    for tr in chain:
        for f, k in zip(tr.frames, tr.det_idx):
            img_d = dets_img[f][k]
            ball[f] = img_d.xy
            r[f] = img_d.r
            conf[f] = img_d.conf
    ball = interpolate_nans(ball, max_gap=max(2, int(0.1 * fps)))
    return ball, r, conf
