"""Pose estimation.

* 2D: RTMW-x wholebody (via rtmlib / ONNX Runtime on CUDA). Gives 133 keypoints;
  we keep the 23 body+feet points (see skeleton.py). Run on every frame.
* 3D: MediaPipe Pose Landmarker (heavy) world landmarks — metric, hip-centred,
  roughly view-invariant. Run on a crop around the kicker only for the frames
  around the kick, and used for joint angles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import interpolate_nans
from .skeleton import NUM_KEYPOINTS

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")

KP_MIN_SCORE = 0.3


@dataclass
class PersonCandidate:
    box: np.ndarray      # (4,) x1 y1 x2 y2
    kp: np.ndarray       # (23, 2)
    score: np.ndarray    # (23,)


def _onnx_device() -> str:
    try:
        import onnxruntime as ort
        # Make the CUDA / cuDNN DLLs that ship with torch visible to ORT.
        if hasattr(ort, "preload_dlls"):
            try:
                import torch  # noqa: F401
                ort.preload_dlls()
            except Exception:  # pragma: no cover - best effort
                pass
        return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    except ImportError:
        return "cpu"


class Pose2D:
    """Multi-person 2D wholebody estimator returning per-frame candidates.

    The person detector runs every ``det_every`` frames; in between, boxes are
    propagated from the previous frame's keypoints (top-down pose tracking).
    Call ``reset()`` between videos.
    """

    def __init__(self, mode: str = "performance", det_every: int = 4):
        from rtmlib import Wholebody
        device = _onnx_device()
        log.info("RTMW wholebody on %s", device)
        wb = Wholebody(mode=mode, backend="onnxruntime", device=device)
        self.det = wb.det_model
        self.pose = wb.pose_model
        self.det_every = det_every
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._prev: list[PersonCandidate] = []

    @staticmethod
    def _box_from_kp(c: PersonCandidate, shape) -> np.ndarray | None:
        good = c.kp[c.score > KP_MIN_SCORE]
        if len(good) < 6:
            return None
        (x1, y1), (x2, y2) = good.min(0), good.max(0)
        pad_x, pad_y = 0.15 * (x2 - x1) + 10, 0.12 * (y2 - y1) + 10
        h, w = shape[:2]
        return np.array([max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)])

    def __call__(self, frame_bgr: np.ndarray) -> list[PersonCandidate]:
        boxes = None
        if self._n % self.det_every != 0 and self._prev:
            boxes = [b for c in self._prev if (b := self._box_from_kp(c, frame_bgr.shape)) is not None]
        if not boxes:
            boxes = self.det(frame_bgr)
        self._n += 1
        if boxes is None or len(boxes) == 0:
            self._prev = []
            return []
        boxes = np.asarray(boxes, float)
        kps, scores = self.pose(frame_bgr, bboxes=boxes)
        # RTMW SimCC maxima: ~0.5-1 on noise, 3-6 on clearly visible joints. Map to 0..1.
        scores = np.clip((scores - 1.0) / 3.0, 0, 1)
        out = [PersonCandidate(np.asarray(b, float), k[:NUM_KEYPOINTS].astype(float), s[:NUM_KEYPOINTS].astype(float))
               for b, k, s in zip(boxes, kps, scores)]
        out = [c for c in out if c.score.mean() > 0.15]   # drop boxes that lost their person
        self._prev = out
        return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def link_people(cands: list[list[PersonCandidate]], max_gap: int = 5) -> list[dict[int, PersonCandidate]]:
    """Greedy IoU tracker. Returns a list of tracks, each {frame: candidate}."""
    tracks: list[dict[int, PersonCandidate]] = []
    last: list[tuple[int, np.ndarray]] = []  # (frame, box) per track
    for t, frame_cands in enumerate(cands):
        used = set()
        order = sorted(range(len(frame_cands)), key=lambda i: -np.mean(frame_cands[i].score))
        for i in order:
            c = frame_cands[i]
            best, best_iou = None, 0.25
            for j, (lt, lb) in enumerate(last):
                if j in used or t - lt > max_gap or lt == t:
                    continue
                iou = _iou(c.box, lb)
                if iou > best_iou:
                    best, best_iou = j, iou
            if best is None:
                tracks.append({t: c})
                last.append((t, c.box))
                used.add(len(tracks) - 1)
            else:
                tracks[best][t] = c
                last[best] = (t, c.box)
                used.add(best)
    return tracks


def select_kicker(tracks: list[dict[int, PersonCandidate]], T: int, ball_xy: np.ndarray | None = None) -> dict[int, PersonCandidate]:
    """Pick the person most likely to be the shooter.

    Favour long, large, confident tracks; if a stationary ball position is known,
    strongly favour the person whose feet come closest to it.
    """
    if not tracks:
        return {}
    best, best_score = tracks[0], -np.inf
    for tr in tracks:
        boxes = np.array([c.box for c in tr.values()])
        heights = boxes[:, 3] - boxes[:, 1]
        conf = np.mean([c.score.mean() for c in tr.values()])
        s = (len(tr) / T) * 2 + conf + 0.5 * np.log1p(np.median(heights))
        if ball_xy is not None and not np.any(np.isnan(ball_xy)):
            feet = np.array([c.kp[15:23] for c in tr.values()]).reshape(-1, 2)
            d = np.nanmin(np.linalg.norm(feet - ball_xy, axis=1)) / (np.median(heights) + 1e-6)
            s += 3.0 * np.exp(-d * 3)
        if s > best_score:
            best, best_score = tr, s
    return best


class OneEuro:
    """One-Euro filter (Casiez et al. 2012) applied along time to an array."""

    def __init__(self, fps: float, min_cutoff: float = 1.7, beta: float = 0.3, d_cutoff: float = 1.0):
        self.fps, self.min_cutoff, self.beta, self.d_cutoff = fps, min_cutoff, beta, d_cutoff

    @staticmethod
    def _alpha(cutoff: float, fps: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * fps)

    def __call__(self, x: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """x: (T, ...). ``scale`` normalises units (e.g. person height in px)."""
        out = np.array(x, dtype=float, copy=True)
        flat = out.reshape(len(out), -1)
        prev = flat[0].copy()
        dprev = np.zeros_like(prev)
        a_d = self._alpha(self.d_cutoff, self.fps)
        for t in range(1, len(flat)):
            cur = flat[t]
            valid = ~np.isnan(cur) & ~np.isnan(prev)
            dx = np.where(valid, (cur - prev) * self.fps / scale, 0.0)
            dhat = a_d * dx + (1 - a_d) * dprev
            cutoff = self.min_cutoff + self.beta * np.abs(dhat)
            a = 1.0 / (1.0 + (1.0 / (2 * np.pi * cutoff)) * self.fps)
            filt = np.where(valid, a * cur + (1 - a) * prev, cur)
            flat[t] = filt
            prev = np.where(np.isnan(filt), prev, filt)
            dprev = dhat
        return flat.reshape(out.shape)


def clean_keypoints(kp: np.ndarray, score: np.ndarray, fps: float, body_px: float) -> np.ndarray:
    """Gate low-confidence points, fill short gaps, then One-Euro smooth."""
    kp = kp.copy()
    kp[score < KP_MIN_SCORE] = np.nan
    kp = interpolate_nans(kp, max_gap=max(2, int(0.15 * fps)))
    # High beta keeps the fast kicking foot responsive at contact.
    return OneEuro(fps, min_cutoff=2.0, beta=0.7)(kp, scale=max(body_px, 1.0))


# MediaPipe 33-landmark index for each of our 23 keypoints.
_MP_TO_OURS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 31, 29, 32, 32, 30]


class Pose3D:
    """MediaPipe world landmarks on a crop around the kicker."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        path = MODELS_DIR / "pose_landmarker_heavy.task"
        if not path.exists():
            _download(MP_MODEL_URL, path)
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1, min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3)
        self._mp = mp
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

    def __call__(self, frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        """Returns (23, 3) metric coords, x right / y up / z toward camera, or None."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        side = max(bw, bh) * 1.4
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        X1, Y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        X2, Y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        crop = frame_bgr[Y1:Y2, X1:X2]
        if crop.size == 0:
            return None
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(img)
        if not res.pose_world_landmarks:
            return None
        lm = res.pose_world_landmarks[0]
        pts = np.array([[lm[i].x, -lm[i].y, -lm[i].z] for i in _MP_TO_OURS], dtype=float)
        vis = np.array([getattr(lm[i], "visibility", 1.0) or 1.0 for i in _MP_TO_OURS])
        pts[vis < 0.2] = np.nan
        return pts


def _download(url: str, dest: Path) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
