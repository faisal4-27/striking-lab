"""End-to-end analysis of one clip.

perception (slow, GPU)                     reasoning (fast, numpy)
──────────────────────                     ─────────────────────────
pass 1: every frame                        events → calibration → ball flight
  RTMW 2D pose (all people)                → target → features → quality
  YOLO ball candidates                     → rules ⊕ model → feedback report
  ORB camera motion
person tracking + kicker choice
ball trajectory reconstruction
pass 2: frames around the kick
  MediaPipe 3D landmarks

Perception results are cached in tracks.npz, so re-scoring after you mark a
target or correct the contact frame takes well under a second.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import calibration, events as events_mod, features as features_mod
from .ball import reconstruct
from .events import body_height_px
from .pose import clean_keypoints, link_people, select_kicker
from .skeleton import EDGES, NUM_KEYPOINTS
from .stabilize import Stabilizer
from .tracks import Tracks
from .video import iter_frames

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]


class Models:
    """Lazily-loaded, process-wide perception models (keep GPU memory warm)."""
    _pose2d = _pose3d = _ball = None

    @classmethod
    def pose2d(cls):
        if cls._pose2d is None:
            from .pose import Pose2D
            cls._pose2d = Pose2D()
        return cls._pose2d

    @classmethod
    def pose3d(cls):
        if cls._pose3d is None:
            from .pose import Pose3D
            cls._pose3d = Pose3D()
        return cls._pose3d

    @classmethod
    def ball(cls):
        if cls._ball is None:
            from .ball import BallDetector
            cls._ball = BallDetector()
        return cls._ball


def perceive(video: Path, info: dict, progress: Progress) -> tuple[Tracks, dict]:
    fps, W, H, n = info["fps"], info["width"], info["height"], max(info["frames"], 1)
    pose2d, ball_det = Models.pose2d(), Models.ball()
    pose2d.reset()
    stab = Stabilizer()

    cands, balls = [], []
    for i, frame in iter_frames(video):
        people = pose2d(frame)
        bdets = ball_det(frame)
        fg = [p.box for p in people] + [np.r_[b.xy - 2 * b.r, b.xy + 2 * b.r] for b in bdets]
        stab.update(frame, fg)
        cands.append(people)
        balls.append(bdets)
        if i % 5 == 0:
            progress(0.05 + 0.7 * i / n, "Tracking body and ball")
    T = len(cands)
    if T == 0:
        raise ValueError("Could not decode any frames from the video")
    transforms, stab_stats = stab.result()

    tr = Tracks.empty(T, fps, W, H)
    tr.transforms = transforms

    tracks = link_people(cands)
    if not tracks:
        raise ValueError("No person detected in the video")
    kicker = select_kicker(tracks, T)

    def fill(person_track):
        tr.kp[:] = np.nan
        tr.kp_score[:] = 0
        tr.person_box[:] = np.nan
        for t, c in person_track.items():
            tr.kp[t], tr.kp_score[t], tr.person_box[t] = c.kp, c.score, c.box

    fill(kicker)
    bh = body_height_px(tr.kp)
    tr.ball, tr.ball_r, tr.ball_conf = reconstruct(balls, transforms, fps, bh)

    # Re-pick the kicker now that we know where the ball sits before the kick.
    first = np.flatnonzero(~np.isnan(tr.ball[:, 0]))
    if len(first) and len(tracks) > 1:
        kicker = select_kicker(tracks, T, ball_xy=tr.ball[first[0]])
        fill(kicker)
        bh = body_height_px(tr.kp)

    tr.kp = clean_keypoints(tr.kp, tr.kp_score, fps, bh)
    tr.meta["n_people_tracks"] = len(tracks)
    return tr, stab_stats


def lift_3d(video: Path, tr: Tracks, ev: events_mod.Events, progress: Progress) -> None:
    w = int(0.6 * tr.fps)
    lo, hi = max(0, ev.contact - w), min(tr.T - 1, ev.contact + w)
    frames = {t for t in range(lo, hi + 1) if not np.any(np.isnan(tr.person_box[t]))}
    if not frames:
        return
    pose3d = Models.pose3d()
    kp3d = np.full((tr.T, NUM_KEYPOINTS, 3), np.nan)
    for k, (i, frame) in enumerate(iter_frames(video, only=frames)):
        p = pose3d(frame, tr.person_box[i])
        if p is not None:
            kp3d[i] = p
        if k % 5 == 0:
            progress(0.78 + 0.15 * k / len(frames), "Measuring joint angles in 3D")
    tr.kp3d = kp3d


def score(tr: Tracks, stab_stats: dict, target: dict | None = None, manual: dict | None = None,
          player_height_m: float = 1.75, foot_hint: str | None = None) -> dict:
    """Pure reasoning stage; cheap enough to rerun on every correction."""
    from scoring.feedback import build_report

    manual = manual or {}
    ev = events_mod.detect(tr, manual_contact=manual.get("contact"), manual_plant=manual.get("plant"),
                           foot_hint=foot_hint)
    bh = body_height_px(tr.kp)
    m_per_px, scale_src = calibration.metres_per_px(tr, ev.contact, bh, player_height_m)
    flight = calibration.ball_flight(tr, ev.contact, m_per_px)
    tgt = calibration.target_outcome(tr, ev.contact, target, flight.get("ball_dir_2d"))
    feats = features_mod.extract(tr, ev, flight, tgt, m_per_px)
    n_feats = sum(feats.get(k) is not None for k in features_mod.FEATURE_NAMES)
    q = features_mod.quality(tr, ev, stab_stats, n_feats)
    q["view"], q["view_angle"] = feats["camera_view"], feats["camera_view_angle"]
    try:
        from scoring import seq
        seq_pred = seq.predict(tr, ev.to_dict())
    except Exception:  # a broken optional model must never break analysis
        log.exception("sequence model failed")
        seq_pred = None
    report = build_report(feats, q, seq_pred)
    return {
        "events": ev.to_dict(),
        "calibration": {"m_per_px": m_per_px, "source": scale_src, "body_px": bh,
                        "player_height_m": player_height_m},
        "flight": flight, "target": tgt, "features": feats, "quality": q,
        "camera": stab_stats, "report": report,
    }


def overlay(tr: Tracks) -> dict:
    """Compact per-frame data for drawing the skeleton + ball on the video."""
    def r(a):
        return [None if np.isnan(v) else round(float(v), 1) for v in a]
    return {
        "fps": tr.fps, "width": tr.width, "height": tr.height, "edges": EDGES,
        "kp": [[r(p) for p in frame] for frame in tr.kp],
        "ball": [r(b) + [None if np.isnan(rad) else round(float(rad), 1)] for b, rad in zip(tr.ball, tr.ball_r)],
    }


def analyze(video: Path, info: dict, out_dir: Path, target: dict | None = None, manual: dict | None = None,
            player_height_m: float = 1.75, foot_hint: str | None = None,
            progress: Progress = lambda p, m: None, reuse: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_path, cam_path = out_dir / "tracks.npz", out_dir / "camera.json"
    t0 = time.time()
    if reuse and tracks_path.exists() and cam_path.exists():
        tr = Tracks.load(tracks_path)
        stab_stats = json.loads(cam_path.read_text())
        progress(0.9, "Re-scoring")
    else:
        progress(0.05, "Tracking body and ball")
        tr, stab_stats = perceive(video, info, progress)
        cam_path.write_text(json.dumps(stab_stats))

    # 3D landmarks only cover a window around contact; (re)compute if the contact
    # moved, e.g. after a manual correction or an improved event detector.
    ev = events_mod.detect(tr, manual_contact=(manual or {}).get("contact"), foot_hint=foot_hint)
    if tr.kp3d is None or np.all(np.isnan(tr.kp3d[ev.contact])):
        progress(0.78, "Measuring joint angles in 3D")
        lift_3d(video, tr, ev, progress)
        tr.save(tracks_path)
    elif not tracks_path.exists():
        tr.save(tracks_path)

    progress(0.95, "Scoring")
    result = score(tr, stab_stats, target, manual, player_height_m, foot_hint)
    result["video"] = info
    result["overlay"] = overlay(tr)
    result["elapsed_s"] = round(time.time() - t0, 2)
    result = to_jsonable(result)
    (out_dir / "analysis.json").write_text(json.dumps(result, allow_nan=False))
    progress(1.0, "Done")
    return result


def to_jsonable(o):
    """numpy → python, NaN/inf → None (browsers reject NaN in JSON)."""
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, np.integer):
        return int(o)
    return o
