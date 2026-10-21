"""Find the key phases of the kick.

All thresholds are expressed in *body heights per second* so they work at any
resolution / camera distance. 1 body height ≈ 1.75 m, so e.g. a 25 m/s shot
is ~14 bh/s and a planted foot moves < 1 bh/s.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import skeleton as S
from .geometry import angle_at, moving_average, nan_argmax, nan_argmin
from .tracks import Tracks


@dataclass
class Events:
    kicking_foot: str        # 'left' | 'right'
    contact: int
    plant: int
    backswing: int
    follow_through: int
    approach_start: int
    contact_source: str      # 'ball' (ball onset) | 'foot+ball' (fast foot reaches ball) | 'foot' | 'manual'
    ball_rest: list[float] | None   # stabilised ball position at contact

    def to_dict(self) -> dict:
        return asdict(self)


def body_height_px(kp: np.ndarray) -> float:
    """Median nose→lower-ankle distance, scaled up to full height."""
    nose_y = kp[:, S.NOSE, 1]
    ankle_y = np.nanmax(kp[:, [S.L_ANKLE, S.R_ANKLE], 1], axis=1)
    h = np.nanmedian(ankle_y - nose_y) if np.any(~np.isnan(nose_y - ankle_y)) else np.nan
    if np.isnan(h) or h <= 0:
        sh = np.nanmedian(np.linalg.norm(kp[:, S.L_SHOULDER] - kp[:, S.L_ANKLE], axis=1))
        h = sh * 1.1 if not np.isnan(sh) else 300.0
    return float(h * 1.12)


def foot_point(kp: np.ndarray, foot: str) -> np.ndarray:
    j = S.side(foot)
    return np.nanmean(np.stack([kp[:, j["ankle"]], kp[:, j["toe"]]]), axis=0)


def _speed(p: np.ndarray, fps: float) -> np.ndarray:
    d = np.full(len(p), np.nan)
    d[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) * fps
    return d


def _clip(i: int, T: int) -> int:
    return int(min(max(i, 0), T - 1))


def ball_onset(ball_s: np.ndarray, fps: float, bh: float) -> int | None:
    """Last frame before the ball starts moving fast (≈ contact)."""
    v = _speed(ball_s, fps) / bh
    T = len(v)
    best, best_jump = None, 0.0
    for t in range(3, T - 3):
        after = np.nanmean(v[t + 1:t + 4]) if np.any(~np.isnan(v[t + 1:t + 4])) else np.nan
        before = np.nanmean(v[t - 2:t + 1]) if np.any(~np.isnan(v[t - 2:t + 1])) else 0.0
        if np.isnan(after) or after < 4.0:
            continue
        jump = after - (before if not np.isnan(before) else 0.0)
        if jump > best_jump:
            best, best_jump = t, jump
    return best


def foot_meets_ball(feet: dict, fspeed: dict, ball_s: np.ndarray, bh: float, fps: float,
                    foot_hint: str | None = None) -> tuple[str, int] | None:
    """Contact when the ball's flight wasn't tracked.

    1. Ball vanishes: a ball that sat still for a while disappears for good with a
       foot right next to it (typical when the shot goes away from the camera or
       is too blurred to detect).
    2. Otherwise: the frame where a fast-moving foot is closest to the ball.
    """
    feet_ok = [foot_hint] if foot_hint in ("left", "right") else ["left", "right"]
    valid = np.flatnonzero(~np.isnan(ball_s[:, 0]))
    T = len(ball_s)
    if len(valid) >= int(0.3 * fps):
        e = int(valid[-1])
        if e < T - int(0.2 * fps):
            lo = max(0, e - int(0.15 * fps))
            best = None
            for f in feet_ok:
                d = np.linalg.norm(feet[f][lo:e + 1] - ball_s[e], axis=1) / bh
                if np.any(~np.isnan(d)):
                    i = int(np.nanargmin(d))
                    if d[i] < 0.25 and (best is None or d[i] < best[2]):
                        best = (f, lo + i, d[i])
            if best is not None:
                # the kicking foot is the one moving fastest into the ball
                f_fast = max(feet_ok, key=lambda f: np.nanmax(fspeed[f][lo:e + 1]))
                return f_fast, best[1]
    best = None
    for f in ([foot_hint] if foot_hint in ("left", "right") else ["left", "right"]):
        d = np.linalg.norm(feet[f] - ball_s, axis=1) / bh
        near = np.flatnonzero(d < 0.2)
        for t in near:
            sp = fspeed[f][t]
            if sp > 3.0 and (best is None or sp > best[2]):
                best = (f, int(t), sp)
    return None if best is None else best[:2]


def detect(tr: Tracks, manual_contact: int | None = None, manual_plant: int | None = None,
           foot_hint: str | None = None) -> Events:
    fps, T = tr.fps, tr.T
    kp = tr.kp_stab()
    ball_s = tr.ball_stab()
    bh = body_height_px(tr.kp)

    feet = {f: foot_point(kp, f) for f in ("left", "right")}
    fspeed = {f: moving_average(np.nan_to_num(_speed(p, fps) / bh), 3) for f, p in feet.items()}

    onset = ball_onset(ball_s, fps, bh)
    w = lambda sec: max(1, int(round(sec * fps)))  # noqa: E731

    # --- kicking foot -------------------------------------------------------
    if foot_hint in ("left", "right"):
        kick = foot_hint
    else:
        lo, hi = (onset - w(0.3), onset + w(0.1)) if onset is not None else (0, T)
        lo, hi = _clip(lo, T), _clip(hi, T) + 1
        kick = max(("left", "right"), key=lambda f: np.max(fspeed[f][lo:hi]) if hi > lo else 0)
    sup = S.other(kick)

    # --- contact ------------------------------------------------------------
    source = "ball"
    if manual_contact is not None:
        contact, source = _clip(manual_contact, T), "manual"
    elif onset is not None:
        # refine: frame near onset where the kicking foot is closest to the ball
        lo, hi = _clip(onset - w(0.1), T), _clip(onset + w(0.03), T) + 1
        d = np.linalg.norm(feet[kick][lo:hi] - ball_s[onset], axis=1)
        contact = lo + nan_argmin(d, default=onset - lo)
    elif (meet := foot_meets_ball(feet, fspeed, ball_s, bh, fps, foot_hint)) is not None:
        # Ball blurred out of detection at the kick: use the fast foot arriving at the ball.
        kick, contact, source = meet[0], meet[1], "foot+ball"
        sup = S.other(kick)
    else:
        contact, source = nan_argmax(fspeed[kick], default=T // 2), "foot"

    ball_rest = ball_s[contact] if not np.any(np.isnan(ball_s[contact])) else None

    # --- plant: support foot comes to rest before contact ---------------------
    if manual_plant is not None:
        plant = _clip(manual_plant, T)
    else:
        sp = fspeed[sup]
        t = contact
        floor = contact - w(0.5)
        while t > max(floor, 0) and sp[t] < 1.2:
            t -= 1
        plant = _clip(t + 1, T)
        if plant >= contact:
            plant = _clip(contact - w(0.1), T)

    # --- backswing: peak kicking-knee flexion (smallest knee angle) ----------
    j = S.side(kick)
    knee = angle_at(tr.kp[:, j["hip"]], tr.kp[:, j["knee"]], tr.kp[:, j["ankle"]])
    if tr.kp3d is not None:
        # 3D angles are view-independent; use them wherever MediaPipe produced a pose
        k3 = angle_at(tr.kp3d[:, j["hip"]], tr.kp3d[:, j["knee"]], tr.kp3d[:, j["ankle"]])
        knee = np.where(np.isnan(k3), knee, k3)
    lo = _clip(contact - w(0.4), T)
    backswing = lo + nan_argmin(knee[lo:contact + 1], default=contact - lo)

    # --- follow-through: kicking foot reaches its highest point --------------
    hi = _clip(contact + w(0.6), T)
    fy = feet[kick][contact:hi + 1, 1]
    follow = contact + nan_argmin(fy, default=0)

    # --- approach start: hips start moving towards the ball ------------------
    hip = np.nanmean(kp[:, [S.L_HIP, S.R_HIP]], axis=1)
    hs = moving_average(np.nan_to_num(_speed(hip, fps) / bh), 5)
    t = plant
    floor = plant - w(3.0)
    while t > max(floor, 0) and hs[t] > 0.35:
        t -= 1
    approach = _clip(t, T)

    return Events(kick, int(contact), int(plant), int(backswing), int(follow), int(approach), source,
                  [float(x) for x in ball_rest] if ball_rest is not None else None)
