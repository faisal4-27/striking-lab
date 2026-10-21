"""Metric scale and ball-flight reconstruction.

Single-camera, no calibration board, so we lean on known sizes:
  * a size-5 ball is 22 cm across → depth from apparent radius (pinhole model)
  * the player's height (default 1.75 m, user-configurable) → px→m scale

Ball flight after contact is fitted in camera 3D coordinates:
    Z_t = f · R / r_t,  X_t = (x_t − cx) · Z_t / f,  Y_t = −(y_t − cy) · Z_t / f
with X, Z linear in time and Y including gravity. Depth from radius is noisy,
so the depth component is only trusted when it fits well; otherwise we fall
back to in-plane speed (a lower bound).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .geometry import angle_between, apply_transform
from .tracks import Tracks

BALL_RADIUS_M = 0.11
G = 9.81


def focal_px(width: int, height: int, hfov_deg: float = 66.0) -> float:
    # hfov applies to the long side of the frame (phone video in either orientation)
    long_side = max(width, height)
    return long_side / (2 * np.tan(np.radians(hfov_deg) / 2))


def metres_per_px(tr: Tracks, contact: int, body_px: float, player_height_m: float = 1.75) -> tuple[float, str]:
    """Scale in reference-frame pixels near the contact frame."""
    s_body = player_height_m / body_px
    lo = max(0, contact - int(0.6 * tr.fps))
    r = tr.ball_r[lo:contact + 1]
    r = r[~np.isnan(r) & (tr.ball_conf[lo:contact + 1] > 0.2)]
    if len(r) >= 3:
        s_ball = BALL_RADIUS_M / float(np.median(r))
        if 0.5 < s_ball / s_body < 2.0:
            return float(np.sqrt(s_ball * s_body)), "ball+body"
    return s_body, "body"


def _fit_flight(t: np.ndarray, P: np.ndarray):
    """Robust fit: X,Z linear, Y with gravity. Returns (v (3,), residual_rms, z_r2)."""
    def resid(p):
        x0, vx, y0, vy, z0, vz = p
        return np.concatenate([
            P[:, 0] - (x0 + vx * t),
            P[:, 1] - (y0 + vy * t - 0.5 * G * t ** 2),
            (P[:, 2] - (z0 + vz * t)) * 0.5,   # depth is noisier: down-weight
        ])
    p0 = np.array([P[0, 0], 0, P[0, 1], 0, P[0, 2], 0.0])
    sol = least_squares(resid, p0, loss="soft_l1", f_scale=0.3)
    x0, vx, y0, vy, z0, vz = sol.x
    zpred = z0 + vz * t
    ss_res = np.sum((P[:, 2] - zpred) ** 2)
    ss_tot = np.sum((P[:, 2] - P[:, 2].mean()) ** 2)
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    # Depth essentially constant (ball moving across the view): the fit is trivially good.
    r2 = 1.0 if ss_tot < 1e-6 * len(t) else float(1 - ss_res / ss_tot)
    return np.array([vx, vy, vz]), rms, r2


def ball_flight(tr: Tracks, contact: int, m_per_px: float, hfov_deg: float = 66.0) -> dict:
    """Speed (km/h), launch angle and direction of the ball just after contact."""
    fps = tr.fps
    hi = min(tr.T, contact + 1 + int(0.35 * fps))
    idx = np.arange(contact + 1, hi)
    ok = ~np.isnan(tr.ball[idx, 0]) & ~np.isnan(tr.ball_r[idx]) & (tr.ball_conf[idx] > 0.05)
    idx = idx[ok]
    out: dict = {"ball_frames_after_contact": int(len(idx))}
    if len(idx) < 3:
        return out

    t = (idx - contact) / fps
    # In-plane estimate from stabilised coordinates (robust lower bound).
    bs = np.array([apply_transform(tr.ball[i], tr.transforms[i]) for i in idx])
    A = np.c_[np.ones_like(t), t]
    vxy = np.linalg.lstsq(A, bs, rcond=None)[0][1] * m_per_px   # m/s, image y down
    v2d = np.array([vxy[0], -vxy[1]])
    speed_2d = float(np.linalg.norm(v2d))

    # 3D estimate from apparent size, in stabilised coords so camera pan/zoom
    # doesn't leak into the ball's velocity.
    f = focal_px(tr.width, tr.height, hfov_deg)
    cx, cy = tr.width / 2, tr.height / 2
    r = np.maximum(tr.ball_r[idx] * np.array([tr.scale_at(i) for i in idx]), 1.0)
    Z = f * BALL_RADIUS_M / r
    X = (bs[:, 0] - cx) * Z / f
    Y = -(bs[:, 1] - cy) * Z / f
    v3, rms, z_r2 = _fit_flight(t, np.c_[X, Y, Z])
    speed_3d = float(np.linalg.norm(v3))

    use_3d = len(idx) >= 5 and z_r2 > 0.6 and speed_3d < 50 and speed_3d >= speed_2d * 0.9
    if use_3d:
        speed, v, method = speed_3d, v3, "3d"
        launch = float(np.degrees(np.arctan2(v[1], np.hypot(v[0], v[2]))))
    else:
        speed, method = speed_2d, "2d"
        launch = float(np.degrees(np.arctan2(v2d[1], abs(v2d[0]) + 1e-9)))

    out.update({
        "ball_speed_kmh": round(speed * 3.6, 1),
        "ball_speed_2d_kmh": round(speed_2d * 3.6, 1),
        "ball_speed_3d_kmh": round(speed_3d * 3.6, 1),
        "speed_method": method,
        "depth_fit_r2": round(z_r2, 3),
        "launch_angle_deg": round(launch, 1),
        "ball_dir_2d": [float(vxy[0]), float(vxy[1])],   # stabilised image direction
    })
    return out


def target_outcome(tr: Tracks, contact: int, target: dict | None, ball_dir_2d: list[float] | None) -> dict:
    """target = {'frame': k, 'points': [[x,y],...]} in normalised image coords of frame k."""
    if not target or not target.get("points"):
        return {}
    k = int(min(max(target.get("frame", 0), 0), tr.T - 1))
    pts = np.array(target["points"], float) * [tr.width, tr.height]
    poly = apply_transform(pts, tr.transforms[k]).astype(np.float32)
    centre = poly.mean(axis=0)
    out: dict = {}

    import cv2
    hi = min(tr.T, contact + int(2.0 * tr.fps))
    hit = None
    prev = None
    for i in range(contact, hi):
        if np.isnan(tr.ball[i, 0]):
            continue
        p = apply_transform(tr.ball[i], tr.transforms[i])
        # a fast ball can jump >1 m between frames: test the whole segment
        seg = [p] if prev is None else [prev + (p - prev) * a for a in np.linspace(0, 1, 12)]
        if i > contact and any(cv2.pointPolygonTest(poly, (float(q[0]), float(q[1])), False) >= 0 for q in seg):
            hit = True
            break
        prev = p
    # Only claim a miss if the ball was tracked long enough to reach the target.
    tracked = np.sum(~np.isnan(tr.ball[contact + 1:hi, 0]))
    out["on_target"] = True if hit else (False if tracked > 0.5 * tr.fps else None)

    if ball_dir_2d is not None and not np.any(np.isnan(tr.ball[contact])):
        start = apply_transform(tr.ball[contact], tr.transforms[contact])
        out["direction_error_deg"] = round(float(angle_between(np.array(ball_dir_2d), centre - start)), 1)
    return out
