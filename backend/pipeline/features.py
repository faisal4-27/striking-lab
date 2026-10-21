"""Biomechanical + outcome features for an instep drive.

Conventions
-----------
* 2D features use stabilised image coords; "forward" is the kick direction
  along image x (``dir_x`` = ±1), image y points down.
* Distances are normalised by leg length (hip→ankle) so they are scale-free.
* Joint angles prefer MediaPipe 3D world landmarks (view-invariant) and fall
  back to 2D when 3D is missing for that frame.
* Every feature may be ``None`` when it cannot be measured; scoring skips it.
"""
from __future__ import annotations

import numpy as np

from . import skeleton as S
from .events import Events, body_height_px, foot_point
from .geometry import angle_at
from .tracks import Tracks

FEATURE_NAMES = [
    "approach_angle_deg", "plant_fore_aft", "plant_knee_flex_deg", "backswing_knee_flex_deg",
    "backswing_thigh_deg", "trunk_lean_deg", "ankle_lock_deg", "foot_speed_mps", "arm_balance_deg",
    "follow_through_height", "momentum_forward", "head_over_ball", "knee_over_ball",
    "ball_speed_kmh", "launch_angle_deg", "direction_error_deg", "on_target",
]


def _f(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 3)


def _joint_angle(tr: Tracks, t: int, a: int, b: int, c: int) -> float:
    if tr.kp3d is not None and not np.any(np.isnan(tr.kp3d[t, [a, b, c]])):
        return float(angle_at(tr.kp3d[t, a], tr.kp3d[t, b], tr.kp3d[t, c]))
    return float(angle_at(tr.kp[t, a], tr.kp[t, b], tr.kp[t, c]))


def _lean_from_vertical(v: np.ndarray, dir_x: float) -> float:
    """Angle of image vector v from straight up, positive when tilted toward the kick direction."""
    return float(np.degrees(np.arctan2(v[0] * dir_x, -v[1])))


def body_frame(kp3d_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(up, right, forward) unit vectors of the body in MediaPipe world coords.

    'forward' is horizontal (perpendicular to world-up and the hip line)."""
    if kp3d_t is None or np.any(np.isnan(kp3d_t[[S.L_HIP, S.R_HIP, S.L_SHOULDER, S.R_SHOULDER]])):
        return None
    world_up = np.array([0.0, 1.0, 0.0])
    right = kp3d_t[S.R_HIP] - kp3d_t[S.L_HIP]
    right = right - (right @ world_up) * world_up
    right /= np.linalg.norm(right) + 1e-9
    fwd = np.cross(world_up, right)
    return world_up, right, fwd


def camera_view(tr: Tracks, c: int) -> tuple[str, float | None]:
    """How the camera sees the kicker at contact: angle between body-forward and the
    camera axis (0° = filmed from straight behind/in front, 90° = pure side view)."""
    if tr.kp3d is None:
        return "side", None
    for t in [c, *range(c - 3, c + 4)]:
        if 0 <= t < tr.T and (fr := body_frame(tr.kp3d[t])) is not None:
            fwd = fr[2]
            ang = float(np.degrees(np.arcsin(np.clip(np.hypot(fwd[0], fwd[1]), 0, 1))))
            view = "side" if ang >= 60 else "oblique" if ang >= 30 else "front/back"
            return view, round(ang, 1)
    return "side", None


def kick_direction(tr: Tracks, ev: Events, flight: dict, kp_s: np.ndarray) -> float:
    if flight.get("ball_dir_2d"):
        dx = flight["ball_dir_2d"][0]
        if abs(dx) > 1e-6:
            return float(np.sign(dx))
    hip = np.nanmean(kp_s[:, [S.L_HIP, S.R_HIP]], axis=1)
    d = hip[ev.contact, 0] - hip[ev.approach_start, 0]
    return float(np.sign(d)) if not np.isnan(d) and d != 0 else 1.0


def extract(tr: Tracks, ev: Events, flight: dict, target_out: dict, m_per_px: float) -> dict:
    kp_s = tr.kp_stab()
    c, fps, T = ev.contact, tr.fps, tr.T
    K, Sup = S.side(ev.kicking_foot), S.side(S.other(ev.kicking_foot))
    dir_x = kick_direction(tr, ev, flight, kp_s)
    view, view_angle = camera_view(tr, c)
    # Fore/aft distances and in-plane speeds are only meaningful when the kick
    # direction is roughly across the image.
    side_ok = view != "front/back"
    speed_gain = 1.0 / np.sin(np.radians(view_angle)) if view == "oblique" and view_angle else 1.0

    leg = np.nanmedian(np.concatenate([
        np.linalg.norm(kp_s[:, S.L_HIP] - kp_s[:, S.L_ANKLE], axis=1),
        np.linalg.norm(kp_s[:, S.R_HIP] - kp_s[:, S.R_ANKLE], axis=1)]))
    if np.isnan(leg) or leg <= 0:
        leg = body_height_px(tr.kp) * 0.5

    ball = np.array(ev.ball_rest) if ev.ball_rest is not None else None
    f: dict[str, float | None] = {}

    # Approach angle: hip path vs ball flight (in-plane; noisy in pure side view).
    hip = np.nanmean(kp_s[:, [S.L_HIP, S.R_HIP]], axis=1)
    appr = hip[ev.plant] - hip[ev.approach_start]
    if flight.get("ball_dir_2d") and np.linalg.norm(appr) > 0.3 * leg:
        bd = np.array(flight["ball_dir_2d"])
        cos = appr @ bd / (np.linalg.norm(appr) * np.linalg.norm(bd) + 1e-9)
        f["approach_angle_deg"] = _f(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    else:
        f["approach_angle_deg"] = None

    sup_foot = foot_point(kp_s, S.other(ev.kicking_foot))
    if not side_ok:
        f["plant_fore_aft"] = f["head_over_ball"] = f["knee_over_ball"] = None
    elif ball is not None:
        f["plant_fore_aft"] = _f((sup_foot[c, 0] - ball[0]) * dir_x / leg)
        f["head_over_ball"] = _f((kp_s[c, S.NOSE, 0] - ball[0]) * dir_x / leg)
        f["knee_over_ball"] = _f((kp_s[c, K["knee"], 0] - ball[0]) * dir_x / leg)
    else:
        # Without a ball, measure relative to the kicking foot at contact.
        kf = foot_point(kp_s, ev.kicking_foot)[c]
        f["plant_fore_aft"] = _f((sup_foot[c, 0] - kf[0]) * dir_x / leg)
        f["head_over_ball"] = _f((kp_s[c, S.NOSE, 0] - kf[0]) * dir_x / leg)
        f["knee_over_ball"] = _f((kp_s[c, K["knee"], 0] - kf[0]) * dir_x / leg)

    f["plant_knee_flex_deg"] = _f(180 - _joint_angle(tr, c, Sup["hip"], Sup["knee"], Sup["ankle"]))
    f["backswing_knee_flex_deg"] = _f(180 - _joint_angle(tr, ev.backswing, K["hip"], K["knee"], K["ankle"]))

    # Thigh behind the body at backswing (positive = behind) and trunk lean at
    # contact (positive = forward): 3D body frame when available, else 2D side view.
    b, fr_b = ev.backswing, body_frame(tr.kp3d[ev.backswing]) if tr.kp3d is not None else None
    if fr_b is not None and not np.any(np.isnan(tr.kp3d[b, [K["hip"], K["knee"]]])):
        up, _, fwd = fr_b
        thigh3 = tr.kp3d[b, K["knee"]] - tr.kp3d[b, K["hip"]]
        f["backswing_thigh_deg"] = _f(np.degrees(np.arctan2(-thigh3 @ fwd, -thigh3 @ up)))
    elif side_ok:
        thigh = kp_s[b, K["knee"]] - kp_s[b, K["hip"]]
        f["backswing_thigh_deg"] = _f(np.degrees(np.arctan2(-thigh[0] * dir_x, thigh[1])))
    else:
        f["backswing_thigh_deg"] = None

    fr_c = body_frame(tr.kp3d[c]) if tr.kp3d is not None else None
    if fr_c is not None:
        up, _, fwd = fr_c
        p3 = tr.kp3d[c]
        torso3 = np.nanmean(p3[[S.L_SHOULDER, S.R_SHOULDER]], axis=0) - np.nanmean(p3[[S.L_HIP, S.R_HIP]], axis=0)
        f["trunk_lean_deg"] = _f(np.degrees(np.arctan2(torso3 @ fwd, torso3 @ up)))
    elif side_ok:
        sh = np.nanmean(kp_s[c, [S.L_SHOULDER, S.R_SHOULDER]], axis=0)
        torso = sh - hip[c]
        f["trunk_lean_deg"] = _f(_lean_from_vertical(torso, dir_x)) if not np.any(np.isnan(torso)) else None
    else:
        f["trunk_lean_deg"] = None

    f["ankle_lock_deg"] = _f(_joint_angle(tr, c, K["knee"], K["ankle"], K["toe"]))

    toe = kp_s[:, K["toe"]]
    v = np.linalg.norm(np.diff(toe, axis=0), axis=1) * fps * m_per_px
    lo = max(0, c - 3)
    f["foot_speed_mps"] = (_f(np.nanmax(v[lo:c + 1]) * speed_gain)
                           if side_ok and c > 0 and np.any(~np.isnan(v[lo:c + 1])) else None)

    # Opposite arm out for balance: angle between upper arm and torso side.
    ArmS = Sup  # the arm on the support-leg side is the one that swings out
    f["arm_balance_deg"] = _f(_joint_angle(tr, c, ArmS["elbow"], ArmS["shoulder"], ArmS["hip"]))

    fy = foot_point(kp_s, ev.kicking_foot)[ev.follow_through, 1]
    f["follow_through_height"] = _f((hip[ev.follow_through, 1] - fy) / leg)

    after = min(T - 1, c + int(0.3 * fps))
    f["momentum_forward"] = _f((hip[after, 0] - hip[c, 0]) * dir_x / leg) if after > c and side_ok else None
    f["_view"], f["_view_angle"] = view, view_angle

    f["ball_speed_kmh"] = _f(flight.get("ball_speed_kmh"))
    f["launch_angle_deg"] = _f(flight.get("launch_angle_deg"))
    f["direction_error_deg"] = _f(target_out.get("direction_error_deg"))
    ot = target_out.get("on_target")
    f["on_target"] = None if ot is None else float(bool(ot))
    out = {k: f.get(k) for k in FEATURE_NAMES}
    out["camera_view"], out["camera_view_angle"] = f["_view"], f["_view_angle"]
    return out


def quality(tr: Tracks, ev: Events, stab_stats: dict, n_features: int) -> dict:
    """0..1 confidence in the analysis, with the parts that drove it."""
    c, w = ev.contact, int(0.25 * tr.fps)
    lo, hi = max(0, c - w), min(tr.T, c + w + 1)
    legs = [S.L_HIP, S.R_HIP, S.L_KNEE, S.R_KNEE, S.L_ANKLE, S.R_ANKLE, S.L_BIG_TOE, S.R_BIG_TOE]
    pose_q = float(np.mean(tr.kp_score[lo:hi][:, legs])) if hi > lo else 0.0
    ball_q = float(np.mean(~np.isnan(tr.ball[lo:hi, 0]))) if hi > lo else 0.0
    shake = stab_stats.get("mean_shift_px", 0.0) / max(tr.width, 1) * 100   # % of width per frame
    cam_q = float(np.clip(1 - shake / 3.0, 0, 1))
    feat_q = min(1.0, n_features / (len(FEATURE_NAMES) - 2))   # target features are optional
    event_q = {"ball": 1.0, "manual": 1.0, "foot+ball": 0.8, "foot": 0.6}[ev.contact_source]
    overall = 0.35 * pose_q + 0.25 * ball_q + 0.15 * cam_q + 0.15 * feat_q + 0.10 * event_q
    return {"overall": round(overall, 3), "pose": round(pose_q, 3), "ball": round(ball_q, 3),
            "camera_steadiness": round(cam_q, 3), "features": round(feat_q, 3), "events": event_q}
