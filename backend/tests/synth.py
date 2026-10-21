"""Procedural instep-drive generator with known ground truth.

Side view, kicker runs left→right (+x), right-footed. Image y points down.
Every quantity the features measure is a parameter here, so tests can assert
that the pipeline recovers it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline import skeleton as S
from pipeline.tracks import Tracks


@dataclass
class KickParams:
    fps: float = 60.0
    T: int = 120
    contact: int = 60
    plant: int = 54
    backswing: int = 50
    height_px: float = 400.0          # body height in px (≈1.75 m)
    approach_mps: float = 3.0
    trunk_lean_deg: float = 10.0      # + forward
    ankle_deg: float = 160.0          # knee-ankle-toe at contact
    backswing_knee_flex: float = 110.0
    plant_knee_flex: float = 25.0
    plant_offset_leg: float = -0.05   # support foot vs ball, leg lengths (+ ahead)
    arm_abduction_deg: float = 70.0
    ball_speed_mps: float = 25.0
    launch_deg: float = 8.0
    camera_pan_px_per_frame: float = 0.0


def _ik_knee(hip, ankle, a, b, forward=1.0):
    d = np.linalg.norm(ankle - hip)
    d = min(d, a + b - 1e-6)
    x = (a * a - b * b + d * d) / (2 * d)
    h = np.sqrt(max(a * a - x * x, 0))
    u = (ankle - hip) / d
    perp = np.array([-u[1], u[0]])
    k1, k2 = hip + x * u + h * perp, hip + x * u - h * perp
    return k1 if (k1[0] - k2[0]) * forward > 0 else k2


def _ray(origin, angle_from_down_deg, length):
    """Point at `length` from origin, angle measured from straight down, + toward +x."""
    a = np.radians(angle_from_down_deg)
    return origin + length * np.array([np.sin(a), np.cos(a)])


def make_kick(p: KickParams = KickParams()) -> tuple[Tracks, dict]:
    H = p.height_px
    px_per_m = H / 1.75
    a = b = 0.245 * H            # thigh / shank
    torso, neck = 0.30 * H, 0.12 * H
    upper, fore, foot = 0.17 * H, 0.15 * H, 0.10 * H
    v = p.approach_mps * px_per_m / p.fps

    hip_y = 300.0
    knee_angle_sup = 180 - p.plant_knee_flex
    d_sup = np.sqrt(a * a + b * b - 2 * a * b * np.cos(np.radians(knee_angle_sup)))

    # kicking leg schedule: thigh angle from down (+ forward), knee flexion
    def kick_leg(t):
        keys_t = [0, p.backswing, p.contact, p.contact + 10, p.T]
        thigh = np.interp(t, keys_t, [0, -30, 10, 70, 20])
        flex = np.interp(t, keys_t, [20, p.backswing_knee_flex, 10, 0, 10])
        return thigh, flex

    hip_c = 600.0
    hip_x = lambda t: hip_c + v * (t - p.contact) * (1.0 if t <= p.contact else 0.6)  # noqa: E731

    kp = np.full((p.T, S.NUM_KEYPOINTS, 2), np.nan)
    # contact geometry first, to place the ball at the toe
    def leg_points(t):
        hip = np.array([hip_x(t), hip_y])
        thigh, flex = kick_leg(t)
        knee = _ray(hip, thigh, a)
        ankle = _ray(knee, thigh - flex, b)
        toe_dir = thigh - flex + (180 - (p.ankle_deg if abs(t - p.contact) <= 2 else 110))
        toe = _ray(ankle, toe_dir, foot)
        return hip, knee, ankle, toe

    _, _, _, toe_c = leg_points(p.contact)
    r_ball = 0.11 * px_per_m
    ball_rest = toe_c + np.array([r_ball * 0.8, 0.0])
    leg_len = np.hypot(a, b) * 1.4  # ~ hip-ankle in standing
    plant_x = ball_rest[0] + p.plant_offset_leg * (a + b)
    # ground height chosen so the support knee has the requested flexion at contact
    ground = hip_y + np.sqrt(max(d_sup ** 2 - (plant_x - hip_c) ** 2, 1.0))

    for t in range(p.T):
        hip, knee, ankle, toe = leg_points(t)
        # support (left) foot: stride in, then planted
        sup_x = plant_x - 2 * v * min(max(p.plant - t, 0), 15)   # swing in, then planted
        sup_ankle = np.array([sup_x, ground])
        sup_knee = _ik_knee(hip, sup_ankle, a, b, forward=1.0)
        sup_toe = sup_ankle + np.array([foot, 0.0])

        lean = np.radians(p.trunk_lean_deg)
        sh = hip + torso * np.array([np.sin(lean), -np.cos(lean)])
        nose = sh + neck * np.array([np.sin(lean), -np.cos(lean)])
        # support-side arm abducted forward/out, kicking-side arm hanging back
        l_elbow = sh + upper * np.array([np.sin(np.radians(p.arm_abduction_deg - p.trunk_lean_deg)),
                                         np.cos(np.radians(p.arm_abduction_deg - p.trunk_lean_deg))])
        l_wrist = l_elbow + fore * np.array([np.sin(np.radians(p.arm_abduction_deg)), np.cos(np.radians(p.arm_abduction_deg))])
        r_elbow = _ray(sh, -20, upper)
        r_wrist = _ray(r_elbow, -10, fore)

        f = kp[t]
        f[S.NOSE] = nose
        f[S.L_EYE] = f[S.R_EYE] = nose + [3, -4]
        f[S.L_EAR] = f[S.R_EAR] = nose + [-6, 0]
        f[S.L_SHOULDER] = f[S.R_SHOULDER] = sh
        f[S.L_ELBOW], f[S.L_WRIST] = l_elbow, l_wrist
        f[S.R_ELBOW], f[S.R_WRIST] = r_elbow, r_wrist
        f[S.L_HIP] = f[S.R_HIP] = hip
        f[S.L_KNEE], f[S.L_ANKLE] = sup_knee, sup_ankle
        f[S.L_BIG_TOE] = f[S.L_SMALL_TOE] = sup_toe
        f[S.L_HEEL] = sup_ankle - [0.04 * H, 0]
        f[S.R_KNEE], f[S.R_ANKLE] = knee, ankle
        f[S.R_BIG_TOE] = f[S.R_SMALL_TOE] = toe
        f[S.R_HEEL] = ankle - (toe - ankle) * 0.4

    # ball: at rest until contact, then ballistic (in-plane)
    ball = np.zeros((p.T, 2))
    vb = p.ball_speed_mps * px_per_m
    g = 9.81 * px_per_m
    for t in range(p.T):
        if t <= p.contact:
            ball[t] = ball_rest
        else:
            dt = (t - p.contact) / p.fps
            ball[t] = ball_rest + [vb * np.cos(np.radians(p.launch_deg)) * dt,
                                   -(vb * np.sin(np.radians(p.launch_deg)) * dt - 0.5 * g * dt * dt)]

    # Handheld pan: image = world shifted by -pan*t; transform maps image -> world (ref frame 0).
    tr = Tracks.empty(p.T, p.fps, 1280, 720)
    for t in range(p.T):
        shift = p.camera_pan_px_per_frame * t
        tr.kp[t] = kp[t] - [shift, 0]
        tr.ball[t] = ball[t] - [shift, 0]
        tr.transforms[t] = np.array([[1.0, 0, shift], [0, 1.0, 0]])
    tr.kp_score[:] = 0.9
    tr.ball_r[:] = r_ball
    tr.ball_conf[:] = 0.8
    xs, ys = tr.kp[..., 0], tr.kp[..., 1]
    tr.person_box = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], axis=1)

    truth = dict(contact=p.contact, plant=p.plant, backswing=p.backswing, leg_len=leg_len,
                 ball_speed_kmh=p.ball_speed_mps * 3.6, px_per_m=px_per_m)
    return tr, truth
