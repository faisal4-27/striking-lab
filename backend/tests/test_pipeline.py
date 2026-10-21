import numpy as np
import pytest

from pipeline import calibration, events, features
from pipeline.analyze import score, to_jsonable
from pipeline.ball import BallDet, build_tracklets, stitch
from pipeline.geometry import angle_at, interpolate_nans
from scoring import rules
from tests.synth import KickParams, make_kick

STILL = {"mean_shift_px": 0.0, "max_shift_px": 0.0, "failures": 0}


def run(p: KickParams):
    tr, truth = make_kick(p)
    ev = events.detect(tr)
    bh = events.body_height_px(tr.kp)
    m, _ = calibration.metres_per_px(tr, ev.contact, bh)
    flight = calibration.ball_flight(tr, ev.contact, m)
    feats = features.extract(tr, ev, flight, {}, m)
    return tr, truth, ev, flight, feats


def test_geometry_helpers():
    assert angle_at(np.array([1.0, 0]), np.array([0.0, 0]), np.array([0.0, 1])) == pytest.approx(90)
    x = np.array([0.0, np.nan, np.nan, 3.0, np.nan])
    assert interpolate_nans(x)[:4].tolist() == [0, 1, 2, 3]
    assert np.isnan(interpolate_nans(x)[4])
    assert np.isnan(interpolate_nans(x, max_gap=1)[1])


def test_events_recover_ground_truth():
    _, truth, ev, _, _ = run(KickParams())
    assert ev.kicking_foot == "right"
    assert ev.contact_source == "ball"
    assert abs(ev.contact - truth["contact"]) <= 1
    assert abs(ev.plant - truth["plant"]) <= 4
    assert abs(ev.backswing - truth["backswing"]) <= 3
    assert ev.follow_through > ev.contact


def test_events_fall_back_to_foot_speed_without_ball():
    tr, truth = make_kick(KickParams())
    tr.ball[:] = np.nan
    ev = events.detect(tr)
    assert ev.contact_source == "foot"
    assert abs(ev.contact - truth["contact"]) <= 6


def test_features_match_parameters():
    _, _, _, flight, f = run(KickParams(trunk_lean_deg=10, ankle_deg=160, backswing_knee_flex=110,
                                        plant_knee_flex=25, arm_abduction_deg=70))
    assert f["trunk_lean_deg"] == pytest.approx(10, abs=1.5)
    assert f["ankle_lock_deg"] == pytest.approx(160, abs=3)
    assert f["backswing_knee_flex_deg"] == pytest.approx(110, abs=6)
    assert f["plant_knee_flex_deg"] == pytest.approx(25, abs=4)
    assert f["arm_balance_deg"] == pytest.approx(70, abs=3)
    assert f["backswing_thigh_deg"] == pytest.approx(30, abs=4)
    assert f["ball_speed_kmh"] == pytest.approx(90, rel=0.12)
    assert f["launch_angle_deg"] == pytest.approx(8, abs=3)
    assert f["momentum_forward"] > 0.2


def test_camera_pan_is_compensated():
    _, _, ev0, _, f0 = run(KickParams())
    _, _, ev1, _, f1 = run(KickParams(camera_pan_px_per_frame=6.0))
    assert ev0.contact == ev1.contact
    for k in ("ball_speed_kmh", "foot_speed_mps", "plant_fore_aft", "momentum_forward"):
        assert f1[k] == pytest.approx(f0[k], rel=0.05, abs=0.02), k


def test_left_footed_kick_mirrors():
    tr, truth = make_kick(KickParams())
    from pipeline.skeleton import FLIP
    tr.kp = tr.kp[:, FLIP]
    ev = events.detect(tr)
    assert ev.kicking_foot == "left"


def test_rule_scores_good_vs_bad():
    _, _, _, _, good = run(KickParams())
    _, _, _, _, bad = run(KickParams(trunk_lean_deg=-15, ankle_deg=115, ball_speed_mps=12, launch_deg=30))
    g, b = rules.score(good), rules.score(bad)
    assert g["overall"] > 75
    assert b["overall"] < g["overall"] - 20
    bad_status = {s["feature"]: s["status"] for s in b["subscores"]}
    assert bad_status["trunk_lean_deg"] == "low"
    assert bad_status["ankle_lock_deg"] == "low"
    assert bad_status["launch_angle_deg"] == "high"


def test_rule_score_shape():
    r = rules.load_rules()[0][0]
    lo, hi = r.ideal
    assert r.score((lo + hi) / 2) == (100.0, None)
    s, side = r.score(lo - r.tolerance)
    assert side == "low" and s == pytest.approx(50)


def test_full_reasoning_report_is_json_safe():
    import json
    tr, _ = make_kick(KickParams(trunk_lean_deg=-12))
    out = to_jsonable(score(tr, STILL, target=None))
    json.dumps(out, allow_nan=False)
    rep = out["report"]
    assert 0 <= rep["overall"] <= 100
    assert rep["top_fixes"][0]["feature"] in {"trunk_lean_deg", "knee_over_ball", "launch_angle_deg"}
    assert rep["source"] == "rules"


def test_target_on_and_off():
    tr, _ = make_kick(KickParams(launch_deg=5))
    ev = events.detect(tr)
    x0, y0 = tr.ball[ev.contact]
    W, H = tr.width, tr.height
    hit = {"frame": 0, "points": [[(x0 + 300) / W, (y0 - 150) / H], [(x0 + 340) / W, (y0 - 150) / H],
                                  [(x0 + 340) / W, (y0 + 20) / H], [(x0 + 300) / W, (y0 + 20) / H]]}
    flight = calibration.ball_flight(tr, ev.contact, 1 / 228.6)
    assert calibration.target_outcome(tr, ev.contact, hit, flight["ball_dir_2d"])["on_target"] is True
    miss = {"frame": 0, "points": [[(x0 + 300) / W, (y0 - 600) / H], [(x0 + 340) / W, (y0 - 600) / H],
                                   [(x0 + 340) / W, (y0 - 500) / H], [(x0 + 300) / W, (y0 - 500) / H]]}
    out = calibration.target_outcome(tr, ev.contact, miss, flight["ball_dir_2d"])
    assert out["on_target"] is False
    assert out["direction_error_deg"] > 10


def test_ball_tracker_handles_kick_impulse_and_clutter():
    fps, T = 60, 60
    dets = []
    for t in range(T):
        x = 100.0 if t <= 30 else 100.0 + 90 * (t - 30)
        frame = [] if t in (31, 40) else [BallDet(np.array([x, 400.0]), 12, 0.6)]
        frame.append(BallDet(np.array([900.0 + np.random.default_rng(t).normal(0, 40), 100.0]), 10, 0.15))
        dets.append(frame)
    chain = stitch(build_tracklets(dets), fps, max_speed_px=40 * 230)
    frames = sorted(f for tr in chain for f in tr.frames)
    ys = {round(tr.xy[i][1]) for tr in chain for i in range(len(tr.frames))}
    assert ys == {400}
    assert frames[0] == 0 and frames[-1] >= 45
