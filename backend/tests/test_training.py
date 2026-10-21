import sys

import numpy as np
import pandas as pd
import pytest

from pipeline import calibration, events, features
from scoring import model as learned
from scoring import rules
from scoring.feedback import build_report
from scoring.model import MODEL_INPUTS, model_inputs
from tests.synth import KickParams, make_kick


def synthetic_rated_shots(n: int, seed: int = 0) -> pd.DataFrame:
    """Random kicks rated by a hidden 'coach' who cares a lot about trunk lean."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = KickParams(trunk_lean_deg=rng.uniform(-20, 25), ankle_deg=rng.uniform(110, 170),
                       backswing_knee_flex=rng.uniform(60, 130), ball_speed_mps=rng.uniform(10, 30),
                       launch_deg=rng.uniform(0, 30))
        tr, _ = make_kick(p)
        ev = events.detect(tr)
        m, _ = calibration.metres_per_px(tr, ev.contact, events.body_height_px(tr.kp))
        fl = calibration.ball_flight(tr, ev.contact, m)
        f = features.extract(tr, ev, fl, {}, m)
        rr = rules.score(f)
        y = 85 - 2.0 * abs(p.trunk_lean_deg - 8) + 0.3 * (p.ankle_deg - 140) + rng.normal(0, 3)
        rows.append({**dict(zip(MODEL_INPUTS, model_inputs(f, rr))), "player": f"p{i % 4}",
                     "y": float(np.clip(y, 0, 100)), "rules_overall": rr["overall"], "shot_id": str(i)})
    return pd.DataFrame(rows)


def test_scorer_training_end_to_end(tmp_path, monkeypatch):
    import training.train_scorer as ts
    df = synthetic_rated_shots(60)
    monkeypatch.setattr(ts, "scorer_frame", lambda: df)
    monkeypatch.setattr(ts, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(ts, "_notify_server", lambda: None)
    monkeypatch.setattr(learned, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_scorer"])
    learned.reload()
    ts.main()
    assert (tmp_path / "scorer.txt").exists()

    st = learned.status()
    assert st["trained"] and st["n_samples"] == 60
    # The hidden coach is learnable from the features: model should beat the generic rules.
    assert st["cv_mae"] < st["rules_mae"]
    assert st["weight"] > 0

    tr, _ = make_kick(KickParams(trunk_lean_deg=-15))
    ev = events.detect(tr)
    m, _ = calibration.metres_per_px(tr, ev.contact, events.body_height_px(tr.kp))
    f = features.extract(tr, ev, calibration.ball_flight(tr, ev.contact, m), {}, m)
    rep = build_report(f, {"overall": 0.9})
    assert rep["source"] == "blend"
    assert rep["model"]["shap"]                     # explanations are present
    learned.reload()


def test_scorer_refuses_tiny_dataset(tmp_path, monkeypatch, capsys):
    import training.train_scorer as ts
    monkeypatch.setattr(ts, "scorer_frame", lambda: synthetic_rated_shots(5))
    monkeypatch.setattr(ts, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_scorer"])
    ts.main()
    assert not (tmp_path / "scorer.txt").exists()
    assert "need at least" in capsys.readouterr().out


def test_sequence_model_shapes_and_fit():
    torch = pytest.importorskip("torch")
    from scoring.seq import C_IN, WINDOW, build_model, preprocess
    from training.train_seq import fit

    xs, ys = [], []
    for lean in (-15, -5, 5, 15, 25, 0, 10, 20):
        tr, _ = make_kick(KickParams(trunk_lean_deg=lean))
        ev = events.detect(tr).to_dict()
        xs.append(preprocess(tr, ev))
        ys.append(80 - abs(lean - 8) * 2)
    X = np.stack(xs)
    assert X.shape == (8, C_IN, WINDOW, 23) and np.isfinite(X).all()

    # mirrored left-footed kick lands on the same canonical skeleton
    tr, _ = make_kick(KickParams())
    from pipeline.skeleton import FLIP
    tr2, _ = make_kick(KickParams())
    tr2.kp = tr2.kp[:, FLIP]
    a = preprocess(tr, events.detect(tr).to_dict())
    b = preprocess(tr2, events.detect(tr2).to_dict())
    assert np.allclose(a[:2], b[:2], atol=1e-4)

    model = fit(X, np.array(ys, np.float32), epochs=3, seed=0, device="cpu")
    with torch.no_grad():
        out = model(torch.from_numpy(X))
    assert out.shape == (8,) and torch.isfinite(out).all()
    assert build_model() is not None
