"""Train the learned shot scorer on your ratings.

    python -m training.train_scorer [--min-samples 20]

Model: LightGBM regressor on the feature vector + rule-engine group scores,
target = your 0-100 rating. Small, shallow and strongly regularised because
the dataset is small (tens to hundreds of shots).

Evaluation: grouped K-fold by player when there are ≥3 players (so the model
is tested on people it hasn't seen), otherwise plain K-fold. We compare its
MAE against the rule engine's MAE on the same shots; scoring/model.py uses
that comparison to decide how much to trust the model.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.model_selection import GroupKFold, KFold

from scoring.model import MODELS_DIR, MODEL_INPUTS, blend_weight

from .export_dataset import scorer_frame

PARAMS = dict(objective="regression_l1", learning_rate=0.03, num_leaves=7, max_depth=3,
              min_data_in_leaf=5, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, verbose=-1, seed=0)
ROUNDS = 400


def main() -> None:
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-samples", type=int, default=20)
    args = ap.parse_args()

    df = scorer_frame()
    n = len(df)
    if n < args.min_samples:
        print(f"Only {n} rated shots; need at least {args.min_samples}. Rate more on the Label page.")
        return
    X = df[MODEL_INPUTS].to_numpy(float)
    y = df["y"].to_numpy(float)
    players = df["player"].to_numpy()

    k = min(5, n // 4)
    if len(set(players)) >= 3:
        splits = GroupKFold(n_splits=min(k, len(set(players)))).split(X, y, players)
        cv_kind = "grouped by player"
    else:
        splits = KFold(n_splits=k, shuffle=True, random_state=0).split(X)
        cv_kind = "random"

    oof = np.full(n, np.nan)
    best_iters = []
    for tr_idx, va_idx in splits:
        dtr = lgb.Dataset(X[tr_idx], y[tr_idx], feature_name=MODEL_INPUTS)
        dva = lgb.Dataset(X[va_idx], y[va_idx], reference=dtr)
        m = lgb.train(PARAMS, dtr, ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_idx] = m.predict(X[va_idx], num_iteration=m.best_iteration)
        best_iters.append(m.best_iteration or ROUNDS)

    cv_mae = float(np.mean(np.abs(oof - y)))
    rules_pred = df["rules_overall"].to_numpy(float)
    ok = ~np.isnan(rules_pred)
    rules_mae = float(np.mean(np.abs(rules_pred[ok] - y[ok])))
    baseline_mae = float(np.mean(np.abs(y - np.median(y))))

    final = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=MODEL_INPUTS), int(np.median(best_iters)))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final.save_model(str(MODELS_DIR / "scorer.txt"))
    imp = dict(zip(MODEL_INPUTS, final.feature_importance("gain").round(1).tolist()))
    meta = {"n_samples": n, "cv_mae": round(cv_mae, 2), "rules_mae": round(rules_mae, 2),
            "baseline_mae": round(baseline_mae, 2), "cv": cv_kind,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "importance": dict(sorted(imp.items(), key=lambda kv: -kv[1]))}
    (MODELS_DIR / "scorer_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Trained on {n} shots ({cv_kind} CV)")
    print(f"  MAE  model {cv_mae:.1f}  |  rules {rules_mae:.1f}  |  predict-median {baseline_mae:.1f}")
    print(f"  -> blend weight {blend_weight(meta):.2f}")
    print("  top features:", ", ".join(list(meta["importance"])[:6]))
    _notify_server()


def _notify_server() -> None:
    import urllib.request
    try:
        urllib.request.urlopen(urllib.request.Request("http://localhost:8000/api/model/reload", method="POST"), timeout=3)
        print("Server reloaded the model.")
    except Exception:
        print("(Server not running; the model loads on next start.)")


if __name__ == "__main__":
    main()
