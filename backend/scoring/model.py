"""Learned scorer: LightGBM on the feature vector, trained on *your* ratings.

``training/train_scorer.py`` writes models/scorer.txt + scorer_meta.json. At
inference we blend:  score = w · model + (1 − w) · rules, where w grows with
the amount of labelled data and with how much the model beats the rule
engine in cross-validation. Per-feature SHAP contributions come straight from
LightGBM (``pred_contrib=True``) and feed the feedback engine.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from pipeline.features import FEATURE_NAMES

log = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Inputs to the learned model: raw features + the rule engine's own verdicts.
MODEL_INPUTS = FEATURE_NAMES + ["rules_technique", "rules_outcome"]


def model_inputs(features: dict, rules_result: dict) -> list[float]:
    row = [np.nan if features.get(k) is None else float(features[k]) for k in FEATURE_NAMES]
    g = rules_result.get("groups", {})
    row += [g.get("technique", np.nan), g.get("outcome", np.nan)]
    return row


def blend_weight(meta: dict) -> float:
    n = meta.get("n_samples", 0)
    cv, base = meta.get("cv_mae"), meta.get("rules_mae")
    if not n or cv is None or base is None or base <= 0:
        return 0.0
    gain = float(np.clip(2 * (1 - cv / base), 0, 1))
    return round(min(0.85, n / 250) * gain, 3)


@lru_cache
def _load():
    path, meta_path = MODELS_DIR / "scorer.txt", MODELS_DIR / "scorer_meta.json"
    if not path.exists() or not meta_path.exists():
        return None, None
    import lightgbm as lgb
    return lgb.Booster(model_file=str(path)), json.loads(meta_path.read_text())


def reload() -> None:
    _load.cache_clear()


def status() -> dict:
    booster, meta = _load()
    if booster is None:
        return {"trained": False, "weight": 0.0}
    return {"trained": True, "weight": blend_weight(meta), **meta}


def predict(features: dict, rules_result: dict) -> dict | None:
    booster, meta = _load()
    if booster is None:
        return None
    x = np.array([model_inputs(features, rules_result)], dtype=float)
    pred = float(np.clip(booster.predict(x)[0], 0, 100))
    contrib = booster.predict(x, pred_contrib=True)[0]   # last column = bias
    shap = {name: round(float(v), 2) for name, v in zip(MODEL_INPUTS, contrib[:-1]) if abs(v) > 0.05}
    return {"score": round(pred, 1), "weight": blend_weight(meta), "shap": shap,
            "n_samples": meta.get("n_samples"), "cv_mae": meta.get("cv_mae")}
