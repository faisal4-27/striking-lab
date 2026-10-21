"""Turn rule subscores + model attributions into a ranked coaching report."""
from __future__ import annotations

from . import model as learned
from . import rules as rule_engine

FILMING_TIPS = {
    "pose": "Keep the whole body in frame, including your feet, and film in good light.",
    "ball": "Keep the ball in frame before and after the kick. 60 fps footage helps a lot.",
    "camera_steadiness": "Ask your friend to hold the phone steady with both hands, or rest it on something.",
    "events": "The ball wasn't tracked at contact, so the strike moment was estimated from foot speed.",
}


def _combine(ml: dict | None, seq: dict | None) -> dict | None:
    """Merge the feature model and the sequence model, weighting each by 1/MAE."""
    if seq is None or not seq.get("cv_mae"):
        return ml
    seq_w = learned.blend_weight({**seq, "rules_mae": seq.get("rules_mae")})
    if ml is None:
        return {"score": seq["score"], "weight": seq_w, "shap": {}, "n_samples": seq["n_samples"],
                "cv_mae": seq["cv_mae"], "parts": {"sequence": seq["score"]}}
    a, b = 1 / max(ml["cv_mae"] or 1e-3, 1e-3), 1 / seq["cv_mae"]
    score = (a * ml["score"] + b * seq["score"]) / (a + b)
    return {**ml, "score": round(score, 1), "weight": max(ml["weight"], seq_w),
            "parts": {"features": ml["score"], "sequence": seq["score"]}}


def build_report(features: dict, quality: dict, seq_pred: dict | None = None) -> dict:
    rules = rule_engine.score(features)
    ml = _combine(learned.predict(features, rules), seq_pred)

    rules_overall = rules["overall"]
    if ml is not None and rules_overall is not None and ml["weight"] > 0:
        overall = ml["weight"] * ml["score"] + (1 - ml["weight"]) * rules_overall
        source = "blend"
    else:
        overall, source = rules_overall, "rules"

    # Rank issues by how many points each one costs the overall score.
    issues = []
    for s in rules["subscores"]:
        if s["status"] == "good" or not s["issue"]:
            continue
        impact = s["weight"] * (100 - s["score"])
        if ml is not None and ml["weight"] > 0:
            # the model's view of this feature (negative SHAP = it hurt the rating)
            impact += max(0.0, -ml["shap"].get(s["feature"], 0.0)) * 5 * ml["weight"]
        issues.append({**s, "impact": round(impact, 1)})
    issues.sort(key=lambda s: -s["impact"])

    strengths = sorted((s for s in rules["subscores"] if s["score"] >= 85),
                       key=lambda s: -s["weight"])[:4]

    filming = [FILMING_TIPS[k] for k in ("pose", "ball", "camera_steadiness")
               if quality.get(k, 1) < 0.5]
    if quality.get("events", 1) < 0.8:
        filming.append(FILMING_TIPS["events"])
    if quality.get("view") == "front/back":
        filming.append("This was filmed from behind or in front, so plant-foot position, foot speed and "
                       "momentum couldn't be measured. Film from the side, roughly level with the ball.")
    elif quality.get("view") == "oblique":
        filming.append("The camera angle was diagonal. A side-on view gives the most accurate measurements.")

    return {
        "overall": None if overall is None else round(overall, 1),
        "source": source,
        "rules": rules,
        "model": ml,
        "issues": issues,
        "top_fixes": issues[:3],
        "strengths": [{"label": s["label"], "score": s["score"]} for s in strengths],
        "confidence": quality.get("overall"),
        "filming_tips": filming,
    }
