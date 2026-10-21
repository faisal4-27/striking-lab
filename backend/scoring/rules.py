"""Rule-based biomechanics scoring (the 'expert prior')."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

RULES_PATH = Path(__file__).with_name("rules.yaml")


@dataclass
class Rule:
    feature: str
    label: str
    group: str
    aspect: str
    weight: float
    ideal: tuple[float, float]
    tolerance: float
    unit: str
    low: dict | None
    high: dict | None

    def score(self, x: float) -> tuple[float, str | None]:
        """Returns (0..100 subscore, 'low' | 'high' | None)."""
        lo, hi = self.ideal
        if x < lo:
            d, side = lo - x, "low"
        elif x > hi:
            d, side = x - hi, "high"
        else:
            return 100.0, None
        return 100.0 * 2.0 ** (-(d / self.tolerance) ** 2), side


@lru_cache
def load_rules(path: Path = RULES_PATH) -> tuple[list[Rule], dict]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    rules = [Rule(feature=r["feature"], label=r["label"], group=r["group"], aspect=r["aspect"],
                  weight=float(r["weight"]), ideal=tuple(r["ideal"]), tolerance=float(r["tolerance"]),
                  unit=r.get("unit", ""), low=r.get("low"), high=r.get("high"))
             for r in cfg["rules"]]
    return rules, cfg["group_weights"]


def score(features: dict) -> dict:
    rules, group_w = load_rules()
    subs = []
    for r in rules:
        x = features.get(r.feature)
        if x is None:
            continue
        s, side = r.score(float(x))
        msg = (r.low if side == "low" else r.high if side == "high" else None) or {}
        subs.append({
            "feature": r.feature, "label": r.label, "group": r.group, "aspect": r.aspect,
            "value": x, "unit": r.unit, "ideal": list(r.ideal), "score": round(s, 1),
            "weight": r.weight, "status": side or "good",
            "issue": msg.get("issue"), "tip": msg.get("tip"),
        })

    groups = {}
    for g in ("technique", "outcome"):
        items = [s for s in subs if s["group"] == g]
        if items:
            w = sum(s["weight"] for s in items)
            groups[g] = round(sum(s["score"] * s["weight"] for s in items) / w, 1)

    if groups:
        gw = {g: group_w[g] for g in groups}
        overall = sum(groups[g] * gw[g] for g in groups) / sum(gw.values())
    else:
        overall = None

    aspects: dict[str, list] = {}
    for s in subs:
        aspects.setdefault(s["aspect"], []).append(s)
    aspect_scores = {a: round(sum(i["score"] * i["weight"] for i in v) / sum(i["weight"] for i in v), 1)
                     for a, v in aspects.items()}

    return {"overall": None if overall is None else round(overall, 1), "groups": groups,
            "aspects": aspect_scores, "subscores": subs}
