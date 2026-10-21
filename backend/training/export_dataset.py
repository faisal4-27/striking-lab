"""Turn your labelled shots into training data.

    python -m training.export_dataset scorer   -> data/datasets/scorer.parquet
    python -m training.export_dataset ball     -> data/datasets/ball/ (YOLO format, pseudo-labelled)

Scorer rows: one per rated shot = feature vector + rule-engine scores + your rating.
Rule scores are recomputed from features with the *current* rules.yaml, so
editing the rules never leaves stale inputs in the dataset.

Ball export: frames from your clips with the current detector's boxes as
pseudo-labels, plus "hard" frames around contact where it found nothing.
Review/fix them in Label Studio or CVAT (YOLO format import) before training;
wrong pseudo-labels teach the detector its own mistakes.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sqlmodel import select

from app.db import DATA_DIR, Label, Shot, init_db, session
from pipeline.features import FEATURE_NAMES
from scoring import rules
from scoring.model import model_inputs, MODEL_INPUTS

DATASETS = DATA_DIR / "datasets"


def labelled_shots() -> list[tuple[Shot, Label, dict]]:
    init_db()
    out = []
    with session() as s:
        for label in s.exec(select(Label)).all():
            shot = s.get(Shot, label.shot_id)
            if shot is None:
                continue
            path = shot.dir / "analysis.json"
            if path.exists():
                out.append((shot, label, json.loads(path.read_text())))
    return out


def scorer_frame() -> pd.DataFrame:
    rows = []
    for shot, label, analysis in labelled_shots():
        feats = analysis["features"]
        rr = rules.score(feats)
        row = dict(zip(MODEL_INPUTS, model_inputs(feats, rr)))
        row.update(shot_id=shot.id, player=shot.player or "unknown", y=float(label.overall),
                   rules_overall=rr["overall"], confidence=analysis["quality"]["overall"],
                   **{f"aspect_{k}": v for k, v in json.loads(label.aspects_json).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def export_scorer() -> Path:
    df = scorer_frame()
    DATASETS.mkdir(parents=True, exist_ok=True)
    out = DATASETS / "scorer.parquet"
    df.to_parquet(out)
    print(f"{len(df)} labelled shots -> {out}")
    return out


def export_ball(every: int = 3, val_frac: float = 0.15, seed: int = 0) -> Path:
    import cv2
    from pipeline.tracks import Tracks

    root = DATASETS / "ball"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    init_db()
    with session() as s:
        shots = [sh for sh in s.exec(select(Shot)).all() if (sh.dir / "tracks.npz").exists()]
    n_img = n_box = 0
    for shot in shots:
        tr = Tracks.load(shot.dir / "tracks.npz")
        ev = json.loads((shot.dir / "analysis.json").read_text())["events"]
        c = ev["contact"]
        # every Nth frame, plus every frame in the critical window around contact
        frames = set(range(0, tr.T, every)) | set(range(max(0, c - 5), min(tr.T, c + int(0.4 * tr.fps))))
        split = "val" if rng.random() < val_frac else "train"   # split by shot, not frame (no leakage)
        cap = cv2.VideoCapture(str(shot.dir / "playback.mp4"))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i in frames:
                name = f"{shot.id}_{i:05d}"
                cv2.imwrite(str(root / "images" / split / f"{name}.jpg"), frame)
                lines = []
                if not np.isnan(tr.ball[i, 0]) and tr.ball_conf[i] > 0:
                    x, y = tr.ball[i]
                    r = tr.ball_r[i] if not np.isnan(tr.ball_r[i]) else 10
                    lines.append(f"0 {x / tr.width:.6f} {y / tr.height:.6f} {2 * r / tr.width:.6f} {2 * r / tr.height:.6f}")
                    n_box += 1
                (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
                n_img += 1
            i += 1
        cap.release()
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    print(f"{n_img} images, {n_box} pseudo-labelled balls -> {root}")
    print("Now review the labels (Label Studio / CVAT, YOLO format) before running training.train_ball")
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["scorer", "ball"])
    ap.add_argument("--every", type=int, default=3)
    a = ap.parse_args()
    export_scorer() if a.what == "scorer" else export_ball(every=a.every)


if __name__ == "__main__":
    main()
