"""Train the ST-GCN sequence scorer (phase 3: worth it at ~300+ rated shots).

    python -m training.train_seq [--epochs 150] [--folds 5]

Cross-validates first (reports MAE next to the rule engine's on the same
folds), then trains on everything and saves models/seq.pt + seq_meta.json.
Augmentation: random scale, small rotation, joint jitter, time shift — all
things that vary between filming sessions but don't change shot quality.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from pipeline.tracks import Tracks
from scoring.seq import MODELS_DIR, build_model, preprocess

from .export_dataset import labelled_shots
from scoring import rules


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, r = [], [], []
    for shot, label, analysis in labelled_shots():
        p = shot.dir / "tracks.npz"
        if not p.exists():
            continue
        X.append(preprocess(Tracks.load(p), analysis["events"]))
        y.append(label.overall)
        r.append(rules.score(analysis["features"])["overall"] or np.nan)
    return np.stack(X), np.array(y, np.float32), np.array(r, float)


def augment(x, rng):
    import torch
    n = x.shape[0]
    s = torch.from_numpy(rng.uniform(0.9, 1.1, (n, 1, 1, 1)).astype(np.float32))
    th = torch.from_numpy(rng.uniform(-0.12, 0.12, n).astype(np.float32))
    cos, sin = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
    x = x.clone()
    for a, b in ((0, 1), (3, 4)):            # rotate joint xy and ball xy channels
        xa, xb = x[:, a].clone(), x[:, b].clone()
        x[:, a] = (cos * xa - sin * xb) * s[:, 0]
        x[:, b] = (sin * xa + cos * xb) * s[:, 0]
    x[:, :2] += 0.02 * torch.randn_like(x[:, :2])
    shift = int(rng.integers(-2, 3))
    return torch.roll(x, shift, dims=2)


def fit(Xtr, ytr, epochs, seed, device):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs)
    X, Y = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 16):
            b = perm[i:i + 16]
            if len(b) < 2:
                continue
            xb = augment(X[b], rng).to(device)
            loss = torch.nn.functional.smooth_l1_loss(model(xb), Y[b].to(device), beta=5.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


def main() -> None:
    import torch
    from sklearn.model_selection import KFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-samples", type=int, default=60)
    args = ap.parse_args()

    X, y, r = load()
    if len(y) < args.min_samples:
        print(f"Only {len(y)} rated shots with tracks; the sequence model needs {args.min_samples}+ "
              f"(ideally 300+). Use train_scorer until then.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oof = np.zeros_like(y)
    for k, (tr_i, va_i) in enumerate(KFold(args.folds, shuffle=True, random_state=0).split(X)):
        m = fit(X[tr_i], y[tr_i], args.epochs, k, device)
        with torch.no_grad():
            oof[va_i] = m(torch.from_numpy(X[va_i]).to(device)).cpu().numpy()
        print(f"fold {k + 1}: MAE {np.mean(np.abs(oof[va_i] - y[va_i])):.1f}")
    cv_mae = float(np.mean(np.abs(oof - y)))
    ok = ~np.isnan(r)
    rules_mae = float(np.mean(np.abs(r[ok] - y[ok])))
    print(f"CV MAE seq {cv_mae:.1f} vs rules {rules_mae:.1f}")

    model = fit(X, y, args.epochs, 123, device).cpu()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "seq.pt")
    (MODELS_DIR / "seq_meta.json").write_text(json.dumps({
        "n_samples": int(len(y)), "cv_mae": round(cv_mae, 2), "rules_mae": round(rules_mae, 2), "hidden": 64,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))
    print(f"Saved {MODELS_DIR / 'seq.pt'}")


if __name__ == "__main__":
    main()
