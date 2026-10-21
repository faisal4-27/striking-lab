"""Sequence model: ST-GCN over the 2D skeleton around the kick.

Hand-made features capture what coaches already know to look for; a graph
network on the raw joint trajectories can pick up timing and coordination
patterns we never wrote down. Needs a few hundred rated shots to beat the
feature model, so it only joins the blend once ``models/seq.pt`` exists.

Input preprocessing (shared by training and inference):
  * window: contact − 0.8 s … contact + 0.5 s, resampled to 48 frames
  * stabilised coords, centred on the hips at contact, scaled by leg length
  * left-footed kicks mirrored onto a right-footed skeleton; kick direction → +x
  * channels: x, y, confidence, ball-dx, ball-dy (ball relative to each joint's frame hip)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from pipeline import skeleton as S
from pipeline.tracks import Tracks

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
WINDOW = 48
PRE_S, POST_S = 0.8, 0.5
C_IN = 5


def adjacency() -> np.ndarray:
    A = np.eye(S.NUM_KEYPOINTS)
    for a, b in S.EDGES:
        A[a, b] = A[b, a] = 1
    d = A.sum(1)
    return A / np.sqrt(np.outer(d, d))


def preprocess(tr: Tracks, events: dict) -> np.ndarray:
    """-> (C_IN, WINDOW, 23) float32."""
    kp = tr.kp_stab()
    ball = tr.ball_stab()
    c = events["contact"]
    t = np.linspace(c - PRE_S * tr.fps, c + POST_S * tr.fps, WINDOW)
    t = np.clip(t, 0, tr.T - 1)
    idx = np.arange(tr.T)

    def resample(a: np.ndarray) -> np.ndarray:
        flat = a.reshape(tr.T, -1)
        out = np.empty((WINDOW, flat.shape[1]))
        for j in range(flat.shape[1]):
            col = flat[:, j]
            good = ~np.isnan(col)
            out[:, j] = np.interp(t, idx[good], col[good]) if good.sum() >= 2 else 0.0
        return out.reshape((WINDOW,) + a.shape[1:])

    k = resample(kp)                       # (W, 23, 2)
    sc = resample(tr.kp_score[..., None])[..., 0]
    b = resample(ball)                     # (W, 2)
    if events["kicking_foot"] == "left":
        k, sc = k[:, S.FLIP], sc[:, S.FLIP]
    hip_c = k[WINDOW * 8 // 13, [S.L_HIP, S.R_HIP]].mean(0)   # ≈ contact frame
    leg = np.nanmedian(np.linalg.norm(k[:, S.R_HIP] - k[:, S.R_ANKLE], axis=-1)) or 1.0
    k = (k - hip_c) / leg
    b = (b - hip_c) / leg
    # kick direction → +x
    hips = k[:, [S.L_HIP, S.R_HIP]].mean(1)
    if hips[-1, 0] - hips[0, 0] < 0:
        k[..., 0] *= -1
        b[..., 0] *= -1
    hip_t = k[:, [S.L_HIP, S.R_HIP]].mean(1, keepdims=True)
    rel_ball = np.repeat((b[:, None, :] - hip_t), S.NUM_KEYPOINTS, axis=1)
    x = np.concatenate([k, sc[..., None], rel_ball], axis=-1)   # (W, 23, 5)
    return np.nan_to_num(x.transpose(2, 0, 1)).astype(np.float32)


def build_model(hidden: int = 64):
    import torch
    from torch import nn

    A = torch.tensor(adjacency(), dtype=torch.float32)

    class STBlock(nn.Module):
        def __init__(self, cin: int, cout: int, stride: int = 1):
            super().__init__()
            self.register_buffer("A", A)
            self.edge = nn.Parameter(torch.ones_like(A))      # learnable edge importance
            self.gcn = nn.Conv2d(cin, cout, 1)
            self.tcn = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(),
                                     nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)),
                                     nn.BatchNorm2d(cout), nn.Dropout(0.2))
            self.res = (nn.Identity() if cin == cout and stride == 1
                        else nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout)))
            self.act = nn.ReLU()

        def forward(self, x):                 # x: (N, C, T, V)
            y = torch.einsum("nctv,vw->nctw", self.gcn(x), self.A * self.edge)
            return self.act(self.tcn(y) + self.res(x))

    class STGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(C_IN * S.NUM_KEYPOINTS)
            self.blocks = nn.Sequential(STBlock(C_IN, hidden), STBlock(hidden, hidden),
                                        STBlock(hidden, 2 * hidden, 2), STBlock(2 * hidden, 2 * hidden))
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, 1))

        def forward(self, x):                 # (N, C, T, V)
            n, c, t, v = x.shape
            x = self.bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t)).reshape(n, c, v, t).permute(0, 1, 3, 2)
            x = self.blocks(x).mean(dim=(2, 3))
            return self.head(x).squeeze(-1) * 25 + 50      # outputs on the 0..100 scale

    return STGCN()


@lru_cache
def _load():
    path, meta_path = MODELS_DIR / "seq.pt", MODELS_DIR / "seq_meta.json"
    if not path.exists() or not meta_path.exists():
        return None, None
    import torch
    model = build_model(json.loads(meta_path.read_text()).get("hidden", 64))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model, json.loads(meta_path.read_text())


def reload() -> None:
    _load.cache_clear()


def predict(tr: Tracks, events: dict) -> dict | None:
    model, meta = _load()
    if model is None:
        return None
    import torch
    with torch.no_grad():
        x = torch.from_numpy(preprocess(tr, events))[None]
        y = float(model(x)[0].clamp(0, 100))
    return {"score": round(y, 1), "cv_mae": meta.get("cv_mae"), "rules_mae": meta.get("rules_mae"),
            "n_samples": meta.get("n_samples")}
