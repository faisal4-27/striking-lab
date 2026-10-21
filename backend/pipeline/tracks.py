"""Container for everything the perception stage extracts from a clip."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import apply_transform
from .skeleton import NUM_KEYPOINTS


@dataclass
class Tracks:
    fps: float
    width: int
    height: int
    kp: np.ndarray            # (T, 23, 2) image px, NaN where missing
    kp_score: np.ndarray      # (T, 23) confidence 0..1
    ball: np.ndarray          # (T, 2) image px, NaN where missing
    ball_r: np.ndarray        # (T,) apparent radius px
    ball_conf: np.ndarray     # (T,)
    transforms: np.ndarray    # (T, 2, 3) affine: frame t -> reference frame coords
    person_box: np.ndarray    # (T, 4) x1,y1,x2,y2
    kp3d: np.ndarray | None = None   # (T, 23, 3) metric, hip-centred, y up; NaN outside window
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.kp)

    @classmethod
    def empty(cls, T: int, fps: float, width: int, height: int) -> "Tracks":
        eye = np.tile(np.array([[1.0, 0, 0], [0, 1.0, 0]]), (T, 1, 1))
        return cls(
            fps=fps, width=width, height=height,
            kp=np.full((T, NUM_KEYPOINTS, 2), np.nan),
            kp_score=np.zeros((T, NUM_KEYPOINTS)),
            ball=np.full((T, 2), np.nan),
            ball_r=np.full(T, np.nan),
            ball_conf=np.zeros(T),
            transforms=eye,
            person_box=np.full((T, 4), np.nan),
        )

    # Stabilised ("world-ish") coordinates: camera motion removed.
    def kp_stab(self) -> np.ndarray:
        out = np.empty_like(self.kp)
        for t in range(self.T):
            out[t] = apply_transform(self.kp[t], self.transforms[t])
        return out

    def ball_stab(self) -> np.ndarray:
        out = np.empty_like(self.ball)
        for t in range(self.T):
            out[t] = apply_transform(self.ball[t], self.transforms[t])
        return out

    def scale_at(self, t: int) -> float:
        """Isotropic scale of the frame->reference transform at frame t."""
        M = self.transforms[t]
        return float(np.sqrt(abs(np.linalg.det(M[:, :2]))))

    def save(self, path) -> None:
        arrays = dict(kp=self.kp, kp_score=self.kp_score, ball=self.ball, ball_r=self.ball_r,
                      ball_conf=self.ball_conf, transforms=self.transforms, person_box=self.person_box,
                      fps=self.fps, width=self.width, height=self.height)
        if self.kp3d is not None:
            arrays["kp3d"] = self.kp3d
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "Tracks":
        d = np.load(path)
        return cls(fps=float(d["fps"]), width=int(d["width"]), height=int(d["height"]),
                   kp=d["kp"], kp_score=d["kp_score"], ball=d["ball"], ball_r=d["ball_r"],
                   ball_conf=d["ball_conf"], transforms=d["transforms"], person_box=d["person_box"],
                   kp3d=d["kp3d"] if "kp3d" in d else None)
