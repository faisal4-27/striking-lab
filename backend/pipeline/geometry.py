"""Small vector / signal helpers shared by events and features."""
import numpy as np


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees (works on (..., D) arrays, NaN-propagating)."""
    ba = a - b
    bc = c - b
    cos = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Unsigned angle between vectors in degrees."""
    cos = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def interpolate_nans(x: np.ndarray, max_gap: int | None = None) -> np.ndarray:
    """Linearly fill NaN runs along axis 0 (each column independently).

    Runs longer than ``max_gap`` are left as NaN. Leading/trailing NaNs stay NaN.
    """
    x = np.array(x, dtype=float, copy=True)
    flat = x.reshape(len(x), -1)
    idx = np.arange(len(x))
    for j in range(flat.shape[1]):
        col = flat[:, j]
        good = ~np.isnan(col)
        if good.sum() < 2:
            continue
        filled = np.interp(idx, idx[good], col[good])
        # keep leading/trailing NaNs
        first, last = idx[good][0], idx[good][-1]
        filled[:first] = np.nan
        filled[last + 1:] = np.nan
        if max_gap is not None:
            bad = ~good
            run_start = None
            for i in range(len(col) + 1):
                if i < len(col) and bad[i]:
                    if run_start is None:
                        run_start = i
                elif run_start is not None:
                    if i - run_start > max_gap:
                        filled[run_start:i] = np.nan
                    run_start = None
        flat[:, j] = filled
    return flat.reshape(x.shape)


def speed(points: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame speed (units/s) of a (T, D) trajectory using central differences."""
    if len(points) < 2:
        return np.full(len(points), np.nan)
    v = np.gradient(points, axis=0) * fps
    return np.linalg.norm(v, axis=-1)


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    pad = k // 2
    xp = np.pad(x, (pad, k - 1 - pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def nan_argmax(x: np.ndarray, default: int = 0) -> int:
    return default if np.all(np.isnan(x)) else int(np.nanargmax(x))


def nan_argmin(x: np.ndarray, default: int = 0) -> int:
    return default if np.all(np.isnan(x)) else int(np.nanargmin(x))


def apply_transform(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine or 3x3 homography to (..., 2) points."""
    pts = np.asarray(points, dtype=float)
    shp = pts.shape
    p = pts.reshape(-1, 2)
    if M.shape == (2, 3):
        out = p @ M[:, :2].T + M[:, 2]
    else:
        h = np.c_[p, np.ones(len(p))] @ M.T
        out = h[:, :2] / h[:, 2:3]
    return out.reshape(shp)
