"""Camera-motion estimation for handheld footage.

For each frame we estimate a similarity transform (rotation + uniform scale +
translation) to the previous frame from background ORB features, masking out
people and the ball so moving foreground does not pull the estimate. Chaining
these gives frame→reference transforms (reference = first frame), which turns
image positions into a stable "pitch" coordinate system for velocities.
"""
from __future__ import annotations

import cv2
import numpy as np

WORK_WIDTH = 640


def _to3(M: np.ndarray) -> np.ndarray:
    return np.vstack([M, [0, 0, 1]])


class Stabilizer:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1500, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev = None          # (keypoints, descriptors)
        self.acc = np.eye(3)      # current frame -> reference
        self.transforms: list[np.ndarray] = []
        self.motion: list[float] = []   # per-frame camera translation, px (full res)
        self.failures = 0

    def update(self, frame_bgr: np.ndarray, fg_boxes: list[np.ndarray]) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        s = WORK_WIDTH / w
        gray = cv2.cvtColor(cv2.resize(frame_bgr, (WORK_WIDTH, int(h * s))), cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for b in fg_boxes:
            if b is None or np.any(np.isnan(b)):
                continue
            x1, y1, x2, y2 = b * s
            pad = 0.15 * max(x2 - x1, y2 - y1)
            cv2.rectangle(mask, (int(x1 - pad), int(y1 - pad)), (int(x2 + pad), int(y2 + pad)), 0, -1)
        kps, des = self.orb.detectAndCompute(gray, mask)

        step = np.eye(3)  # this frame -> previous frame
        if self.prev is not None and des is not None and self.prev[1] is not None and len(kps) > 20:
            pk, pd = self.prev
            matches = self.matcher.knnMatch(des, pd, k=2)
            good = [m for m, *rest in matches if rest and m.distance < 0.75 * rest[0].distance]
            if len(good) >= 15:
                src = np.float32([kps[m.queryIdx].pt for m in good])
                dst = np.float32([pk[m.trainIdx].pt for m in good])
                M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                         ransacReprojThreshold=2.0, maxIters=2000)
                if M is not None and inliers is not None and inliers.sum() >= 12:
                    # rescale from work resolution to full resolution
                    S = np.diag([s, s, 1.0])
                    step = np.linalg.inv(S) @ _to3(M) @ S
                else:
                    self.failures += 1
            else:
                self.failures += 1
        self.prev = (kps, des)
        self.acc = self.acc @ step
        self.transforms.append(self.acc[:2].copy())
        self.motion.append(float(np.linalg.norm(step[:2, 2])))
        return self.acc[:2]

    def result(self) -> tuple[np.ndarray, dict]:
        tf = np.array(self.transforms) if self.transforms else np.zeros((0, 2, 3))
        motion = np.array(self.motion)
        stats = {
            "mean_shift_px": float(motion.mean()) if len(motion) else 0.0,
            "max_shift_px": float(motion.max()) if len(motion) else 0.0,
            "failures": int(self.failures),
        }
        return tf, stats
