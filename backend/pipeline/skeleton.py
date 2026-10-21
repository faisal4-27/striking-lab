"""Keypoint layout used throughout the pipeline.

We keep the 17 COCO body keypoints plus the 6 COCO-WholeBody foot keypoints
(big toe / small toe / heel per foot) = 23 points. RTMW wholebody outputs 133
points whose first 23 are exactly these, so conversion is a slice.
"""

NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
L_BIG_TOE, L_SMALL_TOE, L_HEEL = 17, 18, 19
R_BIG_TOE, R_SMALL_TOE, R_HEEL = 20, 21, 22

NUM_KEYPOINTS = 23

NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
    "l_big_toe", "l_small_toe", "l_heel", "r_big_toe", "r_small_toe", "r_heel",
]

# Edges for drawing (frontend mirrors this list).
EDGES = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (L_ANKLE, L_HEEL), (L_ANKLE, L_BIG_TOE), (L_BIG_TOE, L_SMALL_TOE),
    (R_ANKLE, R_HEEL), (R_ANKLE, R_BIG_TOE), (R_BIG_TOE, R_SMALL_TOE),
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),
]

# Left/right swap permutation, used to mirror left-footed kickers onto a
# canonical right-footed skeleton for the sequence model.
FLIP = [NOSE, R_EYE, L_EYE, R_EAR, L_EAR, R_SHOULDER, L_SHOULDER, R_ELBOW, L_ELBOW,
        R_WRIST, L_WRIST, R_HIP, L_HIP, R_KNEE, L_KNEE, R_ANKLE, L_ANKLE,
        R_BIG_TOE, R_SMALL_TOE, R_HEEL, L_BIG_TOE, L_SMALL_TOE, L_HEEL]


def side(foot: str) -> dict[str, int]:
    """Joint indices for one side of the body ('left' or 'right')."""
    if foot == "left":
        return dict(shoulder=L_SHOULDER, elbow=L_ELBOW, wrist=L_WRIST, hip=L_HIP, knee=L_KNEE,
                    ankle=L_ANKLE, toe=L_BIG_TOE, heel=L_HEEL)
    return dict(shoulder=R_SHOULDER, elbow=R_ELBOW, wrist=R_WRIST, hip=R_HIP, knee=R_KNEE,
                ankle=R_ANKLE, toe=R_BIG_TOE, heel=R_HEEL)


def other(foot: str) -> str:
    return "right" if foot == "left" else "left"
