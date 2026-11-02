from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover - environment dependent
    mp = None
    _MP_IMPORT_ERROR = exc
else:
    _MP_IMPORT_ERROR = None

SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
]


@dataclass
class PoseTrack:
    fps: float
    image: np.ndarray  # (N, 33, 4) normalized image landmarks
    world: np.ndarray  # (N, 33, 3) smoothed world landmarks (meters)
    up: float
    detected: np.ndarray  # (N,) bool

    def __len__(self) -> int:
        return int(self.world.shape[0])

    @property
    def duration(self) -> float:
        return len(self) / self.fps if self.fps else 0.0


class PoseExtractor:
    """Thin wrapper around MediaPipe Pose (legacy solutions API)."""

    def __init__(self, model_complexity: int = 1) -> None:
        if mp is None:
            raise RuntimeError(
                "mediapipe 未安装或无法导入，请运行 pip install -r requirements.txt"
            ) from _MP_IMPORT_ERROR
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr: np.ndarray) -> dict | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None
        image = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark],
            dtype=np.float32,
        )
        world = None
        if result.pose_world_landmarks:
            world = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks.landmark],
                dtype=np.float32,
            )
        return {"image": image, "world": world}

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def mid(pts: np.ndarray, i: int, j: int) -> np.ndarray:
    return (pts[i] + pts[j]) / 2.0


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = norm(v)
    if n < 1e-6:
        return fallback if fallback is not None else np.zeros_like(v)
    return v / n


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees."""
    ba = a - b
    bc = c - b
    denom = norm(ba) * norm(bc)
    if denom < 1e-8:
        return float("nan")
    cosv = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosv))


def up_sign(world: np.ndarray) -> float:
    """Return +1 when world y grows upward, -1 when it grows downward."""
    shoulder = mid(world, 11, 12)
    hip = mid(world, 23, 24)
    return 1.0 if shoulder[1] > hip[1] else -1.0


def height_above(pts: np.ndarray, idx: int, ref: np.ndarray, up: float) -> float:
    return float(up * (pts[idx][1] - ref[1]))


def horizontal(v: np.ndarray) -> np.ndarray:
    out = np.array(v, dtype=np.float64)
    out[1] = 0.0
    return out


def body_frame(world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (lateral, forward, up_axis, up) in the player's body frame.

    lateral points from the player's left hip to right hip, forward points in
    the direction the pelvis/nose faces, up points toward the head.
    """
    up = up_sign(world)
    up_axis = np.array([0.0, up, 0.0])
    hip_l, hip_r = world[23], world[24]
    lateral = unit(horizontal(hip_r - hip_l))
    if norm(lateral) < 1e-6:
        lateral = unit(horizontal(world[12] - world[11]))
    hip_mid = (hip_l + hip_r) / 2.0
    nose = world[0]
    to_nose = horizontal(nose - hip_mid)
    to_nose = to_nose - lateral * np.dot(to_nose, lateral)
    forward = unit(to_nose)
    if norm(forward) < 1e-6:
        forward = unit(np.cross(up_axis, lateral))
    return lateral, forward, up_axis, up


def fill_and_smooth(arr: np.ndarray, fps: float, window_sec: float = 0.12) -> np.ndarray:
    """Linear-interpolate missing rows/cols and Savitzky-Golay smooth along time."""
    from scipy.signal import savgol_filter

    out = np.asarray(arr, dtype=np.float64).copy()
    n = out.shape[0]
    if n == 0:
        return out
    flat = out.reshape(n, -1)
    idx = np.arange(n)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        mask = np.isfinite(col)
        if not mask.any():
            col[:] = 0.0
            continue
        if not mask.all():
            col[~mask] = np.interp(idx[~mask], idx[mask], col[mask])
    window = int(round(fps * window_sec))
    if window % 2 == 0:
        window += 1
    window = max(5, window)
    if n >= window:
        flat = savgol_filter(flat, window_length=window, polyorder=min(3, window - 1), axis=0)
    return flat.reshape(out.shape)


def draw_pose(
    frame: np.ndarray,
    image_pts: np.ndarray,
    color: tuple[int, int, int] = (215, 255, 63),
    wrist_idx: int | None = None,
    thickness: int = 2,
) -> None:
    h, w = frame.shape[:2]
    for a, b in SKELETON_CONNECTIONS:
        if image_pts[a, 3] < 0.25 or image_pts[b, 3] < 0.25:
            continue
        pa = (int(image_pts[a, 0] * w), int(image_pts[a, 1] * h))
        pb = (int(image_pts[b, 0] * w), int(image_pts[b, 1] * h))
        cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)
    for i in range(11, 33):
        if image_pts[i, 3] < 0.25:
            continue
        p = (int(image_pts[i, 0] * w), int(image_pts[i, 1] * h))
        radius = 5 if i == wrist_idx else 3
        point_color = (60, 220, 255) if i == wrist_idx else color
        cv2.circle(frame, p, radius, point_color, -1, cv2.LINE_AA)
