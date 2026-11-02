from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import VIDEO_MAX_WIDTH


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    rotation: int
    step: int
    analysis_fps: float


def _open_capture(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass
    return cap


def _rotate(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def resize_max(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def probe_video(path: Path, analysis_fps: float) -> VideoInfo:
    cap = _open_capture(path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件，请确认格式受支持")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (1.0 <= fps <= 240.0):
        fps = 30.0
    orientation = 0
    try:
        orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
    except Exception:
        orientation = 0
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise ValueError("视频没有可读取的画面")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rotation = 0
    if orientation in (90, 270) and frame.shape[1] > frame.shape[0]:
        rotation = orientation
    elif orientation == 180:
        rotation = 180
    if rotation:
        frame = _rotate(frame, rotation)
    height, width = frame.shape[:2]
    cap.release()
    if frame_count <= 0:
        frame_count = int(round(fps * 60))
    duration = frame_count / fps if fps else 0.0
    step = max(1, int(round(fps / analysis_fps)))
    effective_fps = fps / step
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        rotation=rotation,
        step=step,
        analysis_fps=effective_fps,
    )


def iter_sampled_frames(info: VideoInfo, max_width: int = VIDEO_MAX_WIDTH, max_seconds: float = 120.0) -> Iterator[tuple[int, float, np.ndarray]]:
    cap = _open_capture(info.path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    max_frames = int(max_seconds * info.fps) if max_seconds > 0 else info.frame_count
    idx = 0
    try:
        while idx < max_frames:
            if not cap.grab():
                break
            if idx % info.step == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                frame = _rotate(frame, info.rotation)
                frame = resize_max(frame, max_width)
                yield idx, idx / info.fps, frame
            idx += 1
    finally:
        cap.release()


def read_frames_by_index(path: Path, indices: list[int], rotation: int = 0, max_width: int = VIDEO_MAX_WIDTH) -> dict[int, np.ndarray]:
    cap = _open_capture(path)
    if not cap.isOpened():
        return {}
    frames: dict[int, np.ndarray] = {}
    try:
        for idx in sorted(set(int(i) for i in indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = _rotate(frame, rotation)
            frames[idx] = resize_max(frame, max_width)
    finally:
        cap.release()
    return frames


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes}:{rest:04.1f}"
