from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import MAX_ANALYSIS_SECONDS, STROKE_COLORS, STROKE_LABELS, VIDEO_MAX_WIDTH
from .pose import PoseTrack, draw_pose
from .video_io import VideoInfo, iter_sampled_frames, read_frames_by_index

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
]
_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return (b, g, r)


def draw_text_cn(frame: np.ndarray, lines: list[str], origin: tuple[int, int] = (18, 18), size: int = 26, color: str = "#ffffff", bg: str | None = "#000000") -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = _font(size)
    x, y = origin
    line_height = int(size * 1.45)
    if bg:
        bbox = draw.multiline_textbbox((x, y), "\n".join(lines), font=font, spacing=int(size * 0.25))
        padding = 10
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rounded_rectangle(
            (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding),
            radius=8,
            fill=(0, 0, 0, 150),
        )
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)
    draw.multiline_text((x, y), "\n".join(lines), font=font, fill=color, spacing=int(size * 0.25))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _annotate_one(
    frame: np.ndarray,
    image_pts: np.ndarray,
    wrist_idx: int,
    stroke_hint: str | None,
    subtitle: str,
    trail: np.ndarray | None = None,
    detected: bool = True,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    if detected:
        draw_pose(out, image_pts, color=(215, 255, 63), wrist_idx=wrist_idx, thickness=2)
    if trail is not None and len(trail) > 1:
        points = [(int(row[0] * w), int(row[1] * h)) for row in trail if row[3] > 0.25]
        for a, b in zip(points[:-1], points[1:]):
            cv2.line(out, a, b, (60, 220, 255), 2, cv2.LINE_AA)
    if stroke_hint:
        color = _hex_to_bgr(STROKE_COLORS.get(stroke_hint, "#ffffff"))
        cv2.rectangle(out, (3, 3), (w - 4, h - 4), color, 5)
        cv2.putText(
            out,
            f"{stroke_hint.upper()}  CONTACT",
            (24, 54),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
    if subtitle:
        cv2.putText(out, subtitle, (24, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def annotate_video(
    info: VideoInfo,
    track: PoseTrack,
    strokes: list[dict],
    out_path: Path,
    max_seconds: float = MAX_ANALYSIS_SECONDS,
    on_progress=None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_width = min(info.width, VIDEO_MAX_WIDTH)
    out_width -= out_width % 2
    out_height = int(round(info.height * out_width / max(1, info.width)))
    out_height -= out_height % 2
    writer = None
    try:
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            info.analysis_fps,
            (out_width, out_height),
        )
        if not writer.isOpened():
            writer.release()
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"avc1"),
                info.analysis_fps,
                (out_width, out_height),
            )
        if not writer.isOpened():
            raise RuntimeError("无法创建标注视频，请检查 OpenCV 编解码器")

        wrist_idx = 16 if (strokes[0]["dominant_hand"] if strokes else "right") == "right" else 15
        total = max(1, int(min(info.duration, max_seconds) * info.fps / info.step))
        for pose_index, (native_idx, timestamp, frame) in enumerate(
            iter_sampled_frames(info, max_width=VIDEO_MAX_WIDTH, max_seconds=max_seconds)
        ):
            if pose_index >= len(track):
                break
            if frame.shape[1] != out_width or frame.shape[0] != out_height:
                frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
            pose = track.image[pose_index]
            detected = bool(track.detected[pose_index]) if pose_index < len(track.detected) else False
            trail_start = max(0, pose_index - 12)
            trail = track.image[trail_start: pose_index + 1, wrist_idx, :]
            hint = None
            subtitle = ""
            for stroke in strokes:
                if abs(stroke["contact_frame"] - pose_index) <= 1:
                    hint = stroke["type"]
                    subtitle = f"{STROKE_LABELS[stroke['type']]}  speed {stroke['peak_speed_mps']:.1f} m/s"
                    break
            annotated = _annotate_one(
                frame,
                pose,
                wrist_idx,
                hint,
                subtitle,
                trail=trail,
                detected=detected,
            )
            cv2.putText(
                annotated,
                f"{timestamp:5.1f}s",
                (out_width - 110, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)
            if on_progress and pose_index % 10 == 0:
                on_progress(min(1.0, pose_index / total))
    finally:
        if writer is not None:
            writer.release()
    return out_path


def transcode_for_web(src: Path, dst: Path) -> bool:
    """Convert OpenCV's mp4v output to browser-compatible H.264."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(dst),
            ],
            capture_output=True,
            timeout=600,
        )
        return result.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    except Exception:
        return False


def make_keyframes(
    info: VideoInfo,
    track: PoseTrack,
    strokes: list[dict],
    out_dir: Path,
    max_images: int = 6,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not strokes:
        return []

    by_type: dict[str, list[dict]] = {}
    for stroke in strokes:
        by_type.setdefault(stroke["type"], []).append(stroke)

    selected: list[dict] = []
    for stroke_type, group in by_type.items():
        group = sorted(group, key=lambda s: s["peak_speed_mps"], reverse=True)
        selected.append(group[0])
        if len(group) > 2 and len(selected) < max_images:
            selected.append(group[len(group) // 2])
    selected = selected[:max_images]

    requests: list[tuple[dict, str, int]] = []  # (stroke, phase, analysed sample index)
    for stroke in selected:
        requests.append((stroke, "backswing", stroke["backswing_frame"]))
        requests.append((stroke, "contact", stroke["contact_frame"]))
        if stroke["type"] in ("forehand", "backhand"):
            requests.append((stroke, "follow", stroke["follow_end_frame"]))

    native_indices = [sample_frame * info.step for _, _, sample_frame in requests]
    frames = read_frames_by_index(info.path, native_indices, rotation=info.rotation, max_width=VIDEO_MAX_WIDTH)

    phase_labels = {"backswing": "引拍", "contact": "触球点", "follow": "随挥"}
    results = []
    for stroke, phase, sample_frame in requests:
        native_frame = sample_frame * info.step
        frame = frames.get(native_frame)
        if frame is None:
            continue
        row = max(0, min(len(track) - 1, sample_frame))
        wrist_idx = 16 if stroke["dominant_hand"] == "right" else 15
        color = STROKE_COLORS.get(stroke["type"], "#ffffff")
        annotated = frame.copy()
        draw_pose(annotated, track.image[row], color=_hex_to_bgr(color), wrist_idx=wrist_idx, thickness=2)
        trail_start = max(0, row - 10)
        h, w = annotated.shape[:2]
        points = [
            (int(track.image[j, wrist_idx, 0] * w), int(track.image[j, wrist_idx, 1] * h))
            for j in range(trail_start, row + 1)
            if track.image[j, wrist_idx, 3] > 0.25
        ]
        for a, b in zip(points[:-1], points[1:]):
            cv2.line(annotated, a, b, _hex_to_bgr(color), 2, cv2.LINE_AA)
        label = f"{STROKE_LABELS[stroke['type']]} · {phase_labels[phase]}"
        keyframe_time = native_frame / info.fps
        annotated = draw_text_cn(
            annotated,
            [label, f"{keyframe_time:.2f}s  腕速 {stroke['peak_speed_mps']:.1f} m/s"],
            origin=(18, 16),
            size=28,
            color=color,
        )
        filename = f"stroke{stroke['id']:02d}_{phase}.jpg"
        path = out_dir / filename
        cv2.imwrite(str(path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        results.append(
            {
                "stroke_id": stroke["id"],
                "type": stroke["type"],
                "type_label": STROKE_LABELS[stroke["type"]],
                "phase": phase,
                "phase_label": phase_labels[phase],
                "frame": int(native_frame),
                "time": round(keyframe_time, 3),
                "file": filename,
                "label": label,
            }
        )
    return results
