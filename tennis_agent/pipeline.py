from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import coach
from .config import (
    ANALYSIS_FPS,
    MAX_ANALYSIS_SECONDS,
    OUTPUTS_DIR,
    POSE_COMPLEXITY,
    POSE_MAX_WIDTH,
)
from .metrics import (
    aggregate_by_type,
    compute_stroke_metrics,
    consistency_score,
    estimate_level,
    issue_candidates,
    score_stroke,
    strength_candidates,
)
from .pose import PoseExtractor, PoseTrack, fill_and_smooth, up_sign
from .strokes import detect_strokes
from .video_io import VideoInfo, iter_sampled_frames, probe_video
from .visualize import annotate_video, make_keyframes, transcode_for_web


class AnalysisError(RuntimeError):
    pass


def _empty_report(reason: str) -> dict:
    return {
        "headline": "未识别到明显击球动作",
        "level_assessment": "没有足够的击球样本，暂时无法评估水平。",
        "overall": (
            "视频中检测到了人体，但没有捕捉到明确的挥拍加速。常见原因："
            "人物太小、挥拍不在画面内、光线不足、机位太远，或视频里主要是走位/捡球。"
        ),
        "strengths": [],
        "issues": [],
        "training_plan": {
            "focus": "先拍一段能被稳定分析的击球视频",
            "weeks": [],
            "success_metrics": [
                "全身入镜，人物高度至少占画面一半",
                "固定机位，侧后方约 45°，连续完成 5–10 次同类击球",
                "光线充足，避免逆光和背景杂乱",
            ],
        },
        "encouragement": "先解决拍摄条件，下一次分析会更有价值。",
        "meta": {"provider": "local", "model": None, "reason": reason},
    }


def _progress(callback, percent: float, stage: str) -> None:
    if callback:
        callback(max(0.0, min(100.0, percent)), stage)


def run_analysis(
    job_id: str,
    video_path: Path,
    profile: dict,
    on_progress=None,
    output_root: Path | None = None,
) -> dict:
    output_root = output_root or OUTPUTS_DIR
    out_dir = output_root / job_id
    frames_dir = out_dir / "keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)

    _progress(on_progress, 2, "读取视频信息")
    info: VideoInfo = probe_video(video_path, ANALYSIS_FPS)
    if info.duration > MAX_ANALYSIS_SECONDS:
        truncated = True
    else:
        truncated = False

    _progress(on_progress, 5, "正在提取人体姿态")
    extractor = PoseExtractor(model_complexity=POSE_COMPLEXITY)
    image_rows: list[np.ndarray] = []
    world_rows: list[np.ndarray] = []
    detected: list[bool] = []
    total_frames = max(1, int(min(info.duration, MAX_ANALYSIS_SECONDS) * info.fps / info.step))
    try:
        for i, (_, _, frame) in enumerate(
            iter_sampled_frames(info, max_width=POSE_MAX_WIDTH, max_seconds=MAX_ANALYSIS_SECONDS)
        ):
            result = extractor.process(frame)
            if result is None:
                image_rows.append(np.full((33, 4), np.nan, dtype=np.float32))
                world_rows.append(np.full((33, 3), np.nan, dtype=np.float32))
                detected.append(False)
            else:
                image_rows.append(result["image"])
                world_rows.append(
                    result["world"] if result["world"] is not None else np.full((33, 3), np.nan, dtype=np.float32)
                )
                detected.append(True)
            if i % 5 == 0:
                _progress(on_progress, 5 + 48 * min(1.0, i / total_frames), "正在提取人体姿态")
    finally:
        extractor.close()

    if len(image_rows) < 10:
        raise AnalysisError("视频太短或无法读取足够帧，请上传 5 秒以上的清晰视频。")
    detected_arr = np.array(detected, dtype=bool)
    detection_rate = float(detected_arr.mean())
    if detection_rate < 0.35:
        raise AnalysisError(
            "未检测到足够清晰的人体姿态。请确保全身入镜、光线充足、人物占画面高度至少一半，并尽量固定机位。"
        )

    image = fill_and_smooth(np.stack(image_rows), info.analysis_fps, 0.12)
    world = fill_and_smooth(np.stack(world_rows), info.analysis_fps, 0.15)
    signs = [up_sign(world[i]) for i in range(len(world)) if detected_arr[i]]
    up = float(np.median(signs)) if signs else -1.0
    if up not in (-1.0, 1.0):
        up = 1.0 if up > 0 else -1.0
    track = PoseTrack(
        fps=info.analysis_fps,
        image=image,
        world=world,
        up=up,
        detected=detected_arr,
    )

    _progress(on_progress, 56, "正在识别击球事件")
    handedness, strokes = detect_strokes(track, profile.get("handedness", "auto"))

    per_stroke: list[dict] = []
    for stroke in strokes:
        try:
            metrics = compute_stroke_metrics(track, stroke, handedness)
        except Exception:
            continue
        scoring = score_stroke(stroke["type"], metrics)
        per_stroke.append({**stroke, "metrics": metrics, "scoring": scoring})

    _progress(on_progress, 62, "正在计算生物力学指标")
    by_type = aggregate_by_type(per_stroke)
    stroke_count = len(per_stroke)

    overall_score = None
    if by_type:
        total = sum(v["count"] for v in by_type.values())
        scores = [(v["score"], v["count"]) for v in by_type.values() if v["score"] is not None]
        if scores:
            overall_score = round(sum(s * c for s, c in scores) / sum(c for _, c in scores), 1)

    consistency_values = []
    for data in by_type.values():
        if data.get("consistency") is not None:
            consistency_values.append((data["consistency"], data["count"]))
    consistency = (
        round(sum(v * c for v, c in consistency_values) / sum(c for _, c in consistency_values), 1)
        if consistency_values
        else None
    )
    level = estimate_level(overall_score, stroke_count, consistency)
    issues = issue_candidates(per_stroke)
    strengths = strength_candidates(per_stroke)

    dimensions = {}
    if by_type:
        dim_totals: dict[str, list[tuple[float, int]]] = {}
        for data in by_type.values():
            for dim, value in data.get("dimensions", {}).items():
                if value is not None:
                    dim_totals.setdefault(dim, []).append((value, data["count"]))
        dimensions = {
            dim: round(sum(v * c for v, c in vals) / sum(c for _, c in vals), 1)
            for dim, vals in dim_totals.items()
        }

    _progress(on_progress, 66, "正在生成标注视频")
    annotated_path = out_dir / "annotated.mp4"
    raw_path = out_dir / "annotated_raw.mp4"
    web_compatible = False
    try:
        annotate_video(
            info,
            track,
            per_stroke,
            raw_path,
            max_seconds=MAX_ANALYSIS_SECONDS,
            on_progress=lambda p: _progress(on_progress, 66 + 14 * p, "正在生成标注视频"),
        )
        _progress(on_progress, 80, "正在转换为网页视频")
        web_compatible = transcode_for_web(raw_path, annotated_path)
        if not web_compatible:
            shutil.copyfile(raw_path, annotated_path)
        annotated_ok = annotated_path.exists()
    except Exception:
        annotated_ok = False
    finally:
        if raw_path.exists() and annotated_path.exists():
            try:
                raw_path.unlink()
            except OSError:
                pass

    _progress(on_progress, 83, "正在提取关键帧")
    keyframes = make_keyframes(info, track, per_stroke, frames_dir)
    keyframes_public = []
    keyframes_for_llm = []
    for kf in keyframes:
        keyframes_public.append(
            {
                **kf,
                "url": f"/api/jobs/{job_id}/keyframe/{kf['file']}",
            }
        )
        keyframes_for_llm.append({**kf, "path": str(frames_dir / kf["file"])})

    result = {
        "job_id": job_id,
        "created_at": time.time(),
        "video": {
            "filename": video_path.name,
            "width": info.width,
            "height": info.height,
            "fps": round(info.fps, 2),
            "duration": round(info.duration, 2),
            "analyzed_seconds": round(min(info.duration, MAX_ANALYSIS_SECONDS), 2),
            "truncated": truncated,
        },
        "player": profile,
        "handedness": handedness,
        "dominant_hand_label": "左手" if handedness == "left" else "右手",
        "stroke_count": stroke_count,
        "no_strokes": stroke_count == 0,
        "strokes": [
            {
                "id": s["id"],
                "type": s["type"],
                "type_label": s["type_label"],
                "start_time": s["start_time"],
                "contact_time": s["contact_time"],
                "end_time": s["end_time"],
                "peak_speed_mps": s["peak_speed_mps"],
                "dominant_hand": s["dominant_hand"],
                "metrics": s["metrics"],
                "scoring": s["scoring"],
            }
            for s in per_stroke
        ],
        "by_type": by_type,
        "overall_score": overall_score,
        "dimensions": dimensions,
        "consistency": consistency,
        "level": level,
        "issues": issues,
        "strengths": strengths,
        "keyframes": keyframes_public,
        "annotated_video": {
            "url": f"/api/jobs/{job_id}/video" if annotated_ok else None,
            "available": annotated_ok,
            "web_compatible": web_compatible,
        },
        "camera_notes": [
            "固定机位，侧后方约 45°，全身入镜",
            "人物高度至少占画面一半，光线充足",
            "连续完成 5–10 次同类击球，便于比较稳定性",
            "单机位姿态分析无法测量球速、旋转和落点",
        ],
    }

    _progress(on_progress, 86, "正在生成教练反馈")
    if stroke_count == 0:
        report = _empty_report("no_strokes")
    else:
        report = coach.generate_feedback(result, keyframes_for_llm)
    result["feedback"] = report
    result["llm"] = report.get("meta", {"provider": "local"})
    result["summary"] = {
        "headline": report.get("headline"),
        "score": overall_score,
        "ntrp": (level or {}).get("ntrp"),
        "issue_count": len(report.get("issues", [])),
        "stroke_count": stroke_count,
    }

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    try:
        np.savez_compressed(
            out_dir / "pose_track.npz",
            image=track.image,
            world=track.world,
            up=np.array([track.up]),
            detected=track.detected,
            fps=np.array([track.fps]),
        )
    except Exception:
        pass

    _progress(on_progress, 100, "分析完成")
    return result


__all__ = ["run_analysis", "AnalysisError"]
