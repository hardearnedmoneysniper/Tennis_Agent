from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    STROKE_LABELS,
)
from .pose import angle_at, body_frame, horizontal, mid, norm, unit


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str
    ideal_low: float
    ideal_high: float
    weight: float
    dimension: str
    decimals: int = 1


DIMENSION_LABELS = {
    "contact": "击球点",
    "rotation": "转体与准备",
    "drive": "下肢与发力链",
    "finish": "随挥与平衡",
    "consistency": "稳定性",
}


def _spec(key, label, unit, lo, hi, weight, dimension, decimals=1):
    return MetricSpec(key, label, unit, lo, hi, weight, dimension, decimals)


METRIC_SPECS: dict[str, list[MetricSpec]] = {
    "forehand": [
        _spec("contact_height", "击球点高度（相对髋部）", "× 躯干", 0.25, 0.75, 1.2, "contact"),
        _spec("contact_lateral_abs", "击球点横向距离", "× 肩宽", 0.6, 1.8, 0.9, "contact"),
        _spec("contact_forward", "击球点在身前", "× 肩宽", 0.3, 1.2, 1.0, "contact"),
        _spec("elbow_angle_contact", "触球时肘角", "°", 135, 175, 0.8, "contact", 0),
        _spec("shoulder_rotation", "肩髋分离角", "°", 20, 55, 1.2, "rotation", 0),
        _spec("stance_width", "准备站位宽度", "× 肩宽", 1.3, 2.2, 0.6, "rotation"),
        _spec("knee_bend_min", "屈膝角度（最小）", "°", 115, 155, 1.0, "drive", 0),
        _spec("wrist_speed", "挥拍速度（腕速）", "× 躯干/秒", 8, 24, 1.3, "drive"),
        _spec("follow_through_height", "随挥高度", "× 躯干", 0.1, 1.0, 1.0, "finish"),
        _spec("balance", "重心横向偏移（越小越好）", "× 肩宽", 0.0, 0.35, 0.7, "finish"),
    ],
    "backhand": [
        _spec("contact_height", "击球点高度（相对髋部）", "× 躯干", 0.15, 0.65, 1.2, "contact"),
        _spec("contact_lateral_abs", "击球点横向距离", "× 肩宽", 0.4, 1.5, 0.8, "contact"),
        _spec("contact_forward", "击球点在身前", "× 肩宽", 0.3, 1.2, 1.0, "contact"),
        _spec("elbow_angle_contact", "触球时肘角", "°", 120, 175, 0.7, "contact", 0),
        _spec("shoulder_rotation", "肩髋分离角", "°", 25, 60, 1.2, "rotation", 0),
        _spec("stance_width", "准备站位宽度", "× 肩宽", 1.2, 2.2, 0.6, "rotation"),
        _spec("knee_bend_min", "屈膝角度（最小）", "°", 115, 155, 1.0, "drive", 0),
        _spec("wrist_speed", "挥拍速度（腕速）", "× 躯干/秒", 7, 20, 1.2, "drive"),
        _spec("follow_through_height", "随挥高度", "× 躯干", 0.0, 0.8, 0.9, "finish"),
        _spec("balance", "重心横向偏移（越小越好）", "× 肩宽", 0.0, 0.35, 0.7, "finish"),
    ],
    "serve": [
        _spec("contact_height", "击球点高度（相对髋部）", "× 躯干", 1.0, 1.7, 1.3, "contact"),
        _spec("contact_lateral_abs", "击球点横向偏移", "× 肩宽", 0.0, 0.8, 0.4, "contact"),
        _spec("contact_forward", "击球点在身前", "× 肩宽", 0.2, 0.9, 0.8, "contact"),
        _spec("elbow_angle_contact", "触球时肘角", "°", 150, 180, 1.0, "contact", 0),
        _spec("shoulder_rotation", "肩髋分离角", "°", 20, 55, 1.0, "rotation", 0),
        _spec("stance_width", "站位宽度", "× 肩宽", 0.7, 1.7, 0.5, "rotation"),
        _spec("knee_bend_min", "屈膝角度（最小）", "°", 85, 135, 1.2, "drive", 0),
        _spec("wrist_speed", "挥拍速度（腕速）", "× 躯干/秒", 10, 28, 1.2, "drive"),
        _spec("toss_reach", "抛球手伸展高度", "× 躯干", 0.8, 1.6, 0.9, "drive"),
        _spec("shoulder_tilt", "肩部倾斜（奖杯姿势）", "× 肩宽", 0.05, 0.45, 1.0, "rotation"),
        _spec("hip_rise", "蹬伸幅度", "× 躯干", 0.15, 0.6, 1.0, "drive"),
        _spec("balance", "重心横向偏移（越小越好）", "× 肩宽", 0.0, 0.35, 0.6, "finish"),
    ],
    "overhead": [
        _spec("contact_height", "击球点高度（相对髋部）", "× 躯干", 0.9, 1.6, 1.2, "contact"),
        _spec("contact_forward", "击球点在身前", "× 肩宽", 0.1, 0.8, 0.8, "contact"),
        _spec("elbow_angle_contact", "触球时肘角", "°", 145, 180, 0.9, "contact", 0),
        _spec("shoulder_rotation", "肩髋分离角", "°", 15, 50, 1.0, "rotation", 0),
        _spec("knee_bend_min", "屈膝角度（最小）", "°", 100, 145, 1.0, "drive", 0),
        _spec("wrist_speed", "挥拍速度（腕速）", "× 躯干/秒", 9, 26, 1.1, "drive"),
        _spec("balance", "重心横向偏移（越小越好）", "× 肩宽", 0.0, 0.35, 0.6, "finish"),
    ],
}

WRISTS = {"right": (R_SHOULDER, R_ELBOW, R_WRIST), "left": (L_SHOULDER, L_ELBOW, L_WRIST)}


def _safe(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _elbow_angle(world: np.ndarray, frame: int, handedness: str) -> float | None:
    s, e, w = WRISTS[handedness]
    angle = angle_at(world[frame, s], world[frame, e], world[frame, w])
    return _safe(angle)


def _knee_angle(world: np.ndarray, frame: int, side: str) -> float:
    hip, knee, ankle = (L_HIP, L_KNEE, L_ANKLE) if side == "left" else (R_HIP, R_KNEE, R_ANKLE)
    return angle_at(world[frame, hip], world[frame, knee], world[frame, ankle])


def _shoulder_separation(world: np.ndarray, frame: int) -> float:
    hip_axis = horizontal(world[frame, R_HIP] - world[frame, L_HIP])
    shoulder_axis = horizontal(world[frame, R_SHOULDER] - world[frame, L_SHOULDER])
    if norm(hip_axis) < 1e-6 or norm(shoulder_axis) < 1e-6:
        return float("nan")
    cosv = abs(float(np.dot(unit(hip_axis), unit(shoulder_axis))))
    return math.degrees(math.acos(float(np.clip(cosv, -1.0, 1.0))))


def _vis(track, frame: int, idx: int) -> float:
    try:
        return float(track.image[frame, idx, 3])
    except Exception:
        return 0.0


def _mean_vis(track, frames, idxs) -> float:
    values = [_vis(track, frame, idx) for frame in frames for idx in idxs]
    return float(np.mean(values)) if values else 0.0


def compute_stroke_metrics(track, stroke: dict, handedness: str) -> dict:
    world = track.world
    fps = track.fps
    start, backswing, contact, follow_end = (
        stroke["start_frame"],
        stroke["backswing_frame"],
        stroke["contact_frame"],
        stroke["follow_end_frame"],
    )
    up = track.up
    lateral, forward, _, _ = body_frame(world[contact])
    hip = mid(world[contact], L_HIP, R_HIP)
    shoulder = mid(world[contact], L_SHOULDER, R_SHOULDER)
    torso = norm(shoulder - hip)
    shoulder_width = norm(horizontal(world[contact, R_SHOULDER] - world[contact, L_SHOULDER]))
    if torso < 0.15:
        torso = 0.45
    if shoulder_width < 0.12:
        shoulder_width = 0.42

    wrist_idx = WRISTS[handedness][2]
    shoulder_idx, elbow_idx, _ = WRISTS[handedness]
    wrist = world[contact, wrist_idx]
    contact_height = float(up * (wrist[1] - hip[1]) / torso)
    contact_lateral = float(np.dot(wrist - hip, lateral) / shoulder_width)
    contact_forward = float(np.dot(wrist - hip, forward) / shoulder_width)

    wrist_ok = _vis(track, contact, wrist_idx) >= 0.4
    shoulder_ok = _mean_vis(track, [contact], [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]) >= 0.35
    elbow_ok = _mean_vis(track, [contact], [shoulder_idx, elbow_idx, wrist_idx]) >= 0.3
    leg_frames = range(start, contact + 1)
    legs_ok = _mean_vis(track, leg_frames, [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]) >= 0.3
    ankle_ok = _mean_vis(track, range(start, follow_end + 1), [L_ANKLE, R_ANKLE]) >= 0.35
    non_dom_wrist = L_WRIST if handedness == "right" else R_WRIST
    non_dom_ok = _mean_vis(track, range(start, contact + 1), [non_dom_wrist]) >= 0.3

    shoulder_rot_values = []
    for f in range(start, contact + 1):
        if _mean_vis(track, [f], [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]) < 0.3:
            continue
        value = _shoulder_separation(world, f)
        if math.isfinite(value):
            shoulder_rot_values.append(value)
    shoulder_rotation = max(shoulder_rot_values) if shoulder_rot_values else None

    knee_values = []
    for f in range(start, contact + 1):
        if _mean_vis(track, [f], [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]) < 0.3:
            continue
        left, right = _knee_angle(world, f, "left"), _knee_angle(world, f, "right")
        vals = [v for v in (left, right) if math.isfinite(v)]
        if vals:
            knee_values.append(sum(vals) / len(vals))
    knee_bend_min = min(knee_values) if knee_values else None

    stance_width = None
    if ankle_ok:
        back_lateral = horizontal(world[backswing, R_HIP] - world[backswing, L_HIP])
        if norm(back_lateral) > 1e-6:
            ankle_vec = horizontal(world[backswing, R_ANKLE] - world[backswing, L_ANKLE])
            sw = norm(horizontal(world[backswing, R_SHOULDER] - world[backswing, L_SHOULDER]))
            if sw > 0.12:
                stance_width = norm(ankle_vec) / sw

    follow_through_height = None
    if wrist_ok:
        follow_values = []
        for f in range(contact, max(contact + 1, follow_end + 1)):
            if _vis(track, f, wrist_idx) >= 0.35:
                follow_values.append(float(up * (world[f, wrist_idx][1] - shoulder[1]) / torso))
        follow_through_height = max(follow_values) if follow_values else None

    balance = None
    if ankle_ok:
        hip_sway = 0.0
        for f in range(start, max(contact + 1, follow_end + 1)):
            ankle_mid = mid(world[f], L_ANKLE, R_ANKLE)
            hip_mid = mid(world[f], L_HIP, R_HIP)
            hip_sway = max(hip_sway, abs(float(np.dot(horizontal(hip_mid - ankle_mid), lateral))))
        balance = hip_sway / shoulder_width

    hip_rise = None
    if legs_ok:
        hip_heights = [float(up * mid(world[f], L_HIP, R_HIP)[1]) for f in range(start, contact + 1)]
        hip_rise = (max(hip_heights) - min(hip_heights)) / torso if hip_heights else None

    toss_reach = None
    if non_dom_ok:
        toss_heights = [
            float(up * (world[f, non_dom_wrist][1] - shoulder[1]) / torso)
            for f in range(start, contact + 1)
            if _vis(track, f, non_dom_wrist) >= 0.3
        ]
        toss_reach = max(toss_heights) if toss_heights else None

    shoulder_tilt = None
    if shoulder_ok:
        tilt_frame = backswing
        if handedness == "right":
            shoulder_tilt = float(up * (world[tilt_frame, L_SHOULDER][1] - world[tilt_frame, R_SHOULDER][1]) / shoulder_width)
        else:
            shoulder_tilt = float(up * (world[tilt_frame, R_SHOULDER][1] - world[tilt_frame, L_SHOULDER][1]) / shoulder_width)

    speed = stroke.get("peak_speed_mps")
    wrist_speed = None if speed is None else float(speed) / torso

    return {
        "contact_height": _safe(contact_height) if wrist_ok and shoulder_ok else None,
        "contact_lateral_abs": _safe(abs(contact_lateral)) if wrist_ok and shoulder_ok else None,
        "contact_lateral_signed": _safe(contact_lateral) if wrist_ok and shoulder_ok else None,
        "contact_forward": _safe(contact_forward) if wrist_ok and shoulder_ok else None,
        "elbow_angle_contact": _elbow_angle(world, contact, handedness) if elbow_ok else None,
        "shoulder_rotation": _safe(shoulder_rotation),
        "stance_width": _safe(stance_width),
        "knee_bend_min": _safe(knee_bend_min),
        "wrist_speed": _safe(wrist_speed) if wrist_ok else None,
        "follow_through_height": _safe(follow_through_height),
        "balance": _safe(balance),
        "toss_reach": _safe(toss_reach),
        "shoulder_tilt": _safe(shoulder_tilt),
        "hip_rise": _safe(hip_rise),
        "peak_speed_mps": _safe(speed),
        "torso_m": float(torso),
        "shoulder_width_m": float(shoulder_width),
        "swing_duration": (follow_end - start) / fps,
        "backswing_duration": (backswing - start) / fps,
        "forward_duration": (contact - backswing) / fps,
        "contact_time": contact / fps,
        "visibility": {
            "wrist": wrist_ok,
            "shoulders": shoulder_ok,
            "elbow": elbow_ok,
            "legs": legs_ok,
            "ankles": ankle_ok,
            "toss_arm": non_dom_ok,
        },
    }


def score_metric(spec: MetricSpec, value: float | None) -> float | None:
    value = _safe(value)
    if value is None:
        return None
    if spec.ideal_low <= value <= spec.ideal_high:
        return 100.0
    width = max(spec.ideal_high - spec.ideal_low, 0.2)
    distance = spec.ideal_low - value if value < spec.ideal_low else value - spec.ideal_high
    return max(0.0, 100.0 * (1.0 - distance / width))


def status_for(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 80:
        return "good"
    if score >= 55:
        return "ok"
    return "poor"


def aggregate_by_type(per_stroke: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for stroke in per_stroke:
        grouped.setdefault(stroke["type"], []).append(stroke)

    result: dict[str, dict] = {}
    for stroke_type, strokes in grouped.items():
        rows_by_key: dict[str, list[dict]] = {}
        for stroke in strokes:
            for row in stroke["scoring"]["metrics"]:
                rows_by_key.setdefault(row["key"], []).append(row)

        metrics = []
        for key, rows in rows_by_key.items():
            metrics.append(
                {
                    "key": key,
                    "label": rows[0]["label"],
                    "unit": rows[0]["unit"],
                    "value": round(float(np.mean([r["value"] for r in rows])), rows[0]["decimals"]),
                    "ideal_low": rows[0]["ideal_low"],
                    "ideal_high": rows[0]["ideal_high"],
                    "score": round(float(np.mean([r["score"] for r in rows])), 1),
                    "status": status_for(float(np.mean([r["score"] for r in rows]))),
                    "dimension": rows[0]["dimension"],
                    "dimension_label": rows[0]["dimension_label"],
                    "decimals": rows[0]["decimals"],
                    "samples": len(rows),
                }
            )
        metrics.sort(key=lambda m: (list(DIMENSION_LABELS).index(m["dimension"]), -m["score"]))

        dims = {}
        for dimension in DIMENSION_LABELS:
            vals = [s["scoring"]["dimensions"].get(dimension) for s in strokes]
            vals = [v for v in vals if v is not None]
            if vals:
                dims[dimension] = round(float(np.mean(vals)), 1)

        scores = [s["scoring"]["score"] for s in strokes if s["scoring"]["score"] is not None]
        consistency, consistency_meta = consistency_score(strokes)
        result[stroke_type] = {
            "count": len(strokes),
            "score": round(float(np.mean(scores)), 1) if scores else None,
            "dimensions": dims,
            "metrics": metrics,
            "consistency": consistency,
            "consistency_meta": consistency_meta,
        }
    return result


def score_stroke(stroke_type: str, metrics: dict) -> dict:
    specs = METRIC_SPECS[stroke_type]
    metric_rows: list[dict] = []
    dimension_totals: dict[str, list[tuple[float, float]]] = {}
    weighted_sum = 0.0
    weight_total = 0.0
    missing_weight = 0.0

    for spec in specs:
        value = _safe(metrics.get(spec.key))
        score = score_metric(spec, value)
        if score is None:
            missing_weight += spec.weight
            continue
        metric_rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "unit": spec.unit,
                "value": value,
                "ideal_low": spec.ideal_low,
                "ideal_high": spec.ideal_high,
                "score": round(score, 1),
                "status": status_for(score),
                "dimension": spec.dimension,
                "dimension_label": DIMENSION_LABELS[spec.dimension],
                "weight": spec.weight,
                "decimals": spec.decimals,
            }
        )
        weighted_sum += score * spec.weight
        weight_total += spec.weight
        dimension_totals.setdefault(spec.dimension, []).append((score, spec.weight))

    dimensions = {}
    for dim, values in dimension_totals.items():
        w = sum(weight for _, weight in values)
        dimensions[dim] = round(sum(score * weight for score, weight in values) / w, 1)

    return {
        "score": round(weighted_sum / weight_total, 1) if weight_total else None,
        "metrics": metric_rows,
        "dimensions": dimensions,
        "dimension_labels": DIMENSION_LABELS,
        "confidence": round(weight_total / (weight_total + missing_weight), 2) if weight_total else 0.0,
        "missing_metrics": [s.key for s in specs if metrics.get(s.key) is None],
    }


def consistency_score(strokes: list[dict]) -> tuple[float | None, dict]:
    """Coefficient-of-variation based consistency across grouped strokes."""
    if len(strokes) < 2:
        return None, {"samples": len(strokes)}
    values = {"contact_height": [], "contact_forward": [], "contact_lateral_signed": [], "swing_duration": []}
    for stroke in strokes:
        m = stroke["metrics"]
        for key in values:
            v = _safe(m.get(key))
            if v is not None:
                values[key].append(v)
    cvs = {}
    for key, vals in values.items():
        if len(vals) >= 2:
            mean = float(np.mean(vals))
            sd = float(np.std(vals))
            cvs[key] = abs(sd / mean) if abs(mean) > 1e-6 else 0.0
    if not cvs:
        return None, {"samples": len(strokes)}
    avg_cv = float(np.mean(list(cvs.values())))
    score = max(0.0, min(100.0, 100.0 - avg_cv * 240.0))
    return round(score, 1), {"samples": len(strokes), "cv": {k: round(v, 3) for k, v in cvs.items()}}


def estimate_level(overall_score: float | None, stroke_count: int, consistency: float | None) -> dict | None:
    if overall_score is None:
        return None
    score = overall_score
    if consistency is not None:
        score = 0.85 * score + 0.15 * consistency
    if score >= 88:
        ntrp, label = "4.5", "高级"
    elif score >= 80:
        ntrp, label = "4.0", "进阶高级"
    elif score >= 72:
        ntrp, label = "3.5", "进阶"
    elif score >= 62:
        ntrp, label = "3.0", "中级"
    elif score >= 52:
        ntrp, label = "2.5", "初中级"
    else:
        ntrp, label = "2.0", "初学"
    confidence = "high" if stroke_count >= 5 else "medium" if stroke_count >= 3 else "low"
    return {
        "ntrp": ntrp,
        "label": label,
        "score": round(score, 1),
        "confidence": confidence,
        "note": "基于单机位姿态估算，仅供训练参考，不能替代教练现场评估。",
    }


def issue_candidates(per_stroke: list[dict]) -> list[dict]:
    """Aggregate metric scores across strokes and return the weakest metrics."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for stroke in per_stroke:
        for row in stroke["scoring"]["metrics"]:
            grouped.setdefault((stroke["type"], row["key"]), []).append(row)

    candidates = []
    for (stroke_type, key), rows in grouped.items():
        scores = [r["score"] for r in rows]
        avg_score = float(np.mean(scores))
        if avg_score >= 65:
            continue
        avg_value = float(np.mean([r["value"] for r in rows]))
        spec = next((s for s in METRIC_SPECS[stroke_type] if s.key == key), None)
        if spec is None:
            continue
        candidates.append(
            {
                "stroke_type": stroke_type,
                "stroke_label": STROKE_LABELS[stroke_type],
                "key": key,
                "label": spec.label,
                "unit": spec.unit,
                "value": round(avg_value, spec.decimals),
                "ideal_low": spec.ideal_low,
                "ideal_high": spec.ideal_high,
                "score": round(avg_score, 1),
                "weight": spec.weight,
                "samples": len(rows),
                "direction": "low" if avg_value < spec.ideal_low else "high",
            }
        )
    candidates.sort(key=lambda x: (100 - x["score"]) * x["weight"], reverse=True)
    return candidates


def strength_candidates(per_stroke: list[dict], limit: int = 3) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for stroke in per_stroke:
        for row in stroke["scoring"]["metrics"]:
            grouped.setdefault((stroke["type"], row["key"]), []).append(row)
    strengths = []
    for (stroke_type, key), rows in grouped.items():
        avg_score = float(np.mean([r["score"] for r in rows]))
        if avg_score < 80:
            continue
        spec = next((s for s in METRIC_SPECS[stroke_type] if s.key == key), None)
        if spec is None:
            continue
        strengths.append(
            {
                "stroke_type": stroke_type,
                "stroke_label": STROKE_LABELS[stroke_type],
                "key": key,
                "label": spec.label,
                "value": round(float(np.mean([r["value"] for r in rows])), spec.decimals),
                "unit": spec.unit,
                "score": round(avg_score, 1),
                "samples": len(rows),
            }
        )
    strengths.sort(key=lambda x: x["score"], reverse=True)
    return strengths[:limit]
