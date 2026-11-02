from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .config import L_ELBOW, L_WRIST, MAX_STROKES, NOSE, R_ELBOW, R_WRIST, STROKE_LABELS
from .pose import PoseTrack, body_frame, horizontal, mid, norm


def wrist_speed_series(track: PoseTrack, wrist_idx: int) -> np.ndarray:
    wrist = track.world[:, wrist_idx, :]
    if len(wrist) < 3:
        return np.zeros(len(wrist), dtype=float)
    velocity = np.gradient(wrist, axis=0) * track.fps
    speed = np.linalg.norm(velocity, axis=1)
    if len(speed) >= 5:
        speed = savgol_filter(speed, 5, 2, mode="interp")
    return np.maximum(speed, 0.0)


def infer_handedness(track: PoseTrack, requested: str | None) -> str:
    if requested in ("left", "right"):
        return requested
    right_speed = wrist_speed_series(track, R_WRIST)
    left_speed = wrist_speed_series(track, L_WRIST)
    right_vis = float(np.nanmean(track.image[:, R_WRIST, 3]))
    left_vis = float(np.nanmean(track.image[:, L_WRIST, 3]))
    right_score = float(np.percentile(right_speed, 95)) * (0.5 + 0.5 * right_vis)
    left_score = float(np.percentile(left_speed, 95)) * (0.5 + 0.5 * left_vis)
    if right_score >= left_score * 0.98:
        return "right"
    return "left"


def _classify(track: PoseTrack, peak: int, start: int, handedness: str) -> tuple[str, dict]:
    world = track.world
    up = track.up
    lateral, _, _, _ = body_frame(world[peak])
    hip = mid(world[peak], 23, 24)
    shoulder = mid(world[peak], 11, 12)
    torso = norm(shoulder - hip)
    shoulder_width = norm(horizontal(world[peak, 12] - world[peak, 11]))
    if torso < 0.15:
        torso = 0.45
    if shoulder_width < 0.12:
        shoulder_width = 0.42

    wrist_idx = R_WRIST if handedness == "right" else L_WRIST
    wrist = world[peak, wrist_idx]
    height_above_shoulder = float(up * (wrist[1] - shoulder[1]) / torso)
    height_above_hip = float(up * (wrist[1] - hip[1]) / torso)
    wrist_above_nose = up * (wrist[1] - world[peak, NOSE][1]) > 0.02

    non_dom = L_WRIST if handedness == "right" else R_WRIST
    toss_heights = [
        float(up * (world[f, non_dom][1] - shoulder[1]) / torso)
        for f in range(start, peak + 1)
    ]
    toss_reach = max(toss_heights) if toss_heights else 0.0

    overhead = height_above_shoulder > 0.5 or (wrist_above_nose and height_above_hip > 0.85)
    lateral_offset = float(np.dot(wrist - hip, lateral) / shoulder_width)
    hand_sign = 1.0 if handedness == "right" else -1.0
    # A high finish on a groundstroke also puts the wrist above the head; real
    # overhead contacts are near the body's vertical centerline.
    if overhead and abs(lateral_offset) < 0.7:
        stroke_type = "serve" if toss_reach > 0.45 else "overhead"
        return stroke_type, {"toss_reach": toss_reach, "height_above_shoulder": height_above_shoulder}

    signed = lateral_offset * hand_sign
    if abs(signed) < 0.12:
        elbow_idx = R_ELBOW if handedness == "right" else L_ELBOW
        elbow_offset = float(np.dot(world[peak, elbow_idx] - hip, lateral) / shoulder_width)
        signed = elbow_offset * hand_sign
    stroke_type = "forehand" if signed >= 0 else "backhand"
    return stroke_type, {"lateral_offset": lateral_offset, "height_above_shoulder": height_above_shoulder}


def detect_strokes(track: PoseTrack, requested_handedness: str | None = "auto") -> tuple[str, list[dict]]:
    handedness = infer_handedness(track, requested_handedness)
    wrist_idx = R_WRIST if handedness == "right" else L_WRIST
    speed = wrist_speed_series(track, wrist_idx)
    n = len(speed)
    if n < 8:
        return handedness, []

    torso_samples = []
    for f in range(n):
        shoulder = mid(track.world[f], 11, 12)
        hip = mid(track.world[f], 23, 24)
        torso_samples.append(norm(shoulder - hip))
    torso = float(np.median([t for t in torso_samples if t > 0.15] or [0.45]))
    if torso < 0.15:
        torso = 0.45

    normalized = speed / torso
    threshold_abs = max(1.8, float(np.percentile(speed, 90)) * 0.5)
    threshold_norm = max(4.0, float(np.percentile(normalized, 90)) * 0.5)
    threshold = max(threshold_abs, threshold_norm * torso)

    visibility = track.image[:, wrist_idx, 3]
    peaks, _ = find_peaks(
        speed,
        height=threshold,
        distance=max(4, int(0.55 * track.fps)),
        prominence=max(0.6, threshold * 0.35),
    )
    if len(peaks) == 0:
        return handedness, []

    peaks = sorted(peaks, key=lambda p: float(speed[p]), reverse=True)[:MAX_STROKES]
    peaks = sorted(peaks)

    strokes: list[dict] = []
    for peak in peaks:
        peak = int(peak)
        peak_speed = float(speed[peak])
        if peak < int(0.25 * track.fps) or peak > n - int(0.2 * track.fps):
            continue
        vis_window = visibility[max(0, peak - int(0.2 * track.fps)): peak + int(0.2 * track.fps) + 1]
        if len(vis_window) == 0 or float(np.mean(vis_window)) < 0.35:
            continue

        lower = max(0, peak - int(1.2 * track.fps))
        pre = np.where(speed[lower:peak] < peak_speed * 0.25)[0]
        start = lower + int(pre[-1]) + 1 if len(pre) else max(0, peak - int(0.9 * track.fps))
        start = min(start, peak - max(2, int(0.15 * track.fps)))

        upper = min(n - 1, peak + int(0.9 * track.fps))
        post = np.where(speed[peak + 1: upper + 1] < peak_speed * 0.3)[0]
        follow_end = peak + 1 + int(post[0]) if len(post) else min(n - 1, peak + int(0.6 * track.fps))
        follow_end = max(follow_end, peak + 1)

        backswing = start + int(np.argmin(speed[start:peak]))
        min_gap = max(2, int(0.12 * track.fps))
        if backswing > peak - min_gap:
            backswing = max(start, peak - min_gap)

        # Wrist speed peaks slightly after the true contact, often at the high
        # finish. Within the fast-swing window, the most forward wrist position
        # is a better proxy for the contact point.
        _, forward_peak, _, _ = body_frame(track.world[peak])
        hip_peak = mid(track.world[peak], 23, 24)
        contact_lower = max(backswing, peak - int(0.35 * track.fps))
        fast_frames = [
            f for f in range(contact_lower, peak + 1)
            if speed[f] >= 0.6 * peak_speed
        ]
        contact = peak
        if fast_frames:
            contact = max(
                fast_frames,
                key=lambda f: float(np.dot(track.world[f, wrist_idx] - hip_peak, forward_peak)),
            )
        contact = max(backswing + 1, min(contact, peak))

        wrist_segment = track.world[start:follow_end + 1, wrist_idx, :]
        travel = float(np.sum(np.linalg.norm(np.diff(wrist_segment, axis=0), axis=1))) if len(wrist_segment) > 1 else 0.0
        if travel < max(0.35, 0.5 * torso):
            continue

        stroke_type, extra = _classify(track, contact, start, handedness)
        strokes.append(
            {
                "id": len(strokes) + 1,
                "type": stroke_type,
                "type_label": STROKE_LABELS[stroke_type],
                "start_frame": int(start),
                "backswing_frame": int(backswing),
                "contact_frame": int(contact),
                "follow_end_frame": int(follow_end),
                "start_time": round(start / track.fps, 3),
                "contact_time": round(contact / track.fps, 3),
                "end_time": round(follow_end / track.fps, 3),
                "peak_speed_mps": round(peak_speed, 2),
                "dominant_hand": handedness,
                "extra": extra,
            }
        )

    strokes.sort(key=lambda s: s["contact_frame"])
    deduped: list[dict] = []
    min_gap = max(3, int(0.3 * track.fps))
    for stroke in strokes:
        if deduped and stroke["contact_frame"] - deduped[-1]["contact_frame"] < min_gap:
            if stroke["peak_speed_mps"] > deduped[-1]["peak_speed_mps"]:
                deduped[-1] = stroke
            continue
        deduped.append(stroke)
    for index, stroke in enumerate(deduped, start=1):
        stroke["id"] = index

    return handedness, deduped
