from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
UPLOADS_DIR = ROOT / "uploads"
OUTPUTS_DIR = ROOT / "outputs"

ANALYSIS_FPS = float(os.environ.get("TENNIS_ANALYSIS_FPS", "25"))
MAX_ANALYSIS_SECONDS = float(os.environ.get("TENNIS_MAX_SECONDS", "120"))
MAX_UPLOAD_MB = int(os.environ.get("TENNIS_MAX_UPLOAD_MB", "300"))
VIDEO_MAX_WIDTH = int(os.environ.get("TENNIS_VIDEO_WIDTH", "1280"))
POSE_MAX_WIDTH = int(os.environ.get("TENNIS_POSE_WIDTH", "960"))
POSE_COMPLEXITY = int(os.environ.get("TENNIS_POSE_COMPLEXITY", "1"))
MAX_STROKES = int(os.environ.get("TENNIS_MAX_STROKES", "24"))

LLM_MODEL = os.environ.get("TENNIS_LLM_MODEL", "claude-opus-5-5")
LLM_EFFORT = os.environ.get("TENNIS_LLM_EFFORT", "medium")

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

# MediaPipe Pose landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT_INDEX, R_FOOT_INDEX = 31, 32

STROKE_LABELS = {
    "forehand": "正手",
    "backhand": "反手",
    "serve": "发球",
    "overhead": "高压",
}

STROKE_COLORS = {
    "forehand": "#d7ff3f",
    "backhand": "#55d6ff",
    "serve": "#ffb347",
    "overhead": "#ff7b9c",
}


def ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
