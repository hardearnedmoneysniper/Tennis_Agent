from __future__ import annotations

import json
import logging
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from tennis_agent import coach
from tennis_agent.config import (
    MAX_UPLOAD_MB,
    OUTPUTS_DIR,
    SUPPORTED_EXTENSIONS,
    UPLOADS_DIR,
    WEB_DIR,
    ensure_dirs,
)
from tennis_agent.pipeline import AnalysisError, run_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tennis")

ensure_dirs()

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
ANALYSIS_SEMAPHORE = threading.Semaphore(1)


def _valid_job_id(job_id: str | None) -> bool:
    return bool(job_id and re.fullmatch(r"[A-Za-z0-9_-]{1,40}", job_id))


def _load_saved_result(job_id: str) -> dict | None:
    if not _valid_job_id(job_id):
        return None
    path = OUTPUTS_DIR / job_id / "result.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _job_public(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        result = _load_saved_result(job_id)
        if result is None:
            return None
        return {
            "job_id": job_id,
            "status": "done",
            "progress": 100,
            "stage": "分析完成",
            "error": None,
            "created_at": result.get("created_at", time.time()),
            "result": result,
        }
    payload = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "error": job["error"],
        "created_at": job["created_at"],
    }
    if job["status"] == "done":
        payload["result"] = job["result"]
    return payload


def _update(job_id: str, **changes) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(changes)


def _run_job(job_id: str, video_path: Path, profile: dict) -> None:
    with ANALYSIS_SEMAPHORE:
        _update(job_id, status="running", stage="开始分析", progress=1)
        try:
            result = run_analysis(
                job_id=job_id,
                video_path=video_path,
                profile=profile,
                on_progress=lambda percent, stage: _update(job_id, progress=round(percent, 1), stage=stage),
            )
            _update(job_id, status="done", progress=100, stage="分析完成", result=result)
            log.info("job %s done: %s strokes", job_id, result.get("stroke_count"))
        except AnalysisError as exc:
            log.warning("job %s analysis error: %s", job_id, exc)
            _update(job_id, status="error", error=str(exc), stage="分析失败")
        except Exception as exc:
            log.error("job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            _update(
                job_id,
                status="error",
                error=f"分析过程出错：{type(exc).__name__}: {exc}",
                stage="分析失败",
            )


def _sanitize_profile(raw: dict | None) -> dict:
    raw = raw or {}
    handedness = raw.get("handedness", "auto")
    if handedness not in ("auto", "left", "right"):
        handedness = "auto"
    return {
        "name": str(raw.get("name", ""))[:40],
        "handedness": handedness,
        "level": str(raw.get("level", ""))[:40],
        "goal": str(raw.get("goal", ""))[:200],
        "notes": str(raw.get("notes", ""))[:500],
    }


@app.get("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "llm": coach.llm_status()})


@app.get("/api/llm")
def llm_status():
    return jsonify(coach.llm_status())


@app.post("/api/llm/key")
def llm_set_key():
    data = request.get_json(silent=True) or {}
    key = data.get("key")
    if key is not None and not isinstance(key, str):
        return jsonify({"error": "key 必须是字符串"}), 400
    coach.set_api_key(key)
    return jsonify(coach.llm_status())


@app.post("/api/analyze")
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "请选择视频文件"}), 400
    upload = request.files["video"]
    original = upload.filename or ""
    ext = Path(original).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"不支持的视频格式：{ext or '未知'}。支持 {', '.join(sorted(SUPPORTED_EXTENSIONS))}"}), 400
    if not original:
        return jsonify({"error": "文件名无效"}), 400

    profile_raw = request.form.get("profile", "{}")
    try:
        profile_data = json.loads(profile_raw) if isinstance(profile_raw, str) else {}
    except json.JSONDecodeError:
        return jsonify({"error": "球员信息格式错误"}), 400
    if not isinstance(profile_data, dict):
        profile_data = {}
    profile = _sanitize_profile(profile_data)

    job_id = uuid.uuid4().hex[:12]
    upload_path = UPLOADS_DIR / f"{job_id}{ext}"
    upload.save(str(upload_path))
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "stage": "排队中",
            "error": None,
            "result": None,
            "created_at": time.time(),
            "profile": profile,
            "video": original,
        }
    thread = threading.Thread(target=_run_job, args=(job_id, upload_path, profile), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    payload = _job_public(job_id)
    if payload is None:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(payload)


@app.get("/api/jobs/<job_id>/video")
def job_video(job_id: str):
    if not _valid_job_id(job_id):
        return jsonify({"error": "任务不存在"}), 404
    job_dir = OUTPUTS_DIR / job_id
    if not (job_dir / "annotated.mp4").exists():
        return jsonify({"error": "标注视频不存在"}), 404
    return send_from_directory(str(job_dir), "annotated.mp4", conditional=True)


@app.get("/api/jobs/<job_id>/keyframe/<path:filename>")
def job_keyframe(job_id: str, filename: str):
    if not _valid_job_id(job_id):
        return jsonify({"error": "任务不存在"}), 404
    keyframe_dir = OUTPUTS_DIR / job_id / "keyframes"
    if not (keyframe_dir / filename).is_file():
        return jsonify({"error": "关键帧不存在"}), 404
    return send_from_directory(str(keyframe_dir), filename, conditional=True)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    messages = data.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "消息为空"}), 400
    if not _valid_job_id(job_id):
        return jsonify({"error": "任务不存在"}), 404
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job and job["status"] != "done":
        return jsonify({"error": "分析尚未完成"}), 409
    result = job.get("result") if job and job["status"] == "done" else _load_saved_result(job_id)
    if not result:
        return jsonify({"error": "任务不存在"}), 404
    question = str(messages[-1].get("content", "")).strip()
    if not question:
        return jsonify({"error": "问题为空"}), 400
    if len(question) > 2000:
        return jsonify({"error": "问题太长，请控制在 2000 字以内"}), 400
    history = [
        {"role": m.get("role"), "content": m.get("content")}
        for m in messages[:-1]
        if isinstance(m, dict)
    ]
    reply = coach.chat_reply(result, history, question)
    return jsonify(reply)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify({"error": f"文件太大，上限 {MAX_UPLOAD_MB} MB"}), 413


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify({"error": "接口不存在"}), 404
    return send_from_directory(str(WEB_DIR), "index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)
