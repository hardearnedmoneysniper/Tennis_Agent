from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

from .config import OUTPUTS_DIR, STROKE_LABELS, ensure_dirs
from .pipeline import AnalysisError, run_analysis


def _markdown(result: dict) -> str:
    feedback = result.get("feedback", {})
    level = result.get("level") or {}
    lines = [f"# 网球动作分析报告", ""]
    lines.append(f"- 视频：{result['video']['filename']}（{result['video']['duration']} 秒）")
    lines.append(f"- 检测击球：{result['stroke_count']} 次")
    if level:
        lines.append(f"- 整体评分：{level['score']}/100，估算 NTRP {level['ntrp']}（{level['label']}）")
    lines.append("")
    if feedback.get("headline"):
        lines.append(f"## {feedback['headline']}")
        lines.append("")
    if feedback.get("overall"):
        lines.append(feedback["overall"])
        lines.append("")

    lines.append("## 分项指标")
    for stroke_type, data in result.get("by_type", {}).items():
        lines.append(f"### {STROKE_LABELS.get(stroke_type, stroke_type)}（{data['count']} 次，评分 {data.get('score')}）")
        lines.append("")
        lines.append("| 指标 | 实测 | 参考区间 | 评分 |")
        lines.append("| --- | --- | --- | --- |")
        for m in data.get("metrics", []):
            lines.append(
                f"| {m['label']} | {m['value']} {m['unit']} | {m['ideal_low']}–{m['ideal_high']} {m['unit']} | {m['score']} |"
            )
        lines.append("")

    issues = feedback.get("issues", [])
    if issues:
        lines.append("## 优先改进")
        for issue in issues:
            lines.append(f"### P{issue.get('priority', '')} {issue.get('title', '')}（{issue.get('stroke', '')}）")
            for key, label in (("observation", "观察"), ("why", "原因"), ("fix", "怎么改"), ("drill", "练习")):
                if issue.get(key):
                    lines.append(f"- **{label}**：{issue[key]}")
            lines.append("")

    strengths = feedback.get("strengths", [])
    if strengths:
        lines.append("## 做得好的地方")
        for item in strengths:
            lines.append(f"- **{item.get('title', '')}**：{item.get('detail', '')}")
        lines.append("")

    plan = feedback.get("training_plan", {})
    if plan.get("weeks"):
        lines.append("## 训练计划")
        if plan.get("focus"):
            lines.append(f"重点：{plan['focus']}")
            lines.append("")
        for week in plan["weeks"]:
            lines.append(f"### 第 {week['week']} 周 · {week['theme']}")
            for session in week.get("sessions", []):
                lines.append(f"- **{session['day']} · {session['title']}（{session['duration_min']} 分钟）**")
                for drill in session.get("drills", []):
                    lines.append(f"  - {drill['name']}：{drill['sets']}。{drill['notes']}")
        lines.append("")

    if feedback.get("success_metrics"):
        lines.append("## 验收指标")
        for item in feedback["success_metrics"]:
            lines.append(f"- {item}")
        lines.append("")

    if feedback.get("encouragement"):
        lines.append(f"> {feedback['encouragement']}")
        lines.append("")

    if result.get("keyframes"):
        lines.append("## 关键帧")
        for kf in result["keyframes"]:
            lines.append(f"- {kf['label']}（{kf['time']} 秒）")
        lines.append("")

    lines.append("---")
    lines.append("说明：本报告基于单机位人体姿态估计，未追踪球和球拍，不能替代现场教练或医学诊断。")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
    parser = argparse.ArgumentParser(description="从网球视频分析动作、水平并生成训练计划")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("--hand", choices=["auto", "left", "right"], default="auto", help="惯用手")
    parser.add_argument("--level", default="", help="自评水平，例如 3.0")
    parser.add_argument("--goal", default="", help="本次想改进的目标")
    parser.add_argument("--name", default="", help="称呼")
    parser.add_argument("--notes", default="", help="补充信息")
    args = parser.parse_args(argv)

    ensure_dirs()
    video_path = Path(args.video).expanduser()
    if not video_path.is_file():
        print(f"找不到视频：{video_path}", file=sys.stderr)
        return 2

    profile = {
        "name": args.name,
        "handedness": args.hand,
        "level": args.level,
        "goal": args.goal,
        "notes": args.notes,
    }
    job_id = "cli_" + uuid.uuid4().hex[:8]
    last_stage = ""
    started = time.time()

    def progress(percent, stage):
        nonlocal last_stage
        if stage != last_stage:
            print(f"[{percent:5.1f}%] {stage}", flush=True)
            last_stage = stage

    try:
        result = run_analysis(job_id, video_path, profile, on_progress=progress)
    except AnalysisError as exc:
        print(f"分析失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已取消", file=sys.stderr)
        return 130

    out_dir = OUTPUTS_DIR / job_id
    report_md = _markdown(result)
    report_path = out_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")

    level = result.get("level") or {}
    print()
    print(f"完成，用时 {time.time() - started:.1f} 秒")
    print(f"击球数：{result['stroke_count']}")
    if level:
        print(f"整体评分：{level['score']}/100  NTRP 约 {level['ntrp']}（{level['label']}）")
    print(f"报告：{report_path}")
    if result.get("annotated_video", {}).get("url"):
        print(f"标注视频：{out_dir / 'annotated.mp4'}")
    issues = result.get("feedback", {}).get("issues", [])
    if issues:
        print("优先改进：")
        for issue in issues:
            print(f"  P{issue.get('priority')} {issue.get('title')}（{issue.get('stroke')}）")
    print(f"LLM：{result.get('llm', {}).get('provider')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
