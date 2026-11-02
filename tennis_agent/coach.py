from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import anthropic
from PIL import Image

from .config import LLM_EFFORT, LLM_MODEL, STROKE_LABELS

SYSTEM_PROMPT = (
    "你是一位资深网球教练（中国网球协会认证），擅长通过视频关键帧和姿态数据诊断业余球员。"
    "你的反馈必须具体、可执行、鼓励式，使用简体中文。"
    "你会明确区分“姿态数据测量结果”和“教练判断”。你不做医学诊断，不承诺伤病恢复。"
    "你只基于提供的姿态数据和关键帧分析，不臆测球速、旋转、落点等未测量信息。"
)

_api_key_override: str | None = None


class CoachError(RuntimeError):
    pass


def set_api_key(key: str | None) -> None:
    global _api_key_override
    _api_key_override = (key or "").strip() or None


def _credentials() -> str | None:
    return _api_key_override or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")


def llm_status() -> dict:
    credentials = _credentials()
    return {
        "provider": "claude" if credentials else "local",
        "ready": bool(credentials),
        "model": LLM_MODEL if credentials else None,
        "reason": None if credentials else "未配置 ANTHROPIC_API_KEY，当前使用本地规则教练。",
    }


REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "level_assessment": {"type": "string"},
        "overall": {"type": "string"},
        "strengths": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "required": ["title", "detail"],
                "additionalProperties": False,
            },
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "priority": {"type": "integer"},
                    "title": {"type": "string"},
                    "stroke": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "observation": {"type": "string"},
                    "why": {"type": "string"},
                    "fix": {"type": "string"},
                    "drill": {"type": "string"},
                },
                "required": ["priority", "title", "stroke", "severity", "observation", "why", "fix", "drill"],
                "additionalProperties": False,
            },
        },
        "training_plan": {
            "type": "object",
            "properties": {
                "focus": {"type": "string"},
                "weeks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "week": {"type": "integer"},
                            "theme": {"type": "string"},
                            "sessions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "day": {"type": "string"},
                                        "duration_min": {"type": "integer"},
                                        "title": {"type": "string"},
                                        "drills": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "sets": {"type": "string"},
                                                    "notes": {"type": "string"},
                                                },
                                                "required": ["name", "sets", "notes"],
                                                "additionalProperties": False,
                                            },
                                        },
                                    },
                                    "required": ["day", "duration_min", "title", "drills"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["week", "theme", "sessions"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["focus", "weeks"],
            "additionalProperties": False,
        },
        "success_metrics": {"type": "array", "items": {"type": "string"}},
        "encouragement": {"type": "string"},
    },
    "required": ["headline", "level_assessment", "overall", "strengths", "issues", "training_plan", "success_metrics", "encouragement"],
    "additionalProperties": False,
}


COACHING: dict[str, dict[str, dict[str, str]]] = {
    "contact_height": {
        "low": {
            "title": "击球点偏低",
            "why": "触球点低于合理区间，容易在球下降后才击打，导致被球挤住、回球偏浅或下网。",
            "fix": "提前判断来球落点，用小碎步把身体带到球的侧前方，在腰胸高度、身体前侧击球。",
            "drill": "高位定点击球",
            "detail": "让同伴或发球机把球喂到腰胸高度，在身体前侧定点击球；每组 20 球 × 3 组，重点感受“在球上升/最高点附近击球”。",
        },
        "high": {
            "title": "击球点偏高",
            "why": "触球点高于常规区间。处理高球时这是优势，但普通来球等得太高会损失向前发力和稳定性。",
            "fix": "普通球不要等它升到肩部再打，提前进入击球区；高球则保持拍头高于手腕，向前上方送出。",
            "drill": "高低球交替练习",
            "detail": "一球喂高、一球喂平，交替处理；每组 16 球 × 3 组，练习根据来球高度调整击球点。",
        },
    },
    "contact_lateral_abs": {
        "low": {
            "title": "击球点离身体太近",
            "why": "触球时球拍和身体之间空间不足，手臂被挤住，只能靠手腕补救，力量和稳定性都会下降。",
            "fix": "非持拍手向来球方向伸出做“测量”，击球时保持约一臂距离；脚步先到位，再挥拍。",
            "drill": "一臂距离对墙击球",
            "detail": "距墙约 3 米，对墙连续击球，要求每次触球时持拍手肘与身体之间留出一拳以上空间；每组 30 次 × 3 组。",
        },
        "high": {
            "title": "击球点离身体太远",
            "why": "为了够球把手臂伸得过远，身体重心容易前扑，触球控制力下降。",
            "fix": "用更积极的小碎步靠近来球，在身体侧前方舒服的位置击球，而不是伸手够球。",
            "drill": "碎步调整 + 定点击球",
            "detail": "同伴连续喂左右两侧球，要求先用 2–3 步调整再击球；每组 12 球 × 4 组。",
        },
    },
    "contact_forward": {
        "low": {
            "title": "击球点偏后，被球顶住",
            "why": "触球位置在身体侧后方，挥拍空间被压缩，力量无法向前传递，回球容易浅、飘。",
            "fix": "更早完成引拍，击球时向前迎球，把重心从后脚转移到前脚。",
            "drill": "身前标志物击球",
            "detail": "在身体前侧 30–40cm 放一个标志物（或用胶带划线），每次击球都要求在标志物前方触球；每组 20 球 × 3 组。",
        },
        "high": {
            "title": "击球点过前",
            "why": "触球点过于靠前，容易伸手够球、身体前倾，击球稳定性下降。",
            "fix": "稍微等球进入击球区再出手，保持身体重心在双脚之间，不要抢节奏。",
            "drill": "节奏击球练习",
            "detail": "让喂球节奏放慢，口中数“一、二”再击球；每组 15 球 × 3 组，感受球进入击球区再挥拍。",
        },
    },
    "elbow_angle_contact": {
        "low": {
            "title": "触球时肘部太弯",
            "why": "触球瞬间手臂没有充分伸展，击球半径变小，力量和拍头速度都受限。",
            "fix": "触球时让手臂自然伸展到约 150–170°，但不要锁死肘关节；力量来自转体，不是夹臂。",
            "drill": "伸展击球练习",
            "detail": "用较轻的球做半场定点击球，要求触球瞬间手臂伸展、拍头远离身体；每组 20 球 × 3 组。",
        },
        "high": {
            "title": "触球时手臂过直",
            "why": "肘部完全锁死会失去缓冲和拍头控制，触球后也难以自然收拍。",
            "fix": "保持肘部微屈，触球后让前臂自然内旋、完成随挥，不要僵直地“推”球。",
            "drill": "放松挥拍练习",
            "detail": "空挥时保持握拍放松（4/10 力度），感受肘部微屈和自然收拍；每组 20 次 × 3 组。",
        },
    },
    "shoulder_rotation": {
        "low": {
            "title": "转体不足，主要靠手臂发力",
            "why": "肩髋分离角偏小，身体没有充分参与蓄力，击球力量只能来自手臂，长期容易疲劳且不稳定。",
            "fix": "引拍时把非持拍手带到侧前方，让肩膀转到下巴下方；击球时先转髋、再转肩、最后挥臂。",
            "drill": "转体影子挥拍",
            "detail": "双脚分开站立，双手抱胸做转体，感受髋部带动肩部；然后持拍慢速空挥，每组 15 次 × 4 组。",
        },
        "high": {
            "title": "转体幅度过大",
            "why": "肩髋分离角超过合理范围，身体扭转过多，容易失去平衡，触球后恢复也慢。",
            "fix": "控制引拍幅度，保持头部和重心稳定；转体的目的是蓄力，不是越大越好。",
            "drill": "稳定转体击球",
            "detail": "半场定点击球，要求击球后能立即回到准备姿势；每组 15 球 × 3 组，同伴观察头部是否晃动。",
        },
    },
    "stance_width": {
        "low": {
            "title": "准备站位太窄",
            "why": "双脚距离不足，横向移动和击球时的下盘稳定性都会受影响。",
            "fix": "准备姿势双脚略宽于肩，膝盖微屈，重心落在前脚掌，方便向两侧蹬地。",
            "drill": "分腿垫步 + 左右移动",
            "detail": "听口令做分腿垫步，然后向左右各移动 2 步回位；每组 30 秒 × 5 组。",
        },
        "high": {
            "title": "准备站位过宽",
            "why": "站得太宽会限制启动速度，面对快球时第一步反应变慢。",
            "fix": "把站位调整到约 1.3–1.8 倍肩宽，保持膝盖弹性，不要蹲死。",
            "drill": "窄距启动练习",
            "detail": "用较窄站位做左右启动接球，每组 10 次 × 4 组，重点感受第一步蹬地。",
        },
    },
    "knee_bend_min": {
        "low": {
            "title": "屈膝过深",
            "why": "下肢蹲得过低，虽然重心稳，但启动和还原速度会变慢，连续击球时容易疲劳。",
            "fix": "保持适度屈膝和弹性，击球时重心平稳下降，不要为了“蹲低”而蹲死。",
            "drill": "弹性蹲起",
            "detail": "做半蹲后快速起跳的弹性练习，每组 10 次 × 3 组；击球时保持同样的弹性感。",
        },
        "high": {
            "title": "屈膝不足，腿部没参与发力",
            "why": "准备和引拍阶段膝盖太直，下肢无法蓄力，击球力量只能靠上半身，动作会显得“只用手打”。",
            "fix": "引拍时降低重心，膝盖弯曲，击球时从地面蹬起、转髋带动上身。",
            "drill": "蹬地转髋影子挥拍",
            "detail": "慢动作空挥，感受“屈膝—蹬地—转髋—挥拍”的顺序；每组 12 次 × 4 组。",
        },
    },
    "wrist_speed": {
        "low": {
            "title": "挥拍速度偏慢",
            "why": "拍头速度不足，通常说明力量链条不完整或握拍过紧，球速和旋转都会受限。",
            "fix": "握拍保持放松（约 4/10 力度），用转体和蹬地带动手臂，击球瞬间再加速。",
            "drill": "快速空挥",
            "detail": "全力空挥 10 次，再以 70% 力量空挥 10 次，体会放松与加速的区别；共 3 组。",
        },
        "high": {
            "title": "挥拍速度很快，但要注意控制",
            "why": "拍头速度已经不错，但如果触球点或随挥不稳定，快速度会放大失误。",
            "fix": "在保持速度的同时，固定触球点和随挥方向，先追求稳定性再追求极限速度。",
            "drill": "七成力控球",
            "detail": "用 70% 力量做连续对拉，要求连续 10 球不失误；共 3 组。",
        },
    },
    "follow_through_height": {
        "low": {
            "title": "随挥不完整",
            "why": "击球后收拍过早，拍头没有充分向上/向前送出，球容易下网或缺乏深度。",
            "fix": "击球后让球拍自然收到对侧肩膀附近，保持手臂放松，不要急着停下。",
            "drill": "完整随挥空挥",
            "detail": "慢速空挥，要求收拍时球拍到达对侧肩上方；每组 15 次 × 3 组，再用同样感觉击球。",
        },
        "high": {
            "title": "随挥过高",
            "why": "随挥过于向上，可能把球打得太浅，或说明击球时身体后仰。",
            "fix": "根据目标调整随挥方向：打深球时更多向前送，打上旋时再向上收。",
            "drill": "目标区击球",
            "detail": "在对方场地划出深区目标，要求连续 10 球落入目标区；共 3 组。",
        },
    },
    "balance": {
        "low": {
            "title": "重心稳定",
            "why": "击球过程中身体横向晃动很小，下盘稳定性不错。",
            "fix": "继续保持，并在移动中击球时也保持头部稳定。",
            "drill": "移动中击球",
            "detail": "左右移动接球，要求击球瞬间头部不晃动；每组 10 球 × 3 组。",
        },
        "high": {
            "title": "击球时重心晃动较大",
            "why": "髋部相对双脚的横向偏移偏大，击球时身体不够稳定，会影响触球质量。",
            "fix": "击球时保持头部和躯干稳定，重心落在双脚之间；不要为了够球而侧倒。",
            "drill": "平衡垫步击球",
            "detail": "原地分腿垫步后击球，要求击球后保持 1 秒平衡；每组 12 球 × 3 组。",
        },
    },
    "toss_reach": {
        "low": {
            "title": "抛球手伸展不足",
            "why": "抛球手臂没有充分向上伸展，抛球高度和稳定性容易不足，进而影响发球节奏和击球点。",
            "fix": "抛球时手臂从低到高充分伸展，球在手指间自然释放，球到最高点后再挥拍。",
            "drill": "只抛不击",
            "detail": "每天 50 次抛球，不击球，要求球落在身体前侧同一位置；可用胶带标记落点。",
        },
        "high": {
            "title": "抛球手伸展充分",
            "why": "抛球手臂伸展充分，有利于稳定的高点击球。",
            "fix": "保持这个伸展幅度，同时注意抛球落点稳定。",
            "drill": "抛球落点控制",
            "detail": "在地面标记一个 30cm 圆圈，连续抛球 20 次，要求至少 15 次落在圈内。",
        },
    },
    "shoulder_tilt": {
        "low": {
            "title": "发球奖杯姿势肩部倾斜不足",
            "why": "发球蓄力时持拍肩没有明显低于非持拍肩，身体侧倾和肩部绕环的蓄力不足。",
            "fix": "在奖杯姿势中让非持拍肩指向来球方向、持拍肩下沉，身体形成侧弓。",
            "drill": "靠墙奖杯姿势",
            "detail": "背靠墙做奖杯姿势，保持 5 秒，感受非持拍肩高、持拍肩低；每组 5 次 × 4 组。",
        },
        "high": {
            "title": "发球肩部倾斜过大",
            "why": "身体侧倾过多，可能影响向上的蹬伸和击球点稳定性。",
            "fix": "控制侧倾幅度，保持躯干在髋部上方，向上而不是向侧面倒。",
            "drill": "发球慢动作",
            "detail": "用 50% 速度做完整发球动作，重点观察躯干是否向上伸展；每组 10 次 × 3 组。",
        },
    },
    "hip_rise": {
        "low": {
            "title": "发球蹬伸不足",
            "why": "从屈膝蓄力到触球，髋部向上位移偏小，说明腿部力量没有充分传递到发球。",
            "fix": "屈膝蓄力后主动向上蹬伸，击球时身体尽量向上伸展，而不是只靠手臂。",
            "drill": "发球起跳节奏",
            "detail": "做“屈膝—蹬地—向上伸展”的无球练习，每组 8 次 × 4 组，再结合抛球。",
        },
        "high": {
            "title": "发球蹬伸幅度很大",
            "why": "向上蹬伸充分，是发球力量链的优势；注意落地缓冲和平衡。",
            "fix": "保持蹬伸，同时注意落地后快速回位。",
            "drill": "发球落地平衡",
            "detail": "发球后单脚落地保持 1 秒平衡；每组 8 次 × 3 组。",
        },
    },
}

METRIC_PRAISE = {
    "contact_height": "触球位置控制不错，说明你对来球高度和击球时机判断较好。",
    "contact_lateral_abs": "击球点与身体保持了好距离，手臂有空间舒展发力。",
    "contact_forward": "你能在身体前侧击球，力量传递和回球深度基础不错。",
    "elbow_angle_contact": "触球时手臂伸展合理，击球半径和拍头控制较好。",
    "shoulder_rotation": "肩髋分离明显，身体转体参与了蓄力和发力。",
    "stance_width": "准备站位宽度合理，下盘有稳定性。",
    "knee_bend_min": "下肢屈膝蓄力合理，腿部参与了击球发力。",
    "wrist_speed": "挥拍速度不错，拍头加速能力是你的优势。",
    "follow_through_height": "随挥完整，击球后的收拍方向控制较好。",
    "balance": "击球过程中重心稳定，身体晃动控制得好。",
    "toss_reach": "抛球手伸展充分，发球的高点结构基础不错。",
    "shoulder_tilt": "发球奖杯姿势的肩部倾斜合理，蓄力结构不错。",
    "hip_rise": "发球蹬伸充分，下肢力量能传递到击球。",
}


def _encode_image(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((1024, 1024), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _profile_text(profile: dict) -> str:
    if not profile:
        return "未提供球员画像。"
    parts = []
    if profile.get("name"):
        parts.append(f"称呼：{profile['name']}")
    if profile.get("handedness") in ("left", "right"):
        parts.append(f"惯用手：{'左手' if profile['handedness'] == 'left' else '右手'}")
    if profile.get("level"):
        parts.append(f"自评水平：{profile['level']}")
    if profile.get("goal"):
        parts.append(f"本次目标：{profile['goal']}")
    if profile.get("notes"):
        parts.append(f"补充信息：{profile['notes']}")
    return "；".join(parts) if parts else "未提供球员画像。"


def _analysis_text(analysis: dict) -> str:
    lines = [f"【球员】{_profile_text(analysis.get('player', {}))}"]
    lines.append(
        f"【检测结果】共 {analysis.get('stroke_count', 0)} 次击球："
        + "、".join(
            f"{STROKE_LABELS[k]} {v['count']} 次" for k, v in analysis.get("by_type", {}).items()
        )
    )
    level = analysis.get("level")
    if level:
        lines.append(f"【规则评分】整体 {level['score']}/100，估算 NTRP {level['ntrp']}（{level['label']}），置信度 {level['confidence']}")
    for stroke_type, data in analysis.get("by_type", {}).items():
        lines.append(f"\n【{STROKE_LABELS[stroke_type]}】共 {data['count']} 次，技术评分 {data.get('score')}/100")
        for metric in data.get("metrics", []):
            lines.append(
                f"- {metric['label']}: {metric['value']} {metric['unit']}"
                f"（参考 {metric['ideal_low']}–{metric['ideal_high']}，评分 {metric['score']}，{metric['status']}）"
            )
    issues = analysis.get("issues", [])
    if issues:
        lines.append("\n【规则引擎初筛弱项】")
        for issue in issues[:5]:
            lines.append(
                f"- {issue['stroke_label']} · {issue['label']}: 实测 {issue['value']} {issue['unit']}，"
                f"参考 {issue['ideal_low']}–{issue['ideal_high']}，评分 {issue['score']}"
            )
    strengths = analysis.get("strengths", [])
    if strengths:
        lines.append("\n【规则引擎初筛优势】")
        for item in strengths:
            lines.append(f"- {item['stroke_label']} · {item['label']}: {item['value']} {item['unit']}，评分 {item['score']}")
    lines.append(
        "\n【测量说明】数据来自单机位人体姿态估计，未追踪球和球拍；"
        "球速、旋转、落点、比赛结果均未测量。部分指标对机位角度敏感。"
        "如果某种击球样本少于 3 次，请提示可信度有限。"
    )
    return "\n".join(lines)


def _claude_client():
    key = _credentials()
    if not key:
        raise CoachError("未配置 Claude API Key")
    return anthropic.Anthropic(api_key=key)


def _parse_json_response(message) -> dict:
    if getattr(message, "stop_reason", None) == "refusal":
        raise CoachError("Claude 拒绝了本次分析请求")
    text = next((b.text for b in message.content if getattr(b, "type", None) == "text"), None)
    if not text:
        raise CoachError("Claude 未返回文本结果")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CoachError(f"Claude 返回结果不是合法 JSON: {exc}") from exc


def _claude_report(analysis: dict, keyframes: list[dict]) -> dict:
    client = _claude_client()
    content: list[dict] = []
    for index, keyframe in enumerate(keyframes[:6], start=1):
        path = keyframe.get("path")
        if not path or not Path(path).exists():
            continue
        content.append({"type": "text", "text": f"【图{index}】{keyframe.get('label', '')}"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _encode_image(Path(path)),
                },
            }
        )
    instruction = f"""
请根据下面的姿态指标和关键帧，给这位业余球员一份训练反馈。

{_analysis_text(analysis)}

【任务】
1. 指出 2–3 个最值得优先改进的问题，按重要性排序；每个问题写明：观察到的数据/画面、为什么重要、具体怎么改、一个可执行的练习。
2. 指出 1–3 个做得好的地方。
3. 制定一个 2 周训练计划：每周 3 次、每次 40–60 分钟，包含热身、主练习和收尾；每次训练要具体到练习名称、组数/次数和注意事项。
4. 给出 2–4 条 2 周后的验收指标，要能用视频或数据检验。
5. 语气专业、具体、鼓励，不要空泛。引用最关键的数字，但不要把所有数字重复一遍。
6. 明确说明：这是基于单机位姿态分析的训练建议，不能替代现场教练和医学诊断。
"""
    content.append({"type": "text", "text": instruction})
    try:
        with client.messages.stream(
            model=LLM_MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_config={
                "effort": LLM_EFFORT,
                "format": {"type": "json_schema", "schema": REPORT_SCHEMA},
            },
        ) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise CoachError("Claude API 认证失败，请检查 API Key") from exc
    except anthropic.RateLimitError as exc:
        raise CoachError("Claude API 触发限流，请稍后重试") from exc
    except anthropic.APIStatusError as exc:
        raise CoachError(f"Claude API 返回错误（{exc.status_code}）") from exc
    except anthropic.APIConnectionError as exc:
        raise CoachError("无法连接 Claude API，请检查网络") from exc
    return _parse_json_response(message)


def _fmt(value, unit: str) -> str:
    if isinstance(value, float):
        text = f"{value:.1f}"
    else:
        text = str(value)
    return f"{text} {unit}".strip()


def _issue_from_candidate(candidate: dict, priority: int) -> dict:
    templates = COACHING.get(candidate["key"])
    if not templates:
        templates = {
            "low": {
                "title": f"{candidate['label']}偏低",
                "why": "该指标低于参考区间，可能影响动作效率和稳定性。",
                "fix": "先放慢动作，专注把这一环节做完整，再逐步加力。",
                "drill": "慢动作分解练习",
                "detail": "把动作拆成 3 个节拍慢速完成，每组 10 次 × 3 组。",
            },
            "high": {
                "title": f"{candidate['label']}偏高",
                "why": "该指标高于参考区间，可能带来控制或平衡问题。",
                "fix": "降低幅度，优先保证触球稳定和身体平衡。",
                "drill": "七成力控制练习",
                "detail": "用 70% 力量完成动作，每组 12 次 × 3 组。",
            },
        }
    template = templates.get(candidate["direction"], templates["low"])
    severity = "high" if candidate["score"] < 35 else "medium" if candidate["score"] < 55 else "low"
    return {
        "priority": priority,
        "title": template["title"],
        "stroke": candidate["stroke_label"],
        "severity": severity,
        "observation": (
            f"你的{candidate['stroke_label']}{candidate['label']}平均为 "
            f"{_fmt(candidate['value'], candidate['unit'])}，参考区间 "
            f"{candidate['ideal_low']}–{candidate['ideal_high']} {candidate['unit']}，"
            f"规则评分 {candidate['score']}/100（{candidate['samples']} 次样本）。"
        ),
        "why": template["why"],
        "fix": template["fix"],
        "drill": f"{template['drill']}：{template['detail']}",
        "_drill_name": template["drill"],
        "_label": candidate["label"],
        "_ideal_low": candidate["ideal_low"],
        "_ideal_high": candidate["ideal_high"],
        "_unit": candidate["unit"],
    }


def _local_plan(analysis: dict, issues: list[dict]) -> dict:
    drills = []
    for issue in issues[:3]:
        drills.append(
            {
                "name": issue.get("_drill_name", "针对性技术练习"),
                "sets": "3 组",
                "notes": issue["drill"].split("：", 1)[-1],
            }
        )
    if not drills:
        drills = [
            {"name": "定点击球", "sets": "20 球 × 3 组", "notes": "固定触球点，追求连续命中。"},
            {"name": "分腿垫步 + 左右移动", "sets": "30 秒 × 5 组", "notes": "保持低重心和第一步启动。"},
            {"name": "完整随挥空挥", "sets": "15 次 × 3 组", "notes": "收拍到对侧肩，动作放松。"},
        ]

    def session(day, title, selected, duration=50):
        out = [{"name": "动态热身", "sets": "8 分钟", "notes": "肩、髋、踝活动开，做 10 次分腿垫步。"}]
        out.extend(selected)
        out.append({"name": "收尾放松", "sets": "5 分钟", "notes": "拉伸肩背和前臂，回顾今天的一个重点。"})
        return {"day": day, "duration_min": duration, "title": title, "drills": out}

    first = [drills[0]] if drills else []
    second = [drills[1]] if len(drills) > 1 else [drills[0]]
    weeks = [
        {
            "week": 1,
            "theme": "技术纠正：先把动作框架做对",
            "sessions": [
                session("第 1 次", "重点练习 1", first + [
                    {"name": "慢速影子挥拍", "sets": "12 次 × 3 组", "notes": "按“蹬地—转髋—挥拍—随挥”的节拍做。"}
                ]),
                session("第 2 次", "重点练习 2", second + [
                    {"name": "半场对拉", "sets": "10 分钟", "notes": "只用 70% 力量，优先连续性和触球点。"}
                ]),
                session("第 3 次", "巩固与实战", first + second + [
                    {"name": "计分对打", "sets": "15 分钟", "notes": "只关注今天的一个技术点，不追求制胜分。"}
                ], duration=60),
            ],
        },
        {
            "week": 2,
            "theme": "巩固与实战：把正确动作放进对抗",
            "sessions": [
                session("第 1 次", "压力下的技术", first + [
                    {"name": "左右移动击球", "sets": "10 球 × 3 组", "notes": "移动中保持触球点和平衡。"}
                ]),
                session("第 2 次", "组合练习", second + [
                    {"name": "斜线/直线组合", "sets": "10 分钟", "notes": "先打斜线再变直线，保持完整随挥。"}
                ]),
                session("第 3 次", "实战检验 + 重拍视频", first + second + [
                    {"name": "计分对打", "sets": "15 分钟", "notes": "结束后按同样机位再拍 5–10 个球，和本次视频对比。"}
                ], duration=60),
            ],
        },
    ]

    success = []
    for issue in issues[:3]:
        success.append(
            f"{issue['stroke']}{issue['_label']}进入 {issue['_ideal_low']}–{issue['_ideal_high']} {issue['_unit']} 区间"
        )
    if not success:
        success = ["连续 10 次击球中至少 7 次触球点稳定", "2 周后重拍视频，整体评分提高 5 分以上"]
    success.append("训练后主观疲劳可控，肩肘无疼痛；如有疼痛请停止并咨询专业人士")
    return {"focus": "；".join(i["title"] for i in issues[:3]) or "稳定动作框架，提高连续击球质量", "weeks": weeks, "success_metrics": success}


def local_report(analysis: dict, reason: str | None = None) -> dict:
    issues = [_issue_from_candidate(c, i + 1) for i, c in enumerate(analysis.get("issues", [])[:3])]
    strengths = []
    for item in analysis.get("strengths", [])[:3]:
        strengths.append(
            {
                "title": f"{item['stroke_label']} · {item['label']}",
                "detail": f"{METRIC_PRAISE.get(item['key'], '这项指标在合理区间，继续保持。')} 本次平均 {_fmt(item['value'], item['unit'])}。",
            }
        )
    if not strengths:
        strengths.append(
            {
                "title": "完成了一次可分析的训练记录",
                "detail": "有了基线视频，后面的进步就可以被看见。先把注意力放在一个最想改的点上。",
            }
        )

    level = analysis.get("level") or {}
    by_type = analysis.get("by_type", {})
    stroke_bits = "、".join(f"{STROKE_LABELS[k]} {v['count']} 次" for k, v in by_type.items())
    score_text = f"整体技术评分 {level.get('score', '—')}/100" if level else "样本量较少，暂不给出整体级别"
    headline = issues[0]["title"] if issues else "动作框架不错，下一步提升稳定性"
    level_assessment = (
        f"根据本次 {analysis.get('stroke_count', 0)} 次击球（{stroke_bits}）的姿态指标，"
        f"估算水平约为 NTRP {level.get('ntrp', '—')}（{level.get('label', '样本不足')}）。"
        if level else "本次可分析的击球样本较少，建议每种击球至少录 3–5 次后再评估水平。"
    )
    overall = (
        f"{score_text}。{level_assessment}"
        "本次分析基于单机位姿态估计，重点看身体结构和击球位置；球速、旋转、落点等没有测量。"
    )
    encouragement = (
        "你的动作已经有可分析的结构，这本身就是很好的起点。训练时一次只盯一个点，"
        "用视频记录变化，比一次性改很多地方更有效。"
        if not issues else
        f"先集中解决「{issues[0]['title']}」这一个点，其他问题会跟着变好。两周后再拍一次同机位视频，我们对比数据看进步。"
    )
    plan = _local_plan(analysis, issues)
    report = {
        "headline": headline,
        "level_assessment": level_assessment,
        "overall": overall,
        "strengths": strengths,
        "issues": [{k: v for k, v in issue.items() if not k.startswith("_")} for issue in issues],
        "training_plan": plan,
        "success_metrics": plan["success_metrics"],
        "encouragement": encouragement,
    }
    report["meta"] = {
        "provider": "local",
        "model": None,
        "reason": reason or "未配置 Claude API Key，使用本地规则教练生成反馈。",
    }
    return report


def generate_feedback(analysis: dict, keyframes: list[dict]) -> dict:
    status = llm_status()
    if not status["ready"]:
        return local_report(analysis, status["reason"])
    try:
        report = _claude_report(analysis, keyframes)
        report["meta"] = {"provider": "claude", "model": LLM_MODEL, "reason": None}
        return report
    except CoachError as exc:
        return local_report(analysis, str(exc))
    except Exception as exc:  # boundary: never lose the whole analysis because of the LLM layer
        return local_report(analysis, f"Claude 调用失败：{type(exc).__name__}: {exc}")


def chat_reply(analysis: dict, history: list[dict], question: str) -> dict:
    status = llm_status()
    if not status["ready"]:
        return {
            "reply": "当前是本地规则模式，未配置 Claude API Key，所以还不能进行自由问答。"
            "你仍然可以查看上面的指标、反馈和训练计划；配置 Key 后再来分析页就能追问细节。",
            "provider": "local",
        }
    client = _claude_client()
    context = _analysis_text(analysis)
    system = (
        SYSTEM_PROMPT
        + "\n你正在和球员进行训练后的追问对话。回答要简短、具体，优先引用分析数据；"
        "如果问题超出视频能判断的范围（例如球速、落点、伤病），要直说无法判断。"
        f"\n\n【本次分析上下文】\n{context}"
    )
    messages = []
    for item in history[-12:]:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=4000,
            system=system,
            messages=messages,
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        return {"reply": text or "我没有生成有效回答，请换个问法。", "provider": "claude"}
    except Exception as exc:
        return {"reply": f"Claude 暂时不可用：{type(exc).__name__}。请稍后再试。", "provider": "error"}
