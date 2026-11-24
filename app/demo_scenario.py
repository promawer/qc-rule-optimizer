"""40 通验证码演示场景：基线约 40%，可通过三轮语义排除逐步提升到目标。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

BASE = datetime(2026, 11, 20, 9, 0, tzinfo=timezone(timedelta(hours=8)))
RULE_IDS = {"R001": "rule-insult", "R002": "rule-guarantee", "R003": "rule-code", "R004": "rule-threat"}

POSITIVE = [
    "为了核实身份，请把收到的验证码告诉我。",
    "您把动态密码念一下，我帮您登记。",
    "请把短信验证码报给我。",
    "您把手机收到的六位数字发我。",
    "把验证码给我，我这边帮您处理。",
    "麻烦您把收到的验证码发给我核对。",
]
NORMAL_DONT = [
    "请不要把验证码给任何人。",
    "验证码不要报给别人。",
    "您不要把短信验证码发给我。",
    "请客户自己保管验证码，不要提供。",
    "验证码不要念出来。",
    "您收到的验证码不要告诉坐席。",
    "千万不要把验证码给其他人。",
]
NORMAL_WONT = [
    "我们不会向您索要验证码。",
    "不会让您提供验证码。",
    "我们不会要求客户给验证码。",
    "您放心，我们不会问验证码。",
    "不会让您念动态密码。",
    "我们不会向客户索要动态密码。",
    "工作人员不会问您要验证码。",
    "不会要求您把验证码发过来。",
]
NORMAL_NEVER = [
    "请切勿泄露验证码。",
    "提醒您，动态密码要自己保管。",
    "切勿把验证码告诉任何人。",
    "请客户注意，验证码千万不要外泄。",
    "提醒一下，动态密码不能提供。",
    "切勿向任何人透露短信验证码。",
    "请您自己操作，切勿把验证码给别人。",
    "提醒客户保护好验证码。",
    "切勿把动态密码告诉其他人。",
    "提醒您，短信验证码不要外泄。",
]
NORMAL_PLAIN = [
    "您好，请问是张先生吗？",
    "您的账单已经逾期三天了，请按时还款。",
    "我帮您核对一下身份信息。",
    "您反馈的问题我已经记录。",
    "稍后会有专员与您联系。",
    "请您通过官方渠道办理。",
    "您的业务已经受理。",
    "感谢您的来电。",
    "请您注意接听回访电话。",
]


def _stamp(index: int) -> str:
    value = BASE + timedelta(minutes=index * 7)
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _call(sequence: int, agent_text: str, positive: bool) -> Dict[str, Any]:
    call_id = f"D{sequence:03d}"
    transcript = f"{_stamp(sequence)} 客户：您好，我想咨询一下。\n{_stamp(sequence)} 坐席：{agent_text}\n"
    other_positive = {
        "R001": {1, 9, 17, 25, 33},
        "R002": {2, 10, 18, 26, 34},
        "R004": {3, 11, 19, 27, 35},
    }
    human_codes = []
    annotations = {}
    for code, rule_id in RULE_IDS.items():
        label = "NORMAL"
        if code == "R003" and positive:
            label = "VIOLATION"
        elif code in other_positive and sequence in other_positive[code]:
            label = "VIOLATION"
        annotations[rule_id] = {
            "label": label,
            "reviewStatus": "REVIEWED",
            "source": "demo",
            "reason": "演示场景人工复核",
        }
        if label == "VIOLATION":
            human_codes.append(code)
    return {
        "id": "call-" + call_id.lower(),
        "callId": call_id,
        "groupId": {11: "demo-val-3", 19: "demo-val-2", 20: "demo-val-11", 25: "demo-val-12"}.get(sequence, ""),
        "autoRules": ["R003"] if positive else [],
        "humanRules": human_codes,
        "annotations": annotations,
        "transcript": transcript,
        "importedAt": _stamp(sequence),
    }


def build_demo_calls() -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    sequence = 0
    for text in POSITIVE:
        sequence += 1
        calls.append(_call(sequence, text, True))
    for group in (NORMAL_DONT, NORMAL_WONT, NORMAL_NEVER, NORMAL_PLAIN):
        for text in group:
            sequence += 1
            forced_positive = sequence == 19
            if forced_positive:
                text = "请把验证码报给我，我帮您核对。（第1通）"
            if sequence == 25:
                text = "请切勿泄露验证码。"
            calls.append(_call(sequence, text, forced_positive))
    return calls


__all__ = ["build_demo_calls"]
