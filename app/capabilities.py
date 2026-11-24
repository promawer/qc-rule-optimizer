"""可核验能力执行契约：角色、原文话轮、投诉回应窗口及三态结论。

这里只校验外显证据与业务检查结果，不索取或保存模型的内部推理过程。
"""
from __future__ import annotations

import re
from .regex_safety import safe_search
from typing import Any, Dict, List, Optional

MAX_CAPABILITY_CHARS = 20000
_DATE_TIME = r"\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}(?::\d{2})?"
_TIME = r"(?:\[\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*\]|" + _DATE_TIME + r"|\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)"
_TURN = re.compile(
    r"^[ \t]*[（(]?(?:" + _TIME + r"[ \t]*)?"
    r"(?P<label>客户经理|客服人员|坐席|客服|客户|用户|agent|customer)"
    r"[ \t]*(?:" + _TIME + r"[ \t]*)?[:：][ \t]*",
    re.MULTILINE | re.IGNORECASE,
)
_ROLE = {"客户": "CUSTOMER", "用户": "CUSTOMER", "customer": "CUSTOMER",
         "坐席": "AGENT", "客服": "AGENT", "客服人员": "AGENT", "客户经理": "AGENT", "agent": "AGENT"}


def parse_turns(transcript: str) -> List[Dict[str, Any]]:
    """解析完整话轮，偏移是原始文本的字符偏移，sourceLine 是一基行号。"""
    text = str(transcript or "")
    matches = list(_TURN.finditer(text))
    turns = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        while end > start and text[end - 1].isspace():
            end -= 1
        turns.append({
            "id": "T%d" % (index + 1), "turnId": "T%d" % (index + 1),
            "role": _ROLE[match.group("label").lower()], "text": text[start:end],
            "start": start, "end": end, "sourceStart": match.start(),
            "sourceLine": text.count("\n", 0, match.start()) + 1,
        })
    return turns


def parse_transcript(transcript: str) -> Dict[str, Any]:
    """不从长文本挑片段拼接流程；截断、角色缺失均显式报告。"""
    text = str(transcript or "")
    turns = parse_turns(text)
    errors = []
    if not turns:
        errors.append("转写没有可识别的客户/坐席话轮")
    elif text[:turns[0]["sourceStart"]].strip() not in ("", "（", "("):
        errors.append("转写开头存在无法归属角色的内容")
    for turn in turns:
        if not turn["text"].strip():
            errors.append("%s 话轮内容为空" % turn["turnId"])
        # 不把混在延续行中的未知说话人或内联角色悄悄归属上一位说话人。
        if re.search(r"(?:^|\n)[ \t]*(?:说话人\s*\d+|speaker\s*\d+|未知|其他)[ \t]*[:：]", turn["text"], re.I):
            errors.append("%s 含有未识别的说话人" % turn["turnId"])
        if re.search(r"(?:客户|坐席|客服)[ \t]*[:：]", turn["text"]):
            errors.append("%s 含有未分行的角色标记，请按话轮换行" % turn["turnId"])
    if len(text) > MAX_CAPABILITY_CHARS:
        errors.append("转写超过完整流程评估长度上限，不能使用截断上下文作确定判定")
    return {"turns": turns, "errors": errors, "valid": not errors,
            "truncated": len(text) > MAX_CAPABILITY_CHARS}


def role_context_valid(rule: Dict[str, Any], transcript: str) -> bool:
    if rule.get("workflow") == "COMPLAINT_RESPONSE":
        return bool(parse_transcript(transcript)["valid"])
    if "actor" not in rule or rule.get("actor") == "BOTH":
        return bool(str(transcript or "").strip())
    parsed = parse_transcript(transcript)
    return bool(parsed["valid"] and any(turn["role"] == rule.get("actor") for turn in parsed["turns"]))


def regex_evidence(rule: Dict[str, Any], transcript: str) -> Optional[Dict[str, Any]]:
    """正则只在指定角色的单个话轮内匹配，返回原始文本偏移。"""
    pattern = str(rule.get("regex") or "")
    if not pattern:
        return None
    text = str(transcript or "")
    actor = rule.get("actor")
    if not actor:  # 保留未迁移的旧调用方契约；新规则均显式设置 actor。
        match = safe_search(pattern, text)
        return {"start": match.start(), "end": match.end(), "text": match.group(0),
                "turnId": None, "role": None} if match else None
    parsed = parse_transcript(text)
    if not parsed["valid"]:
        if actor != "BOTH":
            return None
        match = safe_search(pattern, text)
        return {"start": match.start(), "end": match.end(), "text": match.group(0),
                "turnId": None, "role": None} if match else None
    for turn in parsed["turns"]:
        if actor != "BOTH" and turn["role"] != actor:
            continue
        match = safe_search(pattern, turn["text"])
        if match:
            return {"start": turn["start"] + match.start(), "end": turn["start"] + match.end(),
                    "text": match.group(0), "turnId": turn["turnId"], "role": turn["role"]}
    return None


def unknown_result(reason: str, **extra: Any) -> Dict[str, Any]:
    result = {"violation": None, "reason": reason, "evidence": [],
              "steps": {"complaintIntent": None, "adequateResponse": None, "timelyResponse": None},
              "reviewRequired": True, "evidenceValid": False, "evidenceErrors": [reason]}
    result.update(extra)
    result["violation"] = None
    result["reviewRequired"] = True
    return result


def tri_state(value: Any) -> Optional[bool]:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "yes", "1", "是", "违规", "命中"):
            return True
        if text in ("false", "no", "0", "否", "正常", "未命中"):
            return False
        if text in ("", "null", "unknown", "无法判断", "不确定", "待复核"):
            return None
    raise ValueError("检查结果必须为 true、false 或 null")


def approved_knowledge(rule: Dict[str, Any]) -> List[Dict[str, Any]]:
    """只接收服务端解析的已批准知识；knowledgeIds 不会自行变成知识内容。"""
    return [{key: item.get(key) for key in ("id", "version", "title", "content", "scope", "exceptions")}
            for item in rule.get("_approvedKnowledge", []) if isinstance(item, dict)
            and str(item.get("status", "APPROVED")).upper() == "APPROVED"]


def validate_capability_result(data: Dict[str, Any], case: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
    """由程序核对证据和时序，并从检查结果计算结论；不相信模型自报 evidenceValid。"""
    parsed = parse_transcript(str(case.get("transcript") or ""))
    turns = parsed["turns"]
    errors = list(parsed["errors"])
    if case.get("transcriptTruncated") or case.get("truncated"):
        errors.append("输入转写已标记为截断")
    by_id = {turn["turnId"]: turn for turn in turns}
    positions = {turn["turnId"]: index for index, turn in enumerate(turns)}
    evidence = []
    raw_evidence = data.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        errors.append("缺少可核验的原文证据")
        raw_evidence = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            errors.append("证据必须为话轮对象")
            continue
        turn_id, quote, role = str(item.get("turnId") or ""), str(item.get("quote") or ""), str(item.get("role") or "")
        turn = by_id.get(turn_id)
        if not turn:
            errors.append("证据引用不存在的话轮 %s" % turn_id)
        elif role != turn["role"]:
            errors.append("%s 证据角色与原文不一致" % turn_id)
        elif not quote.strip() or quote not in turn["text"]:
            errors.append("%s 引用不是该话轮的原文片段" % turn_id)
        else:
            offset = turn["text"].index(quote)
            evidence.append({"turnId": turn_id, "quote": quote, "role": role,
                             "start": turn["start"] + offset, "end": turn["start"] + offset + len(quote),
                             "sourceLine": turn["sourceLine"]})
    raw_steps = data.get("steps") if isinstance(data.get("steps"), dict) else {}
    steps = {}
    for name in ("complaintIntent", "adequateResponse", "timelyResponse"):
        try:
            steps[name] = tri_state(raw_steps.get(name))
        except ValueError:
            steps[name] = None
            errors.append("%s 不是有效的三态检查结果" % name)
    customer_evidence = [item for item in evidence if item["role"] == "CUSTOMER"]
    agent_evidence = [item for item in evidence if item["role"] == "AGENT"]
    if not customer_evidence:
        errors.append("投诉意图判断必须引用客户原文")
    trigger_id = str(data.get("triggerTurnId") or (customer_evidence[0]["turnId"] if customer_evidence else ""))
    trigger = by_id.get(trigger_id)
    response_id = str(data.get("responseTurnId") or (agent_evidence[0]["turnId"] if agent_evidence else ""))
    response_window = {"turnIds": [], "complete": False}
    violation = None
    if steps["complaintIntent"] is False:
        violation = False
        if steps["adequateResponse"] is not None or steps["timelyResponse"] is not None:
            errors.append("未识别投诉时回应检查应为 null（不适用）")
    elif steps["complaintIntent"] is True:
        if not trigger or trigger["role"] != "CUSTOMER" or trigger_id not in {item["turnId"] for item in customer_evidence}:
            errors.append("投诉触发话轮必须与客户证据对应")
        else:
            index = positions[trigger_id]
            window = max(1, min(50, int(rule.get("responseWindowTurns") or 3)))
            following = turns[index + 1:index + window + 1]
            complete = len(following) >= window or case.get("transcriptComplete") is True
            response_window = {"turnIds": [turn["turnId"] for turn in following], "complete": complete,
                               "limit": window, "triggerTurnId": trigger_id}
            if steps["adequateResponse"] is True:
                response = by_id.get(response_id)
                if not response or response["role"] != "AGENT" or response_id not in {item["turnId"] for item in agent_evidence}:
                    errors.append("适当回应必须引用坐席回应原文")
                elif positions[response_id] <= index:
                    errors.append("坐席回应发生在投诉之前，不能作为本次投诉的回应")
                else:
                    timely = positions[response_id] <= index + window
                    if steps["timelyResponse"] is not timely:
                        errors.append("及时回应检查与原文话轮顺序不一致")
                    violation = not timely
            elif steps["adequateResponse"] is False:
                if not complete:
                    errors.append("尚未观察到完整回应窗口，不能据缺失后续对话判违规")
                observed_agents = {turn["turnId"] for turn in following if turn["role"] == "AGENT"}
                if observed_agents and not observed_agents.intersection(item["turnId"] for item in agent_evidence):
                    errors.append("回应窗口存在坐席发言，需引用其原文支持回应不足的判断")
                if steps["timelyResponse"] is not False:
                    errors.append("不存在适当回应时 timelyResponse 应为 false")
                violation = True
            else:
                errors.append("坐席回应是否适当尚不确定")
    else:
        errors.append("客户投诉意图尚不确定")
    try:
        model_verdict = tri_state(data.get("violation"))
    except ValueError:
        model_verdict = None
        errors.append("模型结论不是有效的三态值")
    if model_verdict is None:
        errors.append("模型结论为待复核")
    elif violation is not None and model_verdict is not violation:
        errors.append("最终结论与投诉/回应检查结果不一致")
    if data.get("reviewRequired") is True:
        errors.append("模型请求人工复核")
    result = {"violation": violation, "reason": str(data.get("reason") or "")[:500],
              "evidence": evidence, "steps": steps, "triggerTurnId": trigger_id or None,
              "responseTurnId": response_id or None, "responseWindow": response_window,
              "reviewRequired": bool(errors), "evidenceValid": not errors,
              "evidenceErrors": list(dict.fromkeys(errors))}
    if errors:
        result["violation"] = None
        result["reason"] = "待复核：" + "；".join(result["evidenceErrors"])
    return result
