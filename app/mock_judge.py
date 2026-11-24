"""模拟语义判定：不调用模型，但按业务话术模拟真实判定的命中、漏判和误判。"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, Tuple


INSULTS = ("傻", "蠢", "笨", "闭嘴", "滚", "神经病", "废物", "脑子", "没脑子")
GUARANTEE = ("保本", "稳赚", "保证收益", "包赚", "零风险", "一定赚")
CODE_WORDS = ("验证码", "动态密码")
THREAT_WORDS = ("上门", "起诉", "后果自负", "单位", "领导", "亲友", "曝光")


def _agent_text(transcript: str) -> str:
    lines = []
    for line in str(transcript or "").splitlines():
        matched = re.search(r"(坐席|客服|专员|agent)\s*[：:]\s*(.*)$", line, flags=re.I)
        if matched:
            lines.append(matched.group(2).strip())
    return "\n".join(lines)


def _sentence(text: str, words) -> str:
    for line in text.splitlines():
        for word in words:
            if word in line:
                return line.strip()[:160]
    return ""


def _hit(code: str, name: str, semantic: str, agent_text: str, logic: str = "OR") -> Tuple[bool, str]:
    text = agent_text or ""
    if code == "R001" or "辱骂" in name:
        if any(word in text for word in INSULTS):
            return True, f"坐席使用贬低性表达「{_sentence(text, INSULTS)}」"
        return False, "坐席发言未出现侮辱、贬低或人身攻击表达"
    if code == "R002" or "保本" in name or "收益" in name:
        if any(word in text for word in ("不保本", "不保证", "风险提示", "不承诺")):
            return False, "坐席是在提示风险或不承诺收益，不构成保本承诺"
        if any(word in text for word in GUARANTEE):
            return True, f"坐席向客户承诺「{_sentence(text, GUARANTEE)}」"
        return False, "坐席未承诺保本、保证收益或零风险"
    if code == "R003" or "验证码" in name:
        exclusions = [line for line in str(semantic or "").splitlines()
                      if line.strip().startswith("排除：")]
        for line in exclusions:
            for term in ("不要", "不会", "切勿", "提醒"):
                if term in line and term in text:
                    return False, f"命中语义排除条件「{term}」，不构成索要"
        if logic == "AND":
            # 演示基线：语义模型过度触发，把“提醒验证码”也当成索要；后续靠排除条件逐步修正。
            if any(word in text for word in CODE_WORDS):
                return True, f"当前语义把「{_sentence(text, CODE_WORDS)}」识别为索要"
            return False, "坐席发言未出现验证码或动态密码"
        if any(word in text for word in ("不要", "不会", "切勿", "提醒")) and any(
            word in text for word in CODE_WORDS
        ):
            return False, "坐席是在提醒客户不要泄露验证码，不是索要"
        # 模拟模型的一个真实弱点：只记住“验证码”，漏掉“动态密码”同义表达。
        if "动态密码" in text and "验证码" not in text:
            return False, "未识别“动态密码”同义表达，模拟漏判"
        if any(word in text for word in CODE_WORDS) and any(
            word in text for word in ("报", "告诉", "发我", "念", "给我", "多少")
        ):
            return True, f"坐席向客户索要「{_sentence(text, CODE_WORDS)}」"
        return False, "坐席未主动索要短信验证码或动态密码"
    if code == "R004" or "威胁" in name or "催收" in name:
        if any(word in text for word in ("不会", "不", "别")) and any(
            word in text for word in THREAT_WORDS
        ):
            return False, "坐席是在说明不会采取威胁方式，不构成威胁"
        # 模拟模型对“去单位找领导”这类没有“上门”字样的威胁漏判。
        if "单位" in text and "上门" not in text:
            return False, "只按“上门”识别，漏掉“去单位找领导”的威胁"
        if any(word in text for word in THREAT_WORDS):
            return True, f"坐席以「{_sentence(text, THREAT_WORDS)}」施压"
        return False, "坐席未以起诉、上门、曝光或联系单位/亲友等方式威胁"
    keywords = [word for word in re.findall(r"[\u4e00-\u9fa5]{2,6}", semantic or "")][:8]
    if keywords and any(word in text for word in keywords):
        return True, f"坐席发言包含语义条件「{next(word for word in keywords if word in text)}」"
    return False, "坐席发言未满足当前语义条件"


async def mock_judge(case: Dict[str, Any], rule: Dict[str, Any], with_reason: bool = True):
    if rule.get("workflow") == "COMPLAINT_RESPONSE":
        return {
            "violation": None,
            "reason": "投诉回应需要核验客户诉求与坐席回应窗口，模拟模式转人工复核",
            "reviewRequired": True,
            "evidenceValid": False,
        }
    await asyncio.sleep(0.06)
    hit, reason = _hit(str(rule.get("code") or ""), str(rule.get("name") or ""),
                       str(rule.get("semantic") or ""), _agent_text(str(case.get("transcript") or "")),
                       str(rule.get("logic") or "OR"))
    return hit, reason if with_reason else ""


__all__ = ["mock_judge"]
