"""内置演示规则与合成通话数据（与 samples/import-calls-10.csv 一致）。

数据口径：
- 一行一通电话，通话文本带时间戳和坐席/客户角色；
- 第三列是自动质检判定的违规规则编号，空表示未判违规；
- 第四列是人工复核的违规规则编号，空表示人工认为没有违规；
- 一通电话可以同时命中多条规则。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .llm import DEFAULT_HEADERS, DEFAULT_REQUEST_TEMPLATE
from .prompts import DEFAULT_SEMANTIC_PROMPT
from .splits import build_splits


def _iso(minutes: int = 0) -> str:
    base = datetime(2026, 11, 1, 21, 0, tzinfo=timezone(timedelta(hours=8)))
    return (base + timedelta(minutes=minutes)).isoformat(timespec="seconds")


RULES: List[Dict[str, Any]] = [
    {
        "id": "rule-insult",
        "code": "R001",
        "name": "辱骂客户",
        "description": "坐席在与客户沟通时使用侮辱、贬低、攻击性语言。",
        "regex": r"(你(这个|真是)?(傻|蠢|笨)|闭嘴|滚(蛋|出去)|神经病|废物)",
        "semantic": "坐席是否使用侮辱、贬低或人身攻击性语言辱骂客户；正常的催促、拒绝办理不算。",
        "logic": "OR",
    },
    {
        "id": "rule-guarantee",
        "code": "R002",
        "name": "承诺保本收益",
        "description": "销售或客服在介绍产品时承诺保本、保证收益、零风险。",
        "regex": r"(保本|稳赚|保证收益|包赚|零风险|一定赚)",
        "semantic": "坐席是否向客户承诺本金安全、保证收益或零风险；仅解释产品风险提示不算。",
        "logic": "OR",
    },
    {
        "id": "rule-code",
        "code": "R003",
        "name": "索要短信验证码",
        "description": "坐席以核实身份、代办业务等理由索要客户收到的短信验证码。",
        "regex": r"验证码",
        "semantic": "坐席是否主动索要客户手机收到的短信验证码或动态密码；提醒客户不要泄露验证码不算。",
        "logic": "OR",
    },
    {
        "id": "rule-threat",
        "code": "R004",
        "name": "威胁催收",
        "description": "催收过程中以起诉、上门、联系单位/亲友等方式威胁客户。",
        "regex": r"(上门(催收|找你)|起诉你|后果自负|不还钱就)",
        "semantic": "坐席是否以起诉、上门、曝光、联系单位或亲友等方式威胁客户还款；正常告知逾期后果不算。",
        "logic": "OR",
    },
]


CALLS: List[Dict[str, Any]] = [
    {
        "id": "call-c001",
        "callId": "C001",
        "autoRules": ["R004"],
        "humanRules": ["R004"],
        "transcript": """（2026-11-01 21:00:01 坐席：您好，请问是张先生吗？
2026-11-01 21:00:04 客户：是的，什么事？
2026-11-01 21:00:08 坐席：您的账单已经逾期三天了，今天不还钱就上门催收。
2026-11-01 21:00:12 客户：我这两天就还。
）""",
    },
    {
        "id": "call-c002",
        "callId": "C002",
        "autoRules": ["R003"],
        "humanRules": ["R003"],
        "transcript": """（2026-11-01 21:05:10 坐席：为了核实您的身份，请把收到的验证码告诉我。
2026-11-01 21:05:14 客户：是 123456 吗？
2026-11-01 21:05:16 坐席：对，报给我就行。
）""",
    },
    {
        "id": "call-c003",
        "callId": "C003",
        "autoRules": [],
        "humanRules": ["R001"],
        "transcript": """（2026-11-01 21:10:02 坐席：您这个操作有问题，就你这脑子还是别办了。
2026-11-01 21:10:06 客户：你怎么说话的？
2026-11-01 21:10:09 坐席：我说的是实话。
）""",
    },
    {
        "id": "call-c004",
        "callId": "C004",
        "autoRules": ["R002"],
        "humanRules": ["R002"],
        "transcript": """（2026-11-01 21:15:20 坐席：这款产品是保本的，您放心买。
2026-11-01 21:15:24 客户：真的不会亏吗？
2026-11-01 21:15:27 坐席：保本保息，零风险。
）""",
    },
    {
        "id": "call-c005",
        "callId": "C005",
        "autoRules": ["R003"],
        "humanRules": [],
        "transcript": """（2026-11-01 21:20:31 坐席：提醒您一下，千万不要把验证码告诉任何人。
2026-11-01 21:20:35 客户：好的，我知道了。
2026-11-01 21:20:37 坐席：我们不会向您索要验证码。
）""",
    },
    {
        "id": "call-c006",
        "callId": "C006",
        "autoRules": [],
        "humanRules": ["R003"],
        "transcript": """（2026-11-01 21:25:11 坐席：您把短信里的动态密码念一下，我帮您登记。
2026-11-01 21:25:15 客户：动态密码是多少？
2026-11-01 21:25:18 坐席：您手机上刚收到的那六位数字。
）""",
    },
    {
        "id": "call-c007",
        "callId": "C007",
        "autoRules": ["R001"],
        "humanRules": ["R001"],
        "transcript": """（2026-11-01 21:30:01 坐席：别再问了，闭嘴行不行。
2026-11-01 21:30:04 客户：我只是想确认费用。
2026-11-01 21:30:07 坐席：确认什么确认，烦死了。
）""",
    },
    {
        "id": "call-c008",
        "callId": "C008",
        "autoRules": ["R004"],
        "humanRules": [],
        "transcript": """（2026-11-01 21:35:40 坐席：我们不会上门催收，也不会威胁您，请放心。
2026-11-01 21:35:44 客户：那就好。
2026-11-01 21:35:47 坐席：只是提醒您按时还款。
）""",
    },
    {
        "id": "call-c009",
        "callId": "C009",
        "autoRules": ["R004"],
        "humanRules": ["R001", "R004"],
        "transcript": """（2026-11-01 21:40:12 坐席：你不还的话，我们就去你单位找你领导。
2026-11-01 21:40:16 客户：别去我单位。
2026-11-01 21:40:18 坐席：那就赶紧还，废物。
）""",
    },
    {
        "id": "call-c010",
        "callId": "C010",
        "autoRules": [],
        "humanRules": [],
        "transcript": """（2026-11-01 21:45:00 坐席：您好，请问需要办理什么业务？
2026-11-01 21:45:03 客户：我想改一下套餐。
2026-11-01 21:45:06 坐席：好的，我先核对您的身份信息。
）""",
    },
]


def build_state() -> Dict[str, Any]:
    now = _iso()
    rules = []
    for item in RULES:
        rules.append({
            **item,
            "version": 1,
            "createdAt": now,
            "updatedAt": now,
        })
    calls = []
    for index, call in enumerate(CALLS):
        calls.append({
            **call,
            "autoRules": list(call["autoRules"]),
            "humanRules": list(call["humanRules"]),
            "importedAt": _iso(index),
        })
    return {
        "version": 2,
        "rules": rules,
        "calls": calls,
        "splits": build_splits(rules, calls),
        "semanticPrompt": DEFAULT_SEMANTIC_PROMPT,
        "semanticPromptVersion": 1,
        "promptHistory": [],
        "runs": [],
        "settings": {
            "mode": "mock",
            "baseUrl": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "apiProtocol": "openai",
            "apiPath": "/chat/completions",
            "headers": [dict(item) for item in DEFAULT_HEADERS],
            "requestTemplate": DEFAULT_REQUEST_TEMPLATE,
            "contentPath": "choices.0.message.content",
            "violationPath": "",
            "reasonPath": "",
            "timeoutSec": 90,
            "maxRetries": 3,
            "retryBackoffSec": 1.5,
            "requestIntervalSec": 0.5,
        },
    }
