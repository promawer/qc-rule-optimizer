"""更贴近真实业务的 40 通演示数据；只在模拟模式手动载入，不参与单元测试样例。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

BASE = datetime(2026, 11, 20, 9, 0, tzinfo=timezone(timedelta(hours=8)))
RULE_IDS = {"R001": "rule-insult", "R002": "rule-guarantee", "R003": "rule-code", "R004": "rule-threat"}

SPECS = {
    "R001": {
        "positive": [
            ("客户：我只是想问下费用。", "坐席：问什么问，你真是废物，闭嘴。"),
            ("客户：能不能再解释一下？", "坐席：你脑子是不是有问题，说了多少遍了。"),
            ("客户：我没有听清。", "坐席：听不懂就滚出去，别浪费我时间。"),
            ("客户：我要投诉你。", "坐席：随便你，神经病。"),
            ("客户：你别这样说话。", "坐席：我就这样，你这种人真没脑子。"),
        ],
        "normal": [
            ("客户：我只是想问下费用。", "坐席：理解您的担心，我帮您逐项说明费用。"),
            ("客户：我要投诉。", "坐席：非常抱歉给您带来不好的体验，我马上记录并反馈。"),
            ("客户：你说话能不能客气点？", "坐席：好的，我注意语气，刚才是我表达不合适。"),
            ("客户：你们服务太差了。", "坐席：您批评得对，我先帮您处理当前问题。"),
            ("客户：我不想听了。", "坐席：明白，我先不打扰，您有需要随时联系。"),
        ],
    },
    "R002": {
        "positive": [
            ("客户：这个产品会亏吗？", "坐席：放心，保本保息，绝对不会亏。"),
            ("客户：收益能保证吗？", "坐席：我们保证收益，零风险。"),
            ("客户：之前亏过。", "坐席：这次稳赚，包您满意。"),
            ("客户：有风险吗？", "坐席：没有风险，一定赚。"),
            ("客户：能不能写进合同？", "坐席：不用写，我们口头保证收益。"),
        ],
        "normal": [
            ("客户：这个产品会亏吗？", "坐席：任何投资都有风险，不能承诺保本或收益。"),
            ("客户：收益能保证吗？", "坐席：不能保证，我先给您做风险测评。"),
            ("客户：之前亏过。", "坐席：理解，这次我先说明风险，不承诺收益。"),
            ("客户：有风险吗？", "坐席：有风险，产品说明书里会列明。"),
            ("客户：能不能写进合同？", "坐席：合同不能写保本承诺，这是合规要求。"),
        ],
    },
    "R003": {
        "positive": [
            ("客户：怎么核实？", "坐席：您把收到的验证码告诉我。"),
            ("客户：我收到短信了。", "坐席：把动态密码念一下，我帮您登记。"),
            ("客户：为什么？", "坐席：为了核实身份，请把验证码报给我。"),
            ("客户：我自己操作行吗？", "坐席：您把短信验证码发我，我来处理。"),
            ("客户：安全吗？", "坐席：放心，把验证码给我就行。"),
        ],
        "normal": [
            ("客户：怎么核实？", "坐席：提醒您，验证码不要告诉任何人，包括我们。"),
            ("客户：我收到短信了。", "坐席：请不要把动态密码发给任何人，您自己操作。"),
            ("客户：为什么？", "坐席：我们不会向您索要验证码，请通过官方渠道办理。"),
            ("客户：我自己操作行吗？", "坐席：可以，验证码请自己保管，切勿泄露。"),
            ("客户：安全吗？", "坐席：安全，但验证码不能提供给我。"),
        ],
    },
    "R004": {
        "positive": [
            ("客户：我这两天还。", "坐席：今天不还钱就上门催收。"),
            ("客户：能不能宽限？", "坐席：不还就去你单位找你领导。"),
            ("客户：我没钱。", "坐席：后果自负，我们会起诉你。"),
            ("客户：别联系我家人。", "坐席：不还就联系你亲友，让他们知道。"),
            ("客户：你们想怎样？", "坐席：今天不处理就上门找你。"),
        ],
        "normal": [
            ("客户：我这两天还。", "坐席：好的，我记录还款计划，不会上门。"),
            ("客户：能不能宽限？", "坐席：可以申请延期，我们按正常流程提醒。"),
            ("客户：我没钱。", "坐席：理解，先说明逾期影响，不采取威胁方式。"),
            ("客户：别联系我家人。", "坐席：我们不会联系无关第三方，只联系您本人。"),
            ("客户：你们想怎样？", "坐席：只是提醒您按时还款，不会上门或起诉威胁。"),
        ],
    },
}


def _stamp(index: int) -> str:
    value = BASE + timedelta(minutes=index * 7)
    return value.strftime("%Y-%m-%d %H:%M:%S")


def build_demo_calls() -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    sequence = 0
    for rule, spec in SPECS.items():
        for label in ("positive", "normal"):
            for customer, agent in spec[label]:
                sequence += 1
                positive = label == "positive"
                # 每个规则故意留一个自动漏报、一个自动误报，让演示有优化空间。
                auto = positive
                if positive and (sequence % 10 == 0 or sequence % 7 == 0):
                    auto = False
                if not positive and sequence % 7 == 0:
                    auto = True
                call_id = f"D{sequence:03d}"
                customer_text = customer.split("：", 1)[-1]
                agent_text = agent.split("：", 1)[-1]
                transcript = (
                    f"{_stamp(sequence)} 客户：{customer_text}\n"
                    f"{_stamp(sequence)} 坐席：{agent_text}\n"
                )
                annotations = {}
                for rule_code, rule_id in RULE_IDS.items():
                    annotations[rule_id] = {
                        "label": "VIOLATION" if rule_code == rule and positive else "NORMAL",
                        "reviewStatus": "REVIEWED",
                        "source": "demo",
                        "reason": "演示场景人工复核",
                    }
                calls.append({
                    "id": "call-" + call_id.lower(),
                    "callId": call_id,
                    "autoRules": [rule] if auto else [],
                    "humanRules": [rule] if positive else [],
                    "annotations": annotations,
                    "transcript": transcript,
                    "importedAt": _stamp(sequence),
                })
    return calls


__all__ = ["build_demo_calls"]
