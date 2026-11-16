"""优化循环：评测 → 让模型给出新规则 → 复评 → 保留最优，直到达标或轮次用尽。"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

from .evaluator import EvaluationError, JobCancelled, evaluate_rule
from .llm import LLMError
from .models import NORMAL, VIOLATION
from .prompts import validate_prompt
from .splits import TEST, TRAIN, VALIDATION, split_cases, split_summary

NEG_WORDS = [
    "不要", "不用", "无需", "切勿", "别", "不会", "不能",
    "谨防", "提醒", "请注意", "注意查收", "没有", "并非",
]

STOP_PHRASES = {
    "坐席", "客户", "客服", "用户", "专员", "您好", "我们", "你们", "这个", "那个",
    "什么", "怎么", "可以", "就是", "现在", "已经", "还是", "如果", "因为", "所以",
    "但是", "然后", "请问", "谢谢", "好的", "知道", "收到", "麻烦", "需要", "没有",
    "不要", "不是", "不会", "一下", "这样", "那样", "时候", "问题", "业务", "办理",
}


def _clean_text(text: str) -> str:
    lines = []
    for line in str(text or "").splitlines():
        line = re.sub(r"^\s*[（(]?\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\s*", "", line)
        line = re.sub(r"^\s*(坐席|客户|客服|用户|专员|agent|customer)\s*[：:]\s*", "", line, flags=re.I)
        lines.append(line)
    return "\n".join(lines)


def _agent_text(text: str) -> str:
    """只保留坐席发言：违规表达应来自坐席，不能把客户的回应写进规则。"""
    lines = []
    for line in str(text or "").splitlines():
        matched = re.search(r"(坐席|客服|专员|agent)\s*[：:]\s*(.*)$", line, flags=re.I)
        if matched:
            lines.append(matched.group(2))
    return "\n".join(lines) if lines else _clean_text(text)


def _ngrams(text: str) -> set:
    cleaned = _agent_text(text)
    grams = set()
    for segment in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,}", cleaned):
        for size in (2, 3, 4):
            for index in range(len(segment) - size + 1):
                grams.add(segment[index:index + size])
    return grams


def _candidate_phrases(fn_texts: List[str], other_texts: List[str], limit: int = 2) -> List[str]:
    counts: Dict[str, int] = {}
    other_agent = [_agent_text(text) for text in other_texts]
    for text in fn_texts:
        for gram in _ngrams(text):
            if gram in STOP_PHRASES or len(gram) < 2 or gram.isdigit():
                continue
            counts[gram] = counts.get(gram, 0) + 1
    scored = []
    for gram, count in counts.items():
        if any(gram in other for other in other_agent):
            continue
        scored.append((count, len(gram), gram))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    picked: List[str] = []
    for _, _, gram in scored:
        if any(gram in existing or existing in gram for existing in picked):
            continue
        picked.append(gram)
        if len(picked) >= limit:
            break
    return picked


def _error_text(result: Dict[str, Any], limit_each: int = 6) -> str:
    rows = result.get("rows") or []
    fn = [r for r in rows if r["predicted"] is False and r["humanResult"] == VIOLATION][:limit_each]
    fp = [r for r in rows if r["predicted"] is True and r["humanResult"] == NORMAL][:limit_each]
    blocks: List[str] = []
    for title, items in (("漏报 FN：人工判违规，规则判正常", fn), ("误报 FP：人工判正常，规则判违规", fp)):
        if not items:
            continue
        blocks.append(f"【{title}】")
        for row in items:
            transcript = str(row.get("transcript") or "").replace("\r", "").strip()
            blocks.append(f"- 通话ID {row.get('callId')}；文本：{transcript[:600]}")
    return "\n".join(blocks) if blocks else "（本轮没有误报/漏报样本）"


def build_optimizer_prompt(
    rule: Dict[str, Any],
    result: Dict[str, Any],
    target: float,
    history: List[Dict[str, Any]],
    semantic_prompt: str,
) -> str:
    metrics = result.get("metrics") or {}
    auto = metrics.get("autoAccuracy")
    history_text = "\n".join(
        f"第{item['index']}轮：准确率 {item['metrics']['accuracy']:.1%}；修改理由：{item.get('rationale', '')[:120]}"
        for item in history
    ) or "（还没有历史尝试）"
    return f"""你是资深的外呼质检规则工程师。请修改一条质检规则的「正则表达式 + 语义描述 + 逻辑关系」，
并视情况改进「通用语义提示词」，使判定尽量接近人工复核结果，并达到目标准确率。

当前规则：
- 名称：{rule.get('name', '')}
- 业务说明：{rule.get('description', '') or '无'}
- 逻辑关系：{rule.get('logic', 'OR')}（OR=正则或语义任一命中即违规；AND=两者都命中才违规）
- 正则表达式：{rule.get('regex', '') or '（空）'}
- 语义描述：{rule.get('semantic', '') or '（空）'}

当前通用语义提示词（生产上所有规则共用；{{rules}} 会被替换成全部语义规则）：
{semantic_prompt}

当前评测（训练集 {metrics.get('total', 0)} 条人工标注样本）：
- 准确率 {metrics.get('accuracy', 0):.1%}，目标准确率 {target:.1%}
- 精确率 {metrics.get('precision', 0):.1%}，召回率 {metrics.get('recall', 0):.1%}，F1 {metrics.get('f1', 0):.1%}
- 误报 FP={metrics.get('fp', 0)}，漏报 FN={metrics.get('fn', 0)}
- 原自动质检准确率：{f'{auto:.1%}' if auto is not None else '未知'}

训练集错误样本：
{_error_text(result)}

历史尝试：
{history_text}

请输出改进后的规则。要求：
1. 只能通过规则内容解决，不要写死通话ID、客户姓名、具体手机号或验证码；
2. 正则使用 Python re 语法，长度不超过 2000 字符；
3. 语义描述要明确“什么情况算违规”，可以补充排除条件；
4. 优先修复错误样本反映的共性问题：漏报补充同义表达，误报收紧或增加排除条件；
5. 必要时可调整 OR/AND，并在 rationale 中说明原因；
6. 至少保留正则、语义中的一项非空；
7. semanticPrompt 可选：如果通用提示词存在系统性缺陷（例如没有要求区分提醒/否定/引用、没有限定只看坐席），可以改进；
8. 如果修改 semanticPrompt，必须保留 {{rules}} 占位符，并在 promptRationale 中说明原因；不改就原样返回。

只输出 JSON：
{{"regex": "...", "semantic": "...", "logic": "OR", "rationale": "规则为什么这样改", "expectedEffect": "预期改善", "semanticPrompt": "完整提示词（含 {{rules}}）", "promptRationale": "提示词为什么改，不改则留空"}}"""


def normalize_proposal(
    data: Dict[str, Any], fallback: Dict[str, Any], fallback_prompt: str
) -> Dict[str, Any]:
    regex = str(data.get("regex", fallback.get("regex", "")) or "").strip()[:2000]
    semantic = str(data.get("semantic", fallback.get("semantic", "")) or "").strip()[:1000]
    logic = str(data.get("logic", fallback.get("logic", "OR")) or "OR").upper()
    if logic not in ("OR", "AND"):
        logic = str(fallback.get("logic") or "OR").upper()
    if not regex and not semantic:
        raise EvaluationError("候选规则的正则和语义都为空")
    if regex:
        try:
            re.compile(regex)
        except re.error as exc:
            raise EvaluationError(f"候选正则无法编译：{exc}") from exc
    prompt = str(data.get("semanticPrompt") or fallback_prompt).strip()
    prompt_changed = prompt != fallback_prompt
    if prompt_changed:
        try:
            prompt = validate_prompt(prompt)
        except ValueError as exc:
            raise EvaluationError(f"候选通用语义提示词不合法：{exc}") from exc
    return {
        "regex": regex,
        "semantic": semantic,
        "logic": logic,
        "rationale": str(data.get("rationale") or "")[:500],
        "expectedEffect": str(data.get("expectedEffect") or "")[:300],
        "semanticPrompt": prompt,
        "promptChanged": prompt_changed,
        "promptRationale": str(data.get("promptRationale") or "")[:500],
    }


class LLMProposer:
    def __init__(self, client: Any) -> None:
        self.client = client

    async def __call__(
        self,
        rule: Dict[str, Any],
        result: Dict[str, Any],
        target: float,
        history: List[Dict[str, Any]],
        semantic_prompt: str,
    ) -> Dict[str, Any]:
        system = "你是外呼质检规则优化专家，只输出 JSON，不输出解释性前后缀。"
        data = await self.client.chat_json(
            system,
            build_optimizer_prompt(rule, result, target, history, semantic_prompt),
            max_tokens=1800,
            temperature=0.3,
        )
        proposal = normalize_proposal(data, rule, semantic_prompt)
        proposal["source"] = "llm"
        return proposal


class MockProposer:
    """无需 API Key 的本地启发式优化器，用于演示页面流程。

    它只根据误报/漏报样本做两类确定性修改：
    - 误报：把只在误报样本里出现的否定/提醒词加入负向前瞻；
    - 漏报：把只在漏报样本里出现的短语补充为正则的候选分支。
    """

    async def __call__(
        self,
        rule: Dict[str, Any],
        result: Dict[str, Any],
        target: float,
        history: List[Dict[str, Any]],
        semantic_prompt: str,
    ) -> Dict[str, Any]:
        rows = result.get("rows") or []
        fp_texts = [str(r["transcript"]) for r in rows if r["predicted"] is True and r["humanResult"] == NORMAL]
        fn_texts = [str(r["transcript"]) for r in rows if r["predicted"] is False and r["humanResult"] == VIOLATION]
        other_texts = [
            str(r["transcript"]) for r in rows
            if (r["predicted"] is True and r["humanResult"] == VIOLATION)
            or (r["predicted"] is False and r["humanResult"] == NORMAL)
        ]
        core = str(rule.get("regex") or "").strip()
        reasons: List[str] = []

        if not core:
            return {
                "regex": "", "semantic": str(rule.get("semantic") or ""),
                "logic": str(rule.get("logic") or "OR"),
                "rationale": "模拟优化器不会改写纯语义规则；请配置真实模型。",
                "expectedEffect": "无", "source": "mock",
                "semanticPrompt": semantic_prompt, "promptChanged": False, "promptRationale": "",
            }

        body = core
        additions = [p for p in _candidate_phrases(fn_texts, other_texts) if p not in core]
        if additions:
            body = "(?:" + core + "|" + "|".join(re.escape(p) for p in additions) + ")"
            reasons.append("补充漏报中的同义表达：" + "、".join(additions))

        fp_agent_texts = [_agent_text(text) for text in fp_texts]
        other_agent_texts = [_agent_text(text) for text in other_texts]
        words = [
            w for w in NEG_WORDS
            if any(w in text for text in fp_agent_texts)
            and not any(w in text for text in other_agent_texts)
        ]
        fp_phrases: List[str] = []
        for text in fp_texts[:4]:
            picked = _candidate_phrases([text], other_texts, limit=1)
            if picked and picked[0] not in words and picked[0] not in fp_phrases:
                fp_phrases.append(picked[0])
        excludes = words + [p for p in fp_phrases if p not in words]
        if excludes and not body.startswith("^(?!"):
            # 在开头校验全文不含误报语境词，同时仍允许规则体出现在任意位置。
            excluded = "|".join(re.escape(w) for w in excludes[:6])
            body = (
                "^(?![\\s\\S]*(?:" + excluded + "))"
                + "(?=[\\s\\S]*(?:" + body + "))"
            )
            reasons.append("增加误报排除条件：" + "、".join(excludes[:6]))

        if not reasons:
            reasons.append("模拟优化器未找到可自动修正的模式；多轮无改善时可切换真实模型。")

        return {
            "regex": body,
            "semantic": str(rule.get("semantic") or ""),
            "logic": str(rule.get("logic") or "OR"),
            "rationale": "；".join(reasons),
            "expectedEffect": "收紧误报并补足漏报表达" if len(reasons) > 1 else "按错误样本做局部修正",
            "source": "mock",
            "semanticPrompt": semantic_prompt,
            "promptChanged": False,
            "promptRationale": "",
        }


Event = Callable[[Dict[str, Any]], None]


def _rule_snapshot(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": rule.get("name", ""),
        "description": rule.get("description", ""),
        "regex": rule.get("regex", ""),
        "semantic": rule.get("semantic", ""),
        "logic": rule.get("logic", "OR"),
    }


def _same_rule(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return (
        str(left.get("regex") or "") == str(right.get("regex") or "")
        and str(left.get("semantic") or "") == str(right.get("semantic") or "")
        and str(left.get("logic") or "OR") == str(right.get("logic") or "OR")
    )


def compute_gate(
    best_validation: Dict[str, Any],
    best_test: Dict[str, Any],
    baseline_validation: Dict[str, Any],
    baseline_test: Dict[str, Any],
    target: float,
    summary: Dict[str, Any],
    improved: bool,
) -> Dict[str, Any]:
    validation = best_validation["metrics"]
    test = best_test["metrics"]
    base_validation = baseline_validation["metrics"]
    base_test = baseline_test["metrics"]
    val_counts = summary.get(VALIDATION, {})
    test_counts = summary.get(TEST, {})

    checks = []

    def add(key: str, label: str, passed: bool, detail: str) -> None:
        checks.append({"key": key, "label": label, "pass": bool(passed), "detail": detail})

    add(
        "validation_classes",
        "验证集同时包含违规和正常样本",
        val_counts.get("positives", 0) >= 1 and val_counts.get("negatives", 0) >= 1,
        f"违规 {val_counts.get('positives', 0)} / 正常 {val_counts.get('negatives', 0)}",
    )
    add(
        "test_classes",
        "测试集同时包含违规和正常样本",
        test_counts.get("positives", 0) >= 1 and test_counts.get("negatives", 0) >= 1,
        f"违规 {test_counts.get('positives', 0)} / 正常 {test_counts.get('negatives', 0)}",
    )
    add(
        "validation_target",
        "验证集达到目标准确率",
        validation["accuracy"] >= target - 1e-9,
        f"{validation['accuracy']:.1%} / 目标 {target:.1%}",
    )
    add(
        "test_no_regression",
        "测试集不低于基线",
        test["accuracy"] >= base_test["accuracy"] - 0.02,
        f"最优 {test['accuracy']:.1%} / 基线 {base_test['accuracy']:.1%}",
    )
    add(
        "test_target",
        "测试集接近目标（允许低 5 个百分点）",
        test["accuracy"] >= target - 0.05,
        f"{test['accuracy']:.1%} / 目标 {target:.1%}",
    )
    add(
        "gap",
        "验证集与测试集差距不超过 15 个百分点",
        abs(validation["accuracy"] - test["accuracy"]) <= 0.15,
        f"验证 {validation['accuracy']:.1%} / 测试 {test['accuracy']:.1%}",
    )
    add(
        "improved",
        "相对基线有实际提升",
        improved or (validation["accuracy"] >= target and test["accuracy"] >= target),
        f"验证 {base_validation['accuracy']:.1%}→{validation['accuracy']:.1%}，测试 {base_test['accuracy']:.1%}→{test['accuracy']:.1%}",
    )

    if not checks[0]["pass"] or not checks[1]["pass"]:
        status = "INSUFFICIENT"
    elif all(check["pass"] for check in checks):
        status = "PASS"
    else:
        status = "FAIL"
    reasons = [f"{check['label']}（{check['detail']}）" for check in checks if not check["pass"]]
    if status == "INSUFFICIENT":
        reasons.insert(0, "验证集或测试集缺少违规/正常样本，无法验证泛化能力，请补充通话数据")
    if status == "PASS":
        reasons = ["验证集与测试集均通过，且相对基线有提升"]
    return {"status": status, "checks": checks, "reasons": reasons}


async def optimize(
    rule: Dict[str, Any],
    cases: List[Dict[str, Any]],
    target: float,
    max_rounds: int,
    judge_factory: Callable[[str, bool], Callable],
    proposer: Callable,
    prompt: str,
    emit: Event,
    is_cancelled: Callable[[], bool],
    cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cache = cache if cache is not None else {}
    grouped = split_cases(cases)
    train_cases, validation_cases, test_cases = grouped[TRAIN], grouped[VALIDATION], grouped[TEST]
    if not train_cases or not validation_cases or not test_cases:
        raise EvaluationError("训练集、验证集、测试集不完整，无法进行可泛化的评测；请补充通话数据")
    summary = split_summary(cases)

    baseline_rule = _rule_snapshot(rule)
    best_rule = dict(baseline_rule)
    baseline_prompt = prompt
    best_prompt = prompt

    async def evaluate_split(candidate_rule, items, semantic_prompt, *, force_parts, with_reason):
        judge = judge_factory(semantic_prompt, with_reason)
        return await evaluate_rule(
            candidate_rule,
            items,
            judge,
            cache=cache,
            progress=lambda done, total: emit({"type": "progress", "done": done, "total": total}),
            is_cancelled=is_cancelled,
            force_parts=force_parts,
            with_reason=with_reason,
            cache_scope=f"{semantic_prompt}\x00{'reason' if with_reason else 'verdict'}",
        )

    emit({"type": "phase", "phase": "基线评测（训练集，含语义理由）", "done": 0, "total": len(train_cases)})
    baseline_train = await evaluate_split(
        baseline_rule, train_cases, baseline_prompt, force_parts=True, with_reason=True
    )
    emit({"type": "phase", "phase": "基线评测（验证集，仅判定不给理由）", "done": 0, "total": len(validation_cases)})
    baseline_validation = await evaluate_split(
        baseline_rule, validation_cases, baseline_prompt, force_parts=False, with_reason=False
    )
    emit({
        "type": "log",
        "message": f"基线：训练集 {baseline_train['metrics']['accuracy']:.1%}，验证集 {baseline_validation['metrics']['accuracy']:.1%}",
    })
    emit({"type": "baseline", "metrics": baseline_train["metrics"], "rule": baseline_train["rule"]})

    best_train = baseline_train
    best_validation = baseline_validation
    iterations: List[Dict[str, Any]] = []
    status = "reached" if baseline_validation["metrics"]["accuracy"] >= target else "running"

    while status == "running" and len(iterations) < max_rounds:
        if is_cancelled():
            raise JobCancelled()
        index = len(iterations) + 1
        emit({"type": "phase", "phase": f"第 {index} 轮：生成候选规则与提示词", "done": 0, "total": len(train_cases)})
        started = time.time()
        # 只把训练集的错误样本交给优化器，验证集/测试集不进入 prompt
        proposal = await proposer(best_rule, best_train, target, iterations, best_prompt)
        candidate_rule = {
            "name": best_rule.get("name", ""),
            "description": best_rule.get("description", ""),
            "regex": proposal["regex"],
            "semantic": proposal["semantic"],
            "logic": proposal["logic"],
        }
        candidate_prompt = proposal.get("semanticPrompt") or best_prompt
        prompt_changed = candidate_prompt != best_prompt
        emit({"type": "log", "message": f"第 {index} 轮候选：{proposal.get('rationale') or '无说明'}"})
        emit({"type": "phase", "phase": f"第 {index} 轮：训练集评测（含语义理由）", "done": 0, "total": len(train_cases)})
        candidate_train = await evaluate_split(
            candidate_rule, train_cases, candidate_prompt, force_parts=True, with_reason=True
        )
        emit({"type": "phase", "phase": f"第 {index} 轮：验证集评测（仅判定）", "done": 0, "total": len(validation_cases)})
        candidate_validation = await evaluate_split(
            candidate_rule, validation_cases, candidate_prompt, force_parts=False, with_reason=False
        )
        accepted = candidate_validation["metrics"]["accuracy"] > best_validation["metrics"]["accuracy"] or (
            candidate_validation["metrics"]["accuracy"] == best_validation["metrics"]["accuracy"]
            and candidate_train["metrics"]["accuracy"] > best_train["metrics"]["accuracy"]
        )
        iteration = {
            "index": index,
            "rule": candidate_train["rule"],
            "metrics": candidate_train["metrics"],
            "validationMetrics": candidate_validation["metrics"],
            "trainRows": candidate_train["rows"],
            "validationRows": candidate_validation["rows"],
            "rationale": proposal.get("rationale", ""),
            "expectedEffect": proposal.get("expectedEffect", ""),
            "source": proposal.get("source", ""),
            "accepted": accepted,
            "semanticPrompt": candidate_prompt if prompt_changed else None,
            "promptChanged": prompt_changed,
            "promptRationale": proposal.get("promptRationale", ""),
            "durationMs": int((time.time() - started) * 1000),
        }
        iterations.append(iteration)
        emit({"type": "iteration", "iteration": iteration})
        if accepted:
            best_train = candidate_train
            best_validation = candidate_validation
            best_rule = candidate_rule
            best_prompt = candidate_prompt
            emit({"type": "log", "message": f"第 {index} 轮验证集提升到 {best_validation['metrics']['accuracy']:.1%}，保留该版本"})
        else:
            emit({"type": "log", "message": f"第 {index} 轮验证集 {candidate_validation['metrics']['accuracy']:.1%}，未超过当前最优，丢弃"})

        if best_validation["metrics"]["accuracy"] >= target:
            status = "reached"
        elif index >= max_rounds:
            status = "exhausted"

    if status == "running":
        status = "exhausted"

    emit({"type": "phase", "phase": "测试集最终验收（只评一次，仅判定）", "done": 0, "total": len(test_cases)})
    baseline_test = await evaluate_split(
        baseline_rule, test_cases, baseline_prompt, force_parts=False, with_reason=False
    )
    if _same_rule(best_rule, baseline_rule) and best_prompt == baseline_prompt:
        best_test = baseline_test
    else:
        best_test = await evaluate_split(
            best_rule, test_cases, best_prompt, force_parts=False, with_reason=False
        )

    improved = (
        best_test["metrics"]["accuracy"] > baseline_test["metrics"]["accuracy"]
        or best_validation["metrics"]["accuracy"] > baseline_validation["metrics"]["accuracy"]
    )
    gate = compute_gate(
        best_validation, best_test, baseline_validation, baseline_test, target, summary, improved
    )
    emit({
        "type": "log",
        "message": f"测试集：基线 {baseline_test['metrics']['accuracy']:.1%}，最优 {best_test['metrics']['accuracy']:.1%}；验收 {gate['status']}",
    })

    return {
        "status": status,
        "baseline": {"train": baseline_train, "validation": baseline_validation, "test": baseline_test},
        "best": {"train": best_train, "validation": best_validation, "test": best_test},
        "iterations": iterations,
        "gate": gate,
        "splitSummary": summary,
        "baselinePrompt": baseline_prompt,
        "bestPrompt": best_prompt,
        "promptChanged": best_prompt != baseline_prompt,
    }

