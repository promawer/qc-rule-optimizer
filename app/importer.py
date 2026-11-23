"""解析通话导入文件（统一通话表格式）。

一行一通电话，四列：
1. 通话ID
2. 通话文本（可含多行、时间戳、坐席/客户角色）
3. 自动质检判定的违规规则编号（多条用 ; 或 、 分隔，空表示未判违规）
4. 人工复核的违规规则编号（空表示没有明确正样本，不能推定正常）
5. 已复核规则（可选；范围内未违规的规则才标记正常）

标注以稳定 ruleId 保存。未复核标签 UNKNOWN，只有显式全量复核导入可以补齐正常。
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openpyxl import load_workbook

from .data_governance import normalize_annotation, rule_identity

MAX_ROWS = 2000
MAX_TRANSCRIPT = 20000

_ID_KEYS = {"通话id", "通话编号", "录音id", "录音编号", "callid", "id", "序号", "工单号", "sessionid"}
_TEXT_KEYS = {"通话文本", "转写文本", "文本内容", "通话内容", "对话内容", "内容", "transcript", "text", "dialog", "对话", "asr文本"}
_AUTO_RULE_KEYS = {
    "自动违规规则", "自动质检违规规则", "自动判定违规规则", "自动规则编号", "自动违规规则编号",
    "auto_rules", "autorules", "auto_rule_codes", "auto_violation_rules",
}
_HUMAN_RULE_KEYS = {
    "人工违规规则", "人工复核违规规则", "人工判定违规规则", "人工规则编号", "人工违规规则编号",
    "human_rules", "humanrules", "human_rule_codes", "human_violation_rules",
}
_REVIEWED_KEYS = {"已复核规则", "已复核规则编号", "复核范围", "reviewedRules", "reviewed_rules"}
_GROUP_KEYS = {"同源组", "同源组ID", "客户组ID", "groupId", "group_id"}
_ANNOTATION_KEYS = {"标注详情", "annotations", "annotation_json"}
_COMPLETE_KEYS = {"转写完整", "通话完整", "transcriptComplete", "transcript_complete"}


def _canon(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[\s_\-（）()【】\[\]:：/\\]", "", text)


def _match(headers: List[str], aliases: set, contains: Tuple[str, ...] = ()) -> str:
    canonical_aliases = {_canon(alias) for alias in aliases}
    for header in headers:
        if _canon(header) in canonical_aliases:
            return header
    if contains:
        for header in headers:
            norm = _canon(header)
            if all(part in norm for part in contains):
                return header
    return ""


def detect_columns(headers: List[str]) -> Dict[str, str]:
    return {
        "callId": _match(headers, _ID_KEYS),
        "transcript": _match(headers, _TEXT_KEYS),
        "autoRules": _match(headers, _AUTO_RULE_KEYS) or _match(headers, _AUTO_RULE_KEYS, ("自动", "规则")),
        "humanRules": _match(headers, _HUMAN_RULE_KEYS) or _match(headers, _HUMAN_RULE_KEYS, ("人工", "规则")),
        "reviewedRules": _match(headers, _REVIEWED_KEYS),
        "groupId": _match(headers, _GROUP_KEYS),
        "annotations": _match(headers, _ANNOTATION_KEYS),
        "transcriptComplete": _match(headers, _COMPLETE_KEYS),
    }


def _read_csv(content: bytes) -> Tuple[List[str], List[List[object]]]:
    text = None
    for encoding in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("无法识别文件编码，请另存为 UTF-8 或 GBK 的 CSV")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = list(csv.reader(io.StringIO(text), dialect))
    reader = [row for row in reader if any(str(cell).strip() for cell in row)]
    if not reader:
        raise ValueError("文件内容为空")
    return [str(cell).strip() for cell in reader[0]], reader[1:]


def _read_json(content: bytes) -> Tuple[List[str], List[List[object]]]:
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON 解析失败：{exc}") from exc
    if isinstance(payload, dict):
        payload = (
            payload.get("calls")
            or payload.get("cases")
            or payload.get("data")
            or payload.get("rows")
            or payload.get("rules")
        )
    if not isinstance(payload, list) or not payload:
        raise ValueError("JSON 需要是非空数组，或包含 calls / cases / data / rows 数组")
    headers: List[str] = []
    for item in payload:
        if isinstance(item, dict):
            for key in item:
                if key not in headers:
                    headers.append(str(key))
    rows = [[item.get(header, "") for header in headers] for item in payload if isinstance(item, dict)]
    return headers, rows


def _read_xlsx(content: bytes) -> Tuple[List[str], List[List[object]]]:
    try:
        workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Excel 读取失败：{exc}") from exc
    sheet = workbook.active
    rows = []
    for row in sheet.iter_rows(values_only=True):
        if row and any(cell is not None and str(cell).strip() for cell in row):
            rows.append(list(row))
    workbook.close()
    if not rows:
        raise ValueError("Excel 内容为空")
    return [str(cell or "").strip() for cell in rows[0]], rows[1:]


def parse_file(filename: str, content: bytes) -> Tuple[List[str], List[List[object]]]:
    name = (filename or "").lower()
    if name.endswith(".json"):
        return _read_json(content)
    if name.endswith(".xlsx"):
        return _read_xlsx(content)
    if name.endswith(".xls"):
        raise ValueError("不支持旧版 .xls，请另存为 .xlsx 或 CSV")
    if name.endswith(".csv") or name.endswith(".txt"):
        return _read_csv(content)
    raise ValueError("仅支持 .csv / .json / .xlsx 文件")


def _split_codes(value: object) -> List[str]:
    if isinstance(value, (list, tuple)):
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    text = str(value or "").strip()
    if not text:
        return []
    tokens = [token.strip() for token in re.split(r"[,，;；、|/\s]+", text) if token.strip()]
    seen = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    return seen


def _resolve_codes(
    tokens: List[str],
    code_by_upper: Dict[str, str],
    code_by_name: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    resolved: List[str] = []
    unknown: List[str] = []
    for token in tokens:
        key = token.upper()
        if key in code_by_upper:
            code = code_by_upper[key]
        elif token in code_by_name:
            code = code_by_name[token]
        else:
            code = key
            unknown.append(code)
        if code not in resolved:
            resolved.append(code)
    return resolved, unknown


def build_calls(
    filename: str,
    content: bytes,
    rules: List[Dict[str, Any]],
    fully_reviewed: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    headers, rows = parse_file(filename, content)
    columns = detect_columns(headers)
    missing = []
    if not columns.get("transcript"):
        missing.append("通话文本")
    if not columns.get("humanRules") and not columns.get("annotations"):
        missing.append("人工违规规则或标注详情")
    if missing:
        raise ValueError(
            "缺少必需列：" + "、".join(missing)
            + "。当前表头：" + "、".join(headers[:12])
            + "。期望格式：通话ID, 通话文本, 自动违规规则, 人工违规规则, 已复核规则"
        )

    code_by_upper = {str(rule["code"]).upper(): rule["code"] for rule in rules if rule.get("code")}
    code_by_name = {str(rule.get("name") or "").strip(): rule["code"] for rule in rules if rule.get("code")}
    index = {header: position for position, header in enumerate(headers)}
    calls: List[Dict[str, Any]] = []
    seen = set()
    skipped = 0
    unknown_auto: List[str] = []
    unknown_human: List[str] = []
    warnings: List[str] = []

    for row_number, row in enumerate(rows, start=2):
        def raw_cell(key: str) -> Any:
            header = columns.get(key) or ""
            position = index.get(header)
            if position is None or position >= len(row):
                return ""
            return row[position]

        def cell(key: str) -> str:
            value = raw_cell(key)
            if value is None:
                return ""
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value).strip()

        full_transcript = cell("transcript")
        transcript = full_transcript[:MAX_TRANSCRIPT].strip()
        if not transcript:
            skipped += 1
            continue
        call_id = cell("callId") or "text-" + hashlib.sha256(full_transcript.encode("utf-8")).hexdigest()[:24]
        if call_id in seen:
            skipped += 1
            continue
        seen.add(call_id)

        auto_rules, unknown_a = _resolve_codes(_split_codes(raw_cell("autoRules")), code_by_upper, code_by_name)
        human_rules, unknown_h = _resolve_codes(_split_codes(raw_cell("humanRules")), code_by_upper, code_by_name)
        reviewed_rules, unknown_reviewed = _resolve_codes(_split_codes(raw_cell("reviewedRules")), code_by_upper, code_by_name)
        if unknown_reviewed:
            warnings.append(f"第 {row_number} 行：复核范围包含未知规则，尚未绑定：" + "、".join(unknown_reviewed))
        raw_annotations = raw_cell("annotations")
        if isinstance(raw_annotations, str) and raw_annotations.strip():
            try:
                raw_annotations = json.loads(raw_annotations)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"第 {row_number} 行：标注详情不是有效 JSON") from exc
        if raw_annotations and not isinstance(raw_annotations, dict):
            raise ValueError(f"第 {row_number} 行：标注详情需要以规则 ID 或编号为键的对象")
        raw_annotations = raw_annotations or {}
        annotations = {}
        for rule in rules:
            identity, code = rule_identity(rule), rule.get("code")
            raw_annotation = raw_annotations.get(identity, raw_annotations.get(code))
            foreign_identity = raw_annotation is None and any(
                isinstance(item, dict) and item.get("ruleCode") == code
                for item in raw_annotations.values()
            )
            if raw_annotation is not None:
                if not isinstance(raw_annotation, dict):
                    raise ValueError(f"第 {row_number} 行：规则 {code} 的标注详情必须是对象")
                annotation = normalize_annotation(raw_annotation, "imported-annotation")
            elif foreign_identity:
                annotation = normalize_annotation({}, "unresolved-rule-identity")
                warnings.append(f"第 {row_number} 行：规则 {code} 的标注属于其他规则 ID，请人工确认重新关联")
            elif code in human_rules:
                annotation = normalize_annotation({"label": "VIOLATION"}, "explicit-positive")
            elif fully_reviewed or code in reviewed_rules:
                annotation = normalize_annotation({"label": "NORMAL"}, "import-full-review" if fully_reviewed else "explicit-reviewed")
            else:
                annotation = normalize_annotation({}, "unreviewed")
            annotation["ruleCode"] = code
            annotations[identity] = annotation
        human_rules = [rule["code"] for rule in rules
                       if annotations[rule_identity(rule)]["label"] == "VIOLATION"] + unknown_h
        reviewed_rules = [rule["code"] for rule in rules
                          if annotations[rule_identity(rule)]["reviewStatus"] == "REVIEWED"]
        for code in unknown_a:
            if code not in unknown_auto:
                unknown_auto.append(code)
        for code in unknown_h:
            if code not in unknown_human:
                unknown_human.append(code)

        call = {
            "id": f"call-{uuid.uuid4().hex[:12]}",
            "callId": call_id,
            "transcript": transcript,
            "autoRules": auto_rules,
            "humanRules": human_rules,
            "reviewedRules": reviewed_rules,
            "annotations": annotations,
            "annotationSchema": 1,
            "groupId": cell("groupId"),
            "autoRuleIds": [rule_identity(rule) for rule in rules if rule.get("code") in auto_rules],
            "unresolvedHumanRules": unknown_h,
            "unresolvedAutoRules": unknown_a,
            "importedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if cell("transcriptComplete"):
            complete = cell("transcriptComplete").lower()
            if complete not in ("true", "false", "1", "0", "是", "否", "完整", "不完整"):
                raise ValueError(f"第 {row_number} 行：转写完整应填写 true/false 或 是/否")
            call["transcriptComplete"] = complete in ("true", "1", "是", "完整")
        if len(full_transcript) > MAX_TRANSCRIPT:
            call["transcriptComplete"] = False
            warnings.append(f"第 {row_number} 行：通话文本超过 {MAX_TRANSCRIPT} 字符，已截断并标记不完整")
        calls.append(call)
        if len(calls) >= MAX_ROWS:
            warnings.append(f"超过 {MAX_ROWS} 行，后续数据未导入")
            break

    if not calls:
        raise ValueError("没有解析到有效数据：请确认每行都有通话文本")
    if unknown_auto or unknown_human:
        warnings.append(
            "存在当前规则库中没有的规则编号（已保留为未绑定编号，后续需人工确认关联）："
            + "、".join(sorted(set(unknown_auto + unknown_human))[:10])
        )

    positives = {rule["code"]: 0 for rule in rules if rule.get("code")}
    for call in calls:
        for code in call["humanRules"]:
            positives[code] = positives.get(code, 0) + 1

    unknown_labels = sum(1 for call in calls for item in call["annotations"].values() if item["label"] == "UNKNOWN")
    if unknown_labels:
        warnings.append(f"{unknown_labels} 个通话/规则组合尚未复核，标为 UNKNOWN，不参与效果计分；请补充已复核规则或逐项标注。")
    report = {
        "imported": len(calls),
        "skipped": skipped,
        "headers": headers,
        "columns": columns,
        "warnings": warnings[:20],
        "unknownAutoRules": unknown_auto,
        "unknownHumanRules": unknown_human,
        "rulePositives": positives,
        "callsWithViolation": sum(1 for call in calls if call["humanRules"]),
        "unknownLabels": unknown_labels,
        "reviewedLabels": sum(len(call["reviewedRules"]) for call in calls),
        "fullyReviewed": bool(fully_reviewed),
    }
    return calls, report


def cases_for_rule(
    calls: List[Dict[str, Any]],
    rule: Dict[str, Any],
    splits: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """把统一通话表派生为某条规则的二分类样本。"""
    code = str(rule.get("code") or "")
    identity = rule_identity(rule)
    rule_splits = (splits or {}).get(code, {})
    result = []
    for call in calls:
        auto_rules = list(call.get("autoRules") or [])
        human_rules = list(call.get("humanRules") or [])
        if isinstance(call.get("annotations"), dict) or call.get("annotationSchema"):
            annotation = normalize_annotation((call.get("annotations") or {}).get(identity), "unreviewed")
        else:
            annotation = normalize_annotation({"label": "VIOLATION" if code in human_rules else "NORMAL"}, "legacy-assumed-full")
        auto_result = identity in (call.get("autoRuleIds") or []) if "autoRuleIds" in call else code in auto_rules
        case = {
            "id": f"{call.get('id') or call.get('callId')}@{identity}",
            "ruleId": rule.get("id"),
            "callId": call.get("callId"),
            "transcript": call.get("transcript"),
            "autoResult": "VIOLATION" if auto_result else "NORMAL",
            "humanResult": annotation["label"],
            "reviewStatus": annotation["reviewStatus"],
            "reviewer": annotation["reviewer"],
            "reviewReason": annotation["reason"],
            "labelSource": annotation["source"],
            "evidence": annotation.get("evidence") or [],
            "humanEvidence": annotation.get("evidence") or [],
            "humanStages": annotation.get("stages") or {},
            "humanSteps": annotation.get("steps") or {},
            "groupId": call.get("groupId") or "",
            "autoRules": auto_rules,
            "humanRules": human_rules,
            "split": rule_splits.get(call.get("callId"), "TRAIN"),
            "importedAt": call.get("importedAt"),
        }
        if "transcriptComplete" in call:
            case["transcriptComplete"] = call["transcriptComplete"]
        result.append(case)
    return result


def calls_to_csv(calls: List[Dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["通话ID", "通话文本", "自动违规规则", "人工违规规则", "已复核规则", "同源组ID", "标注详情", "转写完整"])
    for call in calls:
        writer.writerow([
            call.get("callId", ""),
            call.get("transcript", ""),
            ";".join(call.get("autoRules") or []),
            ";".join(call.get("humanRules") or []),
            ";".join(call.get("reviewedRules") or []),
            call.get("groupId") or "",
            json.dumps(call.get("annotations") or {}, ensure_ascii=False),
            str(call["transcriptComplete"]).lower() if "transcriptComplete" in call else "",
        ])
    return output.getvalue()


def sample_calls_csv(calls: List[Dict[str, Any]]) -> str:
    return calls_to_csv(calls)
