import asyncio
import unittest

from pydantic import ValidationError

from app.capabilities import (
    approved_knowledge, parse_transcript, parse_turns, regex_evidence,
    role_context_valid, validate_capability_result,
)
from app.llm import LLMClient, LLMError, build_judge_prompt, judge_semantic
from app.models import EvaluationPolicy, RuleInput
from app.prompts import DEFAULT_SEMANTIC_PROMPT


RULE = {"code": "COMPLAINT", "name": "投诉须回应", "semantic": "客户投诉后坐席未针对诉求回应",
        "actor": "BOTH", "workflow": "COMPLAINT_RESPONSE", "responseWindowTurns": 3,
        "responseRequirement": "解释未经同意办理的问题，并提供解决方法"}
TRANSCRIPT = "[00:01] 客户：没有征求我的同意为什么就办了？\n00:04 客服：是的，你需要去网点。\n客户：我是问为什么没有征求我的同意。\n坐席：您去网点就行。"


def verdict(adequate=False, timely=False, violation=True):
    return {"violation": violation, "reason": "未回应未经同意办理的具体诉求", "triggerTurnId": "T1", "responseTurnId": "T2",
            "steps": {"complaintIntent": True, "adequateResponse": adequate, "timelyResponse": timely},
            "evidence": [{"turnId": "T1", "role": "CUSTOMER", "quote": "没有征求我的同意为什么就办了？"},
                         {"turnId": "T2", "role": "AGENT", "quote": "是的，你需要去网点。"}]}


class CapabilityModelTest(unittest.TestCase):
    def test_defaults_and_complaint_requires_semantic(self):
        rule = RuleInput(code="r1", name="检查", regex="违规词")
        self.assertEqual(rule.actor, "AGENT")
        self.assertEqual(rule.workflow, "SIMPLE")
        self.assertEqual(EvaluationPolicy().minRecall, .9)
        with self.assertRaises(ValidationError):
            RuleInput(code="r1", name="检查", regex="投诉", workflow="COMPLAINT_RESPONSE")

    def test_roles_timestamps_and_original_offsets(self):
        turns = parse_turns(TRANSCRIPT)
        self.assertEqual([turn["role"] for turn in turns], ["CUSTOMER", "AGENT", "CUSTOMER", "AGENT"])
        for index, turn in enumerate(turns):
            self.assertEqual(TRANSCRIPT[turn["start"]:turn["end"]], turn["text"])
            self.assertEqual(turn["sourceLine"], index + 1)
        agent = regex_evidence({"actor": "AGENT", "regex": "征求"}, TRANSCRIPT)
        customer = regex_evidence({"actor": "CUSTOMER", "regex": "征求"}, TRANSCRIPT)
        self.assertIsNone(agent)
        self.assertEqual(customer["turnId"], "T1")
        self.assertEqual(TRANSCRIPT[customer["start"]:customer["end"]], "征求")

    def test_unparseable_actor_context_does_not_become_normal(self):
        self.assertFalse(role_context_valid({"actor": "AGENT"}, "没有角色的转写"))
        self.assertTrue(role_context_valid({}, "没有角色的旧版文本"))
        self.assertFalse(parse_transcript("未归属开场\n客户：你好")["valid"])
        self.assertFalse(parse_transcript("客户：你好 客服：你好")["valid"])

    def test_knowledge_only_from_approved_server_projection(self):
        rule = {"knowledgeIds": ["draft"], "knowledge": [{"content": "不能直接使用"}],
                "_approvedKnowledge": [{"id": "ok", "content": "知识", "status": "APPROVED"},
                                       {"id": "draft", "content": "草稿", "status": "DRAFT"}]}
        self.assertEqual([item["id"] for item in approved_knowledge(rule)], ["ok"])


class EvidenceValidationTest(unittest.TestCase):
    def test_complaint_no_response_has_verifiable_steps_and_offsets(self):
        result = validate_capability_result(verdict(), {"transcript": TRANSCRIPT}, RULE)
        self.assertTrue(result["violation"])
        self.assertTrue(result["evidenceValid"])
        self.assertFalse(result["reviewRequired"])
        self.assertTrue(result["responseWindow"]["complete"])
        for item in result["evidence"]:
            self.assertEqual(TRANSCRIPT[item["start"]:item["end"]], item["quote"])

    def test_wrong_role_or_invented_quote_requests_review(self):
        for field, value in (("role", "AGENT"), ("quote", "不存在的原话"), ("turnId", "T999")):
            data = verdict()
            data["evidence"][0][field] = value
            result = validate_capability_result(data, {"transcript": TRANSCRIPT}, RULE)
            self.assertIsNone(result["violation"])
            self.assertTrue(result["reviewRequired"])

    def test_no_following_window_is_unknown_until_complete_confirmed(self):
        text = "客户：没有征求我的同意为什么就办了？\n客服：是的，你需要去网点。"
        result = validate_capability_result(verdict(), {"transcript": text}, RULE)
        self.assertIsNone(result["violation"])
        result = validate_capability_result(verdict(), {"transcript": text, "transcriptComplete": True}, RULE)
        self.assertTrue(result["violation"])

    def test_valid_response_in_window_can_be_confirmed_before_call_end(self):
        text = "客户：没有征求我的同意为什么就办了？\n客服：是的，你需要去网点。"
        # 这里只测证据/时序契约；回应的语义适当性由经过校准的判定模型负责。
        result = validate_capability_result(verdict(True, True, False), {"transcript": text}, RULE)
        self.assertFalse(result["violation"])
        self.assertTrue(result["evidenceValid"])

    def test_response_before_complaint_is_invalid(self):
        data = verdict(True, True, False)
        data["triggerTurnId"], data["responseTurnId"] = "T2", "T1"
        data["evidence"][0]["turnId"], data["evidence"][1]["turnId"] = "T2", "T1"
        text = "客服：是的，你需要去网点。\n客户：没有征求我的同意为什么就办了？"
        result = validate_capability_result(data, {"transcript": text, "transcriptComplete": True}, RULE)
        self.assertIsNone(result["violation"])
        self.assertIn("投诉之前", result["reason"])

    def test_late_response_cannot_claim_timely(self):
        rule = dict(RULE, responseWindowTurns=1)
        data = verdict(True, True, False)
        data["responseTurnId"] = "T4"
        data["evidence"][1] = {"turnId": "T4", "role": "AGENT", "quote": "您去网点就行。"}
        result = validate_capability_result(data, {"transcript": TRANSCRIPT}, rule)
        self.assertIsNone(result["violation"])
        data["steps"]["timelyResponse"], data["violation"] = False, True
        result = validate_capability_result(data, {"transcript": TRANSCRIPT}, rule)
        self.assertTrue(result["violation"])

    def test_no_complaint_requires_customer_evidence(self):
        data = {"violation": False, "steps": {"complaintIntent": False, "adequateResponse": None, "timelyResponse": None},
                "evidence": [{"turnId": "T1", "role": "CUSTOMER", "quote": "谢谢"}]}
        result = validate_capability_result(data, {"transcript": "客户：谢谢\n客服：不客气"}, RULE)
        self.assertFalse(result["violation"])
        self.assertTrue(result["evidenceValid"])

    def test_truncated_transcript_is_unknown_even_if_model_claims_normal(self):
        result = validate_capability_result(verdict(True, True, False), {"transcript": TRANSCRIPT, "truncated": True}, RULE)
        self.assertIsNone(result["violation"])


class JudgeContractTest(unittest.TestCase):
    def test_simple_legacy_definite_tuple_and_explicit_unknown(self):
        class Client:
            def __init__(self, result):
                self.result = result
            async def chat_json(self, *args, **kwargs):
                return self.result
        result = asyncio.run(judge_semantic(Client({"violation": False, "reason": "正常"}),
                                            {"transcript": "坐席：您好"}, {}, DEFAULT_SEMANTIC_PROMPT, []))
        self.assertEqual(result, (False, "正常"))
        result = asyncio.run(judge_semantic(Client({"violation": None}), {"transcript": "坐席：您好"}, {}, DEFAULT_SEMANTIC_PROMPT, []))
        self.assertIsNone(result["violation"])
        with self.assertRaises(LLMError):
            asyncio.run(judge_semantic(Client({"reason": "missing"}), {"transcript": "测试"}, {}, DEFAULT_SEMANTIC_PROMPT, []))

    def test_structured_verdict_contract_also_used_without_reason(self):
        class Client:
            async def chat_json(self, system, user, **kwargs):
                self.user, self.kwargs = user, kwargs
                return verdict()
        client = Client()
        result = asyncio.run(judge_semantic(client, {"transcript": TRANSCRIPT}, RULE, DEFAULT_SEMANTIC_PROMPT, [RULE], False))
        self.assertTrue(result["violation"])
        self.assertIn("evidence", client.user)
        self.assertGreater(client.kwargs["max_tokens"], 300)

    def test_bad_transcript_does_not_call_model(self):
        class Client:
            async def chat_json(self, *args, **kwargs):
                raise AssertionError("unparseable context should not call model")
        result = asyncio.run(judge_semantic(Client(), {"transcript": "无角色文本"}, RULE, DEFAULT_SEMANTIC_PROMPT, [RULE]))
        self.assertIsNone(result["violation"])

    def test_actor_prompt_has_no_hardcoded_agent_only_rule(self):
        prompt = build_judge_prompt({"actor": "CUSTOMER", "semantic": "识别客户投诉"}, "客户：我要投诉")
        self.assertIn("检查角色：客户", prompt)
        self.assertNotIn("只有坐席", prompt)
        self.assertNotIn("证据不足，按未违规", DEFAULT_SEMANTIC_PROMPT)

    def test_custom_verdict_path_preserves_steps_and_null(self):
        client = object.__new__(LLMClient)
        client.protocol, client.content_path = "custom", "result"
        client.config = {"violationPath": "result.violation"}
        data = verdict()
        result = client._extract({"result": data}, True)
        self.assertEqual(result["steps"], data["steps"])
        self.assertEqual(result["evidence"], data["evidence"])
        result = client._extract({"result": {"violation": None}}, True)
        self.assertIsNone(result["violation"])


if __name__ == "__main__":
    unittest.main()
