import copy
import json
import tempfile
import unittest
from pathlib import Path

from app.data_governance import ensure_annotations, refresh_dataset
from app.importer import build_calls, calls_to_csv, cases_for_rule
from app.splits import SPLITS, build_splits, split_summary
from app.store import Store, migrate


def rules():
    return [
        {"id": "rule-a", "code": "R001", "name": "投诉", "regex": "投诉", "semantic": "", "logic": "OR"},
        {"id": "rule-b", "code": "R002", "name": "辱骂", "regex": "辱骂", "semantic": "", "logic": "OR"},
    ]


def legacy_state():
    return {
        "version": 2, "rules": rules(), "calls": [
            {"id": "c1", "callId": "C1", "transcript": "客户：我要投诉", "autoRules": ["R001"], "humanRules": ["R001"]},
            {"id": "c2", "callId": "C2", "transcript": "坐席：您好", "autoRules": [], "humanRules": []},
        ], "settings": {"apiKey": "DO-NOT-SNAPSHOT", "headers": [{"name": "Authorization", "value": "SECRET"}]},
    }


class ExplicitAnnotationTest(unittest.TestCase):
    def test_missing_label_is_unknown_and_explicit_review_is_normal(self):
        content = "通话ID,通话文本,人工违规规则,已复核规则\nC1,客户：投诉,R001,\nC2,坐席：您好,,R001\n".encode()
        calls, report = build_calls("calls.csv", content, rules())
        first = cases_for_rule(calls, rules()[0])
        second = cases_for_rule(calls, rules()[1])
        self.assertEqual([case["humanResult"] for case in first], ["VIOLATION", "NORMAL"])
        self.assertEqual([case["humanResult"] for case in second], ["UNKNOWN", "UNKNOWN"])
        self.assertEqual(report["unknownLabels"], 2)
        self.assertEqual(split_summary(second)["TRAIN"]["negatives"], 0)

    def test_fully_reviewed_import_requires_explicit_flag(self):
        content = "通话文本,人工违规规则\n坐席：正常,\n".encode()
        unreviewed, _ = build_calls("calls.csv", content, rules())
        reviewed, _ = build_calls("calls.csv", content, rules(), fully_reviewed=True)
        self.assertEqual(cases_for_rule(unreviewed, rules()[0])[0]["humanResult"], "UNKNOWN")
        self.assertEqual(cases_for_rule(reviewed, rules()[0])[0]["humanResult"], "NORMAL")

    def test_generated_ids_do_not_collide_between_different_files(self):
        first, _ = build_calls("one.csv", "通话文本,人工违规规则\n内容一,R001\n".encode(), rules())
        second, _ = build_calls("two.csv", "通话文本,人工违规规则\n内容二,R001\n".encode(), rules())
        repeat, _ = build_calls("renamed.csv", "通话文本,人工违规规则\n内容一,R001\n".encode(), rules())
        self.assertNotEqual(first[0]["callId"], second[0]["callId"])
        self.assertEqual(first[0]["callId"], repeat[0]["callId"])

    def test_json_array_rule_codes_and_annotation_evidence_round_trip(self):
        payload = [{"callId": "C1", "transcript": "客户：不同意。坐席：您好", "humanRules": ["R001"],
                    "autoRules": ["R001"], "reviewedRules": ["R001"], "groupId": "customer-1", "transcriptComplete": True,
                    "annotations": {"rule-a": {"label": "VIOLATION", "reviewStatus": "REVIEWED", "reviewer": "甲",
                        "reason": "客户诉求未回应", "evidence": [{"turnId": 1, "quote": "不同意"}],
                        "steps": {"complaint": True}, "stages": {"trigger": True}}}}]
        imported, _ = build_calls("calls.json", json.dumps(payload, ensure_ascii=False).encode(), rules())
        exported = calls_to_csv(imported)
        reimported, _ = build_calls("export.csv", exported.encode(), rules())
        self.assertEqual(imported[0]["annotations"], reimported[0]["annotations"])
        self.assertEqual(reimported[0]["groupId"], "customer-1")
        case = cases_for_rule(reimported, rules()[0])[0]
        self.assertTrue(case["transcriptComplete"])
        self.assertEqual(case["humanEvidence"], [{"turnId": 1, "quote": "不同意"}])
        self.assertEqual(case["humanSteps"], {"complaint": True})
        self.assertEqual(cases_for_rule(reimported, rules()[1])[0]["humanResult"], "UNKNOWN")

    def test_stable_id_wins_over_conflicting_display_code(self):
        payload = [{"transcript": "文本", "humanRules": ["R001"], "annotations": {
            "rule-a": {"label": "NORMAL", "reviewStatus": "REVIEWED"},
            "R001": {"label": "VIOLATION", "reviewStatus": "REVIEWED"},
        }}]
        calls, _ = build_calls("calls.json", json.dumps(payload).encode(), rules())
        self.assertEqual(cases_for_rule(calls, rules()[0])[0]["humanResult"], "NORMAL")

    def test_truncated_transcript_is_incomplete(self):
        payload = [{"transcript": "文" * 20001, "humanRules": [], "transcriptComplete": True}]
        calls, report = build_calls("calls.json", json.dumps(payload).encode(), rules())
        self.assertFalse(calls[0]["transcriptComplete"])
        self.assertTrue(any("截断" in item for item in report["warnings"]))

    def test_exported_deleted_rule_is_not_bound_to_recreated_code(self):
        state = migrate(legacy_state())
        csv_text = calls_to_csv(state["calls"])
        recreated = rules()
        recreated[0]["id"] = "replacement-rule-a"
        imported, report = build_calls("old.csv", csv_text.encode(), recreated)
        self.assertEqual(cases_for_rule(imported, recreated[0])[0]["humanResult"], "UNKNOWN")
        self.assertTrue(any("其他规则 ID" in message for message in report["warnings"]))


class StableIdentityTest(unittest.TestCase):
    def test_rename_preserves_positive_labels_and_export_codes(self):
        state = migrate(legacy_state())
        state["rules"][0]["code"] = "R101"
        refresh_dataset(state, "规则编号重命名")
        self.assertEqual(cases_for_rule(state["calls"], state["rules"][0])[0]["humanResult"], "VIOLATION")
        self.assertEqual(state["calls"][0]["humanRules"], ["R101"])
        self.assertEqual(state["calls"][0]["autoRules"], ["R101"])
        self.assertEqual(state["calls"][0]["annotations"]["rule-a"]["ruleCode"], "R101")
        self.assertIn("R101", calls_to_csv(state["calls"]))

    def test_new_rule_does_not_inherit_legacy_full_review(self):
        state = migrate(legacy_state())
        new_rule = {"id": "new-rule", "code": "R003", "name": "新规则"}
        state["rules"].append(new_rule)
        refresh_dataset(state, "新增规则")
        self.assertEqual({case["humanResult"] for case in cases_for_rule(state["calls"], new_rule)}, {"UNKNOWN"})
        self.assertTrue(state["dataWarnings"])

    def test_unresolved_legacy_code_is_not_automatically_assigned_to_new_rule(self):
        old = legacy_state()
        old["calls"][0]["humanRules"].append("R999")
        state = migrate(old)
        new_rule = {"id": "new-rule", "code": "R999", "name": "新规则"}
        state["rules"].append(new_rule)
        refresh_dataset(state, "创建曾出现但未绑定的规则编号")
        self.assertEqual(cases_for_rule(state["calls"], new_rule)[0]["humanResult"], "UNKNOWN")
        self.assertIn("R999", state["calls"][0]["unresolvedHumanRules"])
        self.assertNotIn("R999", state["calls"][0]["humanRules"])


class GlobalSplitTest(unittest.TestCase):
    def test_all_rules_share_splits_and_group_connections_are_transitive(self):
        calls = [
            {"callId": "A", "transcript": "2026-11-01 12:00:01 坐席：您好", "groupId": "same-customer"},
            {"callId": "B", "transcript": "坐席：不同内容", "groupId": "same-customer"},
            {"callId": "C", "transcript": "2026-11-21 13:00:05 坐席：您好"},
        ]
        splits = build_splits(rules(), calls)
        self.assertEqual(splits["R001"], splits["R002"])
        self.assertEqual(len(set(splits["R001"].values())), 1)

    def test_regular_append_is_stable_and_labels_do_not_influence_split(self):
        calls = [{"callId": str(i), "transcript": "内容" + str(i), "humanRules": ["R001"]} for i in range(100)]
        original = build_splits(rules(), calls)
        for call in calls:
            call["humanRules"] = []
        appended = calls + [{"callId": "new", "transcript": "全新文本"}]
        new = build_splits(rules(), appended)
        self.assertEqual(original["R001"], {key: new["R001"][key] for key in original["R001"]})
        self.assertEqual(set(original["R001"].values()), set(SPLITS))

    def test_few_groups_are_not_duplicated_to_fill_sets(self):
        split = build_splits(rules(), [{"callId": "only", "transcript": "唯一文本"}])
        self.assertEqual(len(split["R001"]), 1)


class DatasetVersionTest(unittest.TestCase):
    def test_snapshot_is_frozen_and_does_not_include_config_secrets(self):
        state = migrate(legacy_state())
        first = copy.deepcopy(state["datasetVersions"][0])
        state["calls"][0]["annotations"]["rule-a"]["label"] = "NORMAL"
        refresh_dataset(state, "人工复核更新")
        self.assertEqual(state["datasetVersions"][0], first)
        self.assertEqual(state["datasetVersion"], 2)
        self.assertNotEqual(first["fingerprint"], state["datasetFingerprint"])
        self.assertNotIn("DO-NOT-SNAPSHOT", json.dumps(state["datasetVersions"]))
        self.assertNotIn("SECRET", json.dumps(state["datasetVersions"]))

    def test_store_backs_up_before_migration_and_does_not_resplit_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.json"
            original = json.dumps(legacy_state(), ensure_ascii=False).encode()
            path.write_bytes(original)
            store = Store(path)
            backups = list(Path(directory).glob("store.v2-backup-*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(store.snapshot()["version"], 3)
            store.mutate(lambda state: state["splits"]["R001"].update({"C1": "TEST"}))
            before = store.snapshot()
            restored = Store(path).snapshot()
            self.assertEqual(before["splits"], restored["splits"])
            self.assertEqual(before["datasetVersions"], restored["datasetVersions"])
            self.assertEqual(len(list(Path(directory).glob("store.v2-backup-*.json"))), 1)

    def test_interrupted_release_keeps_holdout_ownership(self):
        state = migrate(legacy_state())
        state["releases"] = [{"id": "release-1", "status": "EVALUATING"}]
        state["holdouts"] = [{"id": "holdout-1", "usedBy": "release-1"}]
        restored = migrate(state)
        self.assertEqual(restored["releases"][0]["status"], "INTERRUPTED")
        self.assertEqual(restored["holdouts"][0]["usedBy"], "release-1")


if __name__ == "__main__":
    unittest.main()
