import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate.zac2026.agent import run_sample
from lifemate.zac2026.guard import Guard


class GuardToolLoopTest(unittest.TestCase):
    def test_rejected_reference_is_not_executed_and_model_can_recover(self):
        specs = {
            "contacts.search": {"name": "contacts.search", "kind": "read", "parameters": {
                "properties": {"query": {"type": "string"}}}},
            "phone.call": {"name": "phone.call", "kind": "write", "parameters": {
                "x-one-of": [["contact_id"], ["phone_number"]],
                "properties": {"contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                               "phone_number": {"type": "string"}}}},
        }
        row = {
            "id": "recovery", "audio": "request", "history": [],
            "context": {"current_time": "2026-11-12T10:00:00+07:00",
                        "installed_apps": {"contacts": {"tools": ["contacts.search"]},
                                           "phone": {"tools": ["phone.call"]}}},
        }
        actual_calls = []
        validated = []

        def validate(name, args):
            validated.append((name, args))
            return {"ok": True}

        def call(name, args):
            actual_calls.append({"name": name, "arguments": args})
            if name == "contacts.search":
                return {"ok": True, "contacts": [{"id": "contact_123", "name": "Lan"}]}
            return {"ok": True}

        env = SimpleNamespace(validate=validate, call=call, get_tool_calls=lambda: actual_calls,
                              get_state_hash=lambda: "hash")
        fewshot = SimpleNamespace(topk=lambda text, count: [])
        steps = [
            ("phone__call", {"contact_id": "contact_999"}),
            ("contacts__search", {"query": "Lan"}),
            ("phone__call", {"contact_id": "contact_123"}),
            (None, {}),
        ]
        feedback = []

        def chat(messages, tools, deadline):
            if messages[-1]["role"] == "tool":
                feedback.append((messages[-1]["tool_call_id"], json.loads(messages[-1]["content"])))
            name, args = steps.pop(0)
            tool_calls = [SimpleNamespace(id=str(len(steps)), function=SimpleNamespace(
                name=name, arguments=json.dumps(args)))] if name else []
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=tool_calls))])

        run = run_sample(row, env, specs, lambda audio: "gọi Lan", fewshot, {}, chat, 0, 10)
        self.assertNotIn("error", run)
        self.assertEqual(validated, [("contacts.search", {"query": "Lan"}),
                                     ("phone.call", {"contact_id": "contact_123"})])
        self.assertEqual([c["name"] for c in run["tool_calls"]], ["contacts.search", "phone.call"])
        self.assertEqual(feedback[0], ("3", {
            "ok": False,
            "error": "Tham số contact_id='contact_999' chưa được xác thực. Hãy gọi tool tìm kiếm/danh sách "
                     "của ứng dụng contacts trước để lấy ID chính xác.",
        }))
        self.assertEqual(feedback[-1][1], {"ok": True})

    def test_write_and_delete_invalidate_both_read_caches_for_one_app(self):
        specs = {"alarm.list": {"kind": "read"}, "alarm.update": {"kind": "write"},
                 "alarm.delete": {"kind": "delete"}, "alarm_backup.list": {"kind": "read"}}
        row = {
            "context": {"current_time": "2026-11-12T10:00:00+07:00"},
            "history": [{"tool_calls": [{"name": name, "arguments": {}} for name in
                                        ("alarm.list", "alarm_backup.list")],
                         "tool_results": [{"ok": True}, {"ok": True}]}],
        }
        guard = Guard(specs, row, {}, "")
        for name in ("alarm.update", "alarm.delete"):
            with self.subTest(name=name):
                guard.record_result("alarm.list", {"filter": "active"}, {"ok": True, "alarms": [1]})
                guard.record_result("alarm_backup.list", {"filter": "active"}, {"ok": True})
                self.assertIsNotNone(guard.cached("alarm.list", {"filter": "active"}))
                guard.record_result(name, {"alarm_id": "alm_1"}, {"ok": True})
                self.assertIsNone(guard.cached("alarm.list", {}))
                self.assertIsNone(guard.cached("alarm.list", {"filter": "active"}))
                self.assertIsNone(guard.cached(name, {"alarm_id": "alm_1"}))
                self.assertIsNotNone(guard.cached("alarm_backup.list", {}))
                self.assertIsNotNone(guard.cached("alarm_backup.list", {"filter": "active"}))

    def test_history_read_before_write_is_not_a_cache_hit(self):
        specs = {"alarm.list": {"kind": "read"}, "alarm.update": {"kind": "write"}}
        row = {
            "context": {"current_time": "2026-11-12T10:00:00+07:00"},
            "history": [{"tool_calls": [{"name": name, "arguments": {}} for name in specs],
                         "tool_results": [{"ok": True}, {"ok": True}]}],
        }
        self.assertIsNone(Guard(specs, row, {}, "").cached("alarm.list", {}))


if __name__ == "__main__":
    unittest.main()
