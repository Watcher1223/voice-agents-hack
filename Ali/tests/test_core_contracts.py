import unittest

from intent.parser import _rule_based_parse
from intent.schema import IntentObject, KnownGoal
from orchestrator.orchestrator import _path_for_file_action, _resolve_params
from orchestrator.plans import get_plan
from orchestrator.visual_planner import NextAction, _fallback_action


class ParserRuleTests(unittest.TestCase):
    def test_apply_intent_detected(self):
        intent = _rule_based_parse("apply to YC with my resume")
        self.assertEqual(intent.goal, KnownGoal.APPLY_TO_JOB)

    def test_message_intent_extracts_slots(self):
        intent = _rule_based_parse("text Hanzi I'll be ten minutes late")
        self.assertEqual(intent.goal, KnownGoal.SEND_MESSAGE)
        self.assertIn("contact", intent.slots)
        self.assertIn("body", intent.slots)

    def test_calendar_intent_detected(self):
        intent = _rule_based_parse("add calendar event for tomorrow at 3")
        self.assertEqual(intent.goal, KnownGoal.ADD_CALENDAR_EVENT)


class OrchestratorContractTests(unittest.TestCase):
    def test_plans_exist_for_core_goals(self):
        goals = [
            KnownGoal.APPLY_TO_JOB,
            KnownGoal.SEND_MESSAGE,
            KnownGoal.SEND_EMAIL,
            KnownGoal.ADD_CALENDAR_EVENT,
        ]
        for goal in goals:
            steps = get_plan(goal)
            self.assertIsNotNone(steps)
            self.assertGreater(len(steps), 0)

    def test_resolve_params_substitutes_slots(self):
        params = {"contact": "$contact", "body": "$body", "constant": "ok"}
        data = {"contact": "hanzi@example.com", "body": "hello"}
        resolved = _resolve_params(params, data)
        self.assertEqual(resolved["contact"], "hanzi@example.com")
        self.assertEqual(resolved["body"], "hello")
        self.assertEqual(resolved["constant"], "ok")

    def test_visual_action_schema_requires_confirmation_for_irreversible(self):
        action = NextAction(
            action_type="click_text",
            reason="submit",
            expected_outcome="submitted",
            safety_level="irreversible",
            confirm_required=False,
            params={},
        )
        with self.assertRaises(ValueError):
            action.validate()

    def test_visual_fallback_apply_first_step_navigates(self):
        intent = _rule_based_parse("apply to YC with my resume")
        action = _fallback_action(
            intent=intent,
            observation={"scope": "browser", "url": "https://example.com"},
            collected_data={"slots": intent.slots},
            step_index=0,
        )
        self.assertEqual(action.action_type, "navigate")
        self.assertEqual(action.params.get("url"), "https://apply.ycombinator.com")

    def test_fallback_find_file_emits_run_script_reveal(self):
        intent = IntentObject(
            goal=KnownGoal.FIND_FILE,
            target={"type": "file", "value": "taxes"},
            uses_local_data=[],
            requires_browser=False,
            requires_submission=False,
            slots={"file_query": "taxes"},
            raw_transcript="find my taxes",
        )
        action = _fallback_action(
            intent=intent,
            observation={"scope": "desktop"},
            collected_data={"resolved_local_files": {"found": "/tmp/taxes-2024.pdf"}},
            step_index=0,
        )
        self.assertEqual(action.action_type, "run_script")
        self.assertEqual(action.params.get("name"), "reveal_in_finder")
        self.assertEqual(action.params.get("args", {}).get("path"), "/tmp/taxes-2024.pdf")
        action.validate()  # must accept run_script as safe

    def test_fallback_find_file_without_resolution_asks_user(self):
        intent = IntentObject(
            goal=KnownGoal.FIND_FILE,
            target={"type": "file", "value": "taxes"},
            uses_local_data=[],
            requires_browser=False,
            requires_submission=False,
            slots={},
            raw_transcript="find my taxes",
        )
        action = _fallback_action(
            intent=intent,
            observation={"scope": "desktop"},
            collected_data={"resolved_local_files": {}},
            step_index=0,
        )
        self.assertEqual(action.action_type, "ask_user")

    def test_fallback_find_file_completes_after_reveal(self):
        intent = IntentObject(
            goal=KnownGoal.FIND_FILE,
            target={"type": "file", "value": "taxes"},
            uses_local_data=[],
            requires_browser=False,
            requires_submission=False,
            slots={"file_query": "taxes"},
            raw_transcript="find my taxes",
        )
        action = _fallback_action(
            intent=intent,
            observation={"scope": "desktop"},
            collected_data={
                "resolved_local_files": {"found": "/tmp/taxes.pdf"},
                "script_result": {"name": "reveal_in_finder", "returncode": 0},
            },
            step_index=1,
        )
        self.assertEqual(action.action_type, "complete")

    def test_fallback_find_file_aborts_after_first_fruitless_step(self):
        intent = IntentObject(
            goal=KnownGoal.FIND_FILE,
            target={"type": "file", "value": "taxes"},
            uses_local_data=[],
            requires_browser=False,
            requires_submission=False,
            slots={"file_query": "taxes"},
            raw_transcript="find my taxes",
        )
        action = _fallback_action(
            intent=intent,
            observation={"scope": "desktop"},
            collected_data={"resolved_local_files": {}, "slots": intent.slots},
            step_index=1,
        )
        self.assertEqual(action.action_type, "abort")

    def test_action_schema_accepts_run_and_author_script(self):
        for atype in ("run_script", "author_script"):
            action = NextAction(
                action_type=atype,
                reason="r",
                expected_outcome="o",
                safety_level="safe",
                confirm_required=False,
                params={},
            )
            action.validate()


class FileRoleRoutingTests(unittest.TestCase):
    def _stub_fs(self):
        class _StubFs:
            def find_by_alias(self, alias):
                raise FileNotFoundError(f"no alias {alias}")

        return _StubFs()

    def test_path_helper_picks_file_role_from_resolved_map(self):
        data = {
            "resolved_local_files": {"attachment": "/docs/deck.pdf", "resume": "/r.pdf"},
        }
        self.assertEqual(
            _path_for_file_action(data, {"file_role": "attachment"}, self._stub_fs()),
            "/docs/deck.pdf",
        )

    def test_path_helper_falls_back_to_resume_path_for_resume_role(self):
        data = {"resume_path": "/legacy/resume.pdf", "resolved_local_files": {}}
        self.assertEqual(
            _path_for_file_action(data, {}, self._stub_fs()),
            "/legacy/resume.pdf",
        )

    def test_path_helper_falls_through_to_alias(self):
        class _Fs:
            def find_by_alias(self, alias):
                return f"/aliases/{alias}.pdf"

        data = {"resolved_local_files": {}}
        self.assertEqual(
            _path_for_file_action(data, {"file_role": "cover_letter"}, _Fs()),
            "/aliases/cover_letter.pdf",
        )


class ParserRuleHintsTests(unittest.TestCase):
    def test_find_file_intent_detected(self):
        intent = _rule_based_parse("find my 2024 taxes pdf")
        self.assertEqual(intent.goal, KnownGoal.FIND_FILE)
        self.assertIn("file_query", intent.slots)
        self.assertIn("taxes", intent.slots["file_query"].lower())

    def test_email_with_attachment_hint_detected(self):
        intent = _rule_based_parse("email me the Q1 deck attachment")
        self.assertEqual(intent.goal, KnownGoal.SEND_EMAIL)
        self.assertIn("attachment", intent.uses_local_data)


if __name__ == "__main__":
    unittest.main()
