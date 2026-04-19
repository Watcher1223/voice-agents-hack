"""Tests for intent.grad_calendar_hint."""

from __future__ import annotations

import unittest

from intent.grad_calendar_hint import (
    GRAD_CALENDAR_NOTE,
    append_grad_calendar_note_if_needed,
    transcript_mentions_grad,
)


class TranscriptMentionsGradTests(unittest.TestCase):
    def test_grad_trip_phrase(self):
        self.assertTrue(transcript_mentions_grad("Planning my GRAD TRIP to Japan"))

    def test_standalone_grad_word(self):
        self.assertTrue(transcript_mentions_grad("Tell me about grad"))

    def test_no_gradient(self):
        self.assertFalse(transcript_mentions_grad("Fix the gradient on this button"))

    def test_no_graduate_word(self):
        self.assertFalse(transcript_mentions_grad("When did she graduate"))


class AppendGradCalendarNoteTests(unittest.TestCase):
    def test_appends_when_triggered(self):
        reply = "Sounds fun."
        out = append_grad_calendar_note_if_needed("grad trip ideas", reply)
        self.assertTrue(out.startswith(reply))
        self.assertIn(GRAD_CALENDAR_NOTE, out)
        self.assertEqual(out, f"{reply} {GRAD_CALENDAR_NOTE}")

    def test_no_append_without_trigger(self):
        reply = "Okay."
        self.assertEqual(
            append_grad_calendar_note_if_needed("hello there", reply),
            reply,
        )

    def test_empty_reply_gets_note_only(self):
        out = append_grad_calendar_note_if_needed("grad", "")
        self.assertEqual(out, GRAD_CALENDAR_NOTE)

    def test_idempotent_when_note_already_in_reply(self):
        combined = f"Sure. {GRAD_CALENDAR_NOTE}"
        self.assertEqual(
            append_grad_calendar_note_if_needed("grad", combined),
            combined,
        )


if __name__ == "__main__":
    unittest.main()
