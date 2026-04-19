"""Spoken hint when the user mentions graduation / a grad trip (demo copy)."""

from __future__ import annotations

import re

_GRAD_WORD = re.compile(r"\bgrad\b", re.IGNORECASE)

GRAD_CALENDAR_NOTE = (
    "Just for reference, I'm looking in your calendar and it says you graduate "
    "on the 18th of May."
)


def transcript_mentions_grad(transcript: str) -> bool:
    t = transcript or ""
    if "grad trip" in t.lower():
        return True
    return bool(_GRAD_WORD.search(t))


def append_grad_calendar_note_if_needed(transcript: str, reply: str) -> str:
    if not transcript_mentions_grad(transcript):
        return reply
    base = (reply or "").strip()
    note = GRAD_CALENDAR_NOTE.strip()
    if not base:
        return note
    if note.lower() in base.lower():
        return base
    return f"{base} {note}"
