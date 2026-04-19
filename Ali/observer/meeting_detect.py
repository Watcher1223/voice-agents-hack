"""Detect whether the user is currently in a live meeting.

Two signals:
  - Active app / window-title patterns (Zoom, Google Meet, Slack
    Huddle, Microsoft Teams, Discord voice, Webex).
  - Explicit override via env var for demos / tests.

When `is_meeting_active()` returns True, the ambient layer should:
  - Keep voice OUTPUT off (no TTS interruptions).
  - Still execute tasks (user said: don't pile them up, just run).
  - Rely on the overlay for confirmations — click not voice.
"""
from __future__ import annotations

import os


_APP_PATTERNS = (
    "zoom",           # zoom.us
    "microsoft teams",
    "google meet",
    "webex",
    "slack",          # Slack Huddles appear in the Slack window title
    "discord",        # voice channel if title mentions it
    "cisco",          # fallback: Cisco Webex
    "facetime",
)

_TITLE_PATTERNS = (
    "zoom meeting",
    "meet -",
    "- meet",
    "huddle",
    "voice channel",
    "— voice",
    "- voice",
    "ongoing call",
    "webex",
    "in a call",
)


def is_meeting_active(app: str, window_title: str) -> bool:
    """True if the active app or window title matches a known meeting
    surface. Case-insensitive substring match."""
    if os.environ.get("VOICE_AGENT_FORCE_MEETING", "0").lower() in {"1", "true", "yes"}:
        return True
    a = (app or "").lower()
    t = (window_title or "").lower()
    if any(p in a for p in _APP_PATTERNS):
        # App alone is a weak signal (Slack is often open without a huddle);
        # require a meeting-ish title for chatty apps.
        if a.strip() in {"slack", "discord"}:
            return any(p in t for p in _TITLE_PATTERNS)
        return True
    if any(p in t for p in _TITLE_PATTERNS):
        return True
    return False
