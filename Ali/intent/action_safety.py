"""Safety classifier for ambient tier-3 actions.

Decides whether a suggested action should auto-execute (SAFE — read-only
info lookups, navigation) or require the user's explicit confirmation
(NEEDS_CONFIRM — anything that writes/sends/creates/deletes).

This is the gate that keeps the ambient assistant from sending an email
behind your back. Tier-3 suggestions with NEEDS_CONFIRM are surfaced to
the overlay and the agent listens for "yes" / backtick for ~10s before
quietly dropping the suggestion.

Classification rules:
  • kind='opencli' reading adapters (search, top, hot, news, timeline,
    today, feed, read)              → SAFE
  • kind='local' goals that are local-only reads                → SAFE
    (find_file, open_url)
  • kind='local' goals that send/create                           → NEEDS_CONFIRM
    (send_email, send_message, create_calendar_event, book_flight)
  • kind='browser_task' with read verbs (open, show, tell, check)  → SAFE
  • kind='browser_task' with write verbs (reply, send, post, submit,
    schedule, draft then send)                                   → NEEDS_CONFIRM
  • Anything unknown                                            → NEEDS_CONFIRM
    (safer default — let the user opt in if unsure)
"""
from __future__ import annotations

from typing import Literal

Safety = Literal["safe", "needs_confirm"]

# OpenCLI subcommands that only read data.
_OPENCLI_READ_SUBCOMMANDS = {
    "top", "hot", "search", "news", "timeline", "today",
    "feed", "read", "list", "question", "item", "assets",
    "suggest", "trends",
}

# Local goals that are read-only / navigation.
_LOCAL_READ_GOALS = {"find_file", "open_url"}

# Local goals that write/send.
_LOCAL_WRITE_GOALS = {
    "send_email", "send_message", "compose_mail",
    "create_calendar_event", "add_calendar_event",
    "book_flight", "apply_to_job",
}

# Verbs that signal a destructive browser_task.
_WRITE_VERB_TOKENS = {
    "reply", "send", "post", "submit", "schedule",
    "draft and send", "comment", "delete", "archive",
    "like", "follow", "unfollow", "pay", "buy",
    "book", "purchase", "apply",
}

# Verbs that signal a read-only browser_task.
_READ_VERB_TOKENS = {
    "open", "show", "tell", "check", "read", "find",
    "look", "see", "get", "fetch", "list", "summarize",
    "search", "navigate",
}


def classify(action: dict | None) -> Safety:
    """Return 'safe' or 'needs_confirm' for an ambient action dict.

    `action` is the shape produced by ambient_analysis.analyse — either
    None or {'kind': 'opencli'|'browser_task'|'local', 'text': '...'}."""
    if not action or not isinstance(action, dict):
        return "needs_confirm"

    kind = str(action.get("kind") or "").lower()
    text = str(action.get("text") or "").strip().lower()
    if not kind or not text:
        return "needs_confirm"

    if kind == "opencli":
        # first space-separated token after adapter is the subcommand
        parts = text.split()
        if len(parts) >= 2 and parts[1] in _OPENCLI_READ_SUBCOMMANDS:
            return "safe"
        return "needs_confirm"

    if kind == "local":
        if text in _LOCAL_READ_GOALS:
            return "safe"
        if text in _LOCAL_WRITE_GOALS:
            return "needs_confirm"
        return "needs_confirm"

    if kind == "browser_task":
        # Check write tokens first — "open and send" should still confirm.
        tokens = text.split()
        for w in _WRITE_VERB_TOKENS:
            if w in text:
                return "needs_confirm"
        for r in _READ_VERB_TOKENS:
            if r in tokens[:3]:
                return "safe"
        return "needs_confirm"

    return "needs_confirm"
