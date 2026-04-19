"""Unit test the action safety classifier.

Covers the matrix of (kind, text) → expected safety bucket so we don't
regress the 'don't send email without asking' invariant while the
ambient prompt evolves.

    cd Ali && .venv/bin/python -m scripts.test_action_safety
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from intent.action_safety import classify  # noqa: E402


CASES = [
    # (action dict, expected, label)
    ({"kind": "opencli", "text": "hackernews top"}, "safe", "hn top"),
    ({"kind": "opencli", "text": "wikipedia search IRR"}, "safe", "wiki search"),
    ({"kind": "opencli", "text": "linkedin timeline"}, "safe", "linkedin feed"),
    ({"kind": "opencli", "text": "linkedin post hello world"}, "needs_confirm", "linkedin post"),
    ({"kind": "local",   "text": "find_file"}, "safe", "find_file"),
    ({"kind": "local",   "text": "open_url"}, "safe", "open_url"),
    ({"kind": "local",   "text": "send_email"}, "needs_confirm", "send_email"),
    ({"kind": "local",   "text": "send_message"}, "needs_confirm", "send_message"),
    ({"kind": "local",   "text": "create_calendar_event"}, "needs_confirm", "calendar"),
    ({"kind": "browser_task", "text": "open gmail"}, "safe", "browser open"),
    ({"kind": "browser_task", "text": "show my inbox"}, "safe", "browser show"),
    ({"kind": "browser_task", "text": "reply to hanzi saying hi"}, "needs_confirm", "browser reply"),
    ({"kind": "browser_task", "text": "send a message on slack"}, "needs_confirm", "browser send"),
    ({"kind": "browser_task", "text": "post to linkedin"}, "needs_confirm", "browser post"),
    (None, "needs_confirm", "None action"),
    ({}, "needs_confirm", "empty dict"),
    ({"kind": "weird"}, "needs_confirm", "unknown kind"),
]


def main() -> None:
    ok = 0
    for action, expected, label in CASES:
        got = classify(action)
        mark = "✓" if got == expected else "✗"
        print(f"  {mark} {label:28s} expected={expected:14s} got={got}")
        if got == expected:
            ok += 1
    print(f"\n{ok}/{len(CASES)} cases correct")
    if ok != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
