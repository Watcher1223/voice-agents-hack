"""Unit test the conversation-mode detector."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from intent.conversation_modes import Mode, detect_mode  # noqa: E402


CASES = [
    (
        "sales call — discovery questions + use case",
        [
            "thanks for hopping on this discovery call",
            "how do you currently handle your deal pipeline",
            "our product is built to solve exactly that",
            "what's your budget for a tool like this",
            "who is the decision maker on your side",
        ],
        Mode.SALES_CALL,
    ),
    (
        "meeting — action items + owners",
        [
            "let's go through the agenda",
            "hanzi will own the pitch deck by Friday",
            "action item: email the investors tonight",
            "we decided that Korin leads the infra track",
            "follow-up next week on the design spec",
        ],
        Mode.MEETING,
    ),
    (
        "generic — casual chat",
        [
            "what did you think of the anthropic paper",
            "it's about constitutional AI",
            "pretty interesting",
            "should we build something like that",
            "anyway what do you want for lunch",
        ],
        Mode.GENERIC,
    ),
    (
        "generic — ambiguous (one sales keyword)",
        [
            "we should discuss pricing later",
            "okay sounds good",
            "anyway what's for dinner",
        ],
        Mode.GENERIC,   # only one hit → shouldn't flip
    ),
    (
        "empty history",
        [],
        Mode.GENERIC,
    ),
]


def main() -> None:
    ok = 0
    for label, history, expected in CASES:
        got = detect_mode(history)
        mark = "✓" if got == expected else "✗"
        print(f"  {mark} {label:50s} expected={expected.value:12s} got={got.value}")
        if got == expected:
            ok += 1
    print(f"\n{ok}/{len(CASES)} cases correct")
    if ok != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
