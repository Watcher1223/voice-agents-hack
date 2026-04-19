"""Smoke test for glass-style ambient analysis.

Feeds fixed transcript windows to `ambient_analysis.analyse` (bypassing the
Deepgram stream) and prints the tier + headline the LLM returns. Use it to
validate the prompt before running the real ambient loop end-to-end.

    cd Ali && .venv/bin/python -m scripts.test_ambient_analysis
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from intent.ambient_analysis import analyse  # noqa: E402


SCENARIOS = [
    (
        "tier 1 — open question",
        [
            "me: so we're thinking about YC's late-stage multiple",
            "them: wait, what's IRR again?",
            "them: i always forget how it differs from ROI",
            "me: yeah it's one of those things",
            "me: okay let's talk about it next round",
        ],
    ),
    (
        "tier 2 — proper noun to define",
        [
            "me: did you see the arxiv paper",
            "them: about what",
            "me: the one from Anthropic on constitutional AI",
            "me: they basically get the model to critique its own outputs",
            "them: interesting",
        ],
    ),
    (
        "tier 3 — suggestable action (open email)",
        [
            "me: did you see the email from Hanzi this morning",
            "them: no i missed it",
            "me: he wants us to send the pitch deck before friday",
            "them: we should probably email him back tonight",
            "me: yeah let's reply to him",
        ],
    ),
    (
        "tier 4 — nothing useful (chitchat)",
        [
            "me: yeah",
            "them: mhm",
            "me: lol",
            "them: haha",
            "me: ok",
        ],
    ),
]


async def main() -> None:
    for label, history in SCENARIOS:
        print(f"\n─── {label} ───")
        result = await analyse(history, previous=None)
        print(f"  tier:     {result.tier}")
        print(f"  headline: {result.headline!r}")
        if result.detail:
            print(f"  detail:   {result.detail[:200]!r}")
        if result.action:
            print(f"  action:   {result.action}")
        print(f"  surface?  {result.should_surface()}")


if __name__ == "__main__":
    asyncio.run(main())
