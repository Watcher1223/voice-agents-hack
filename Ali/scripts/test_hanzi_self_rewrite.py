"""Smoke test for the "me"/"my" -> "hanzi" rewrite + multi-action extraction.

Run from the Ali/ directory:

    cd Ali
    python scripts/test_hanzi_self_rewrite.py

Prints the pre/post-rewrite transcripts. If GEMINI_API_KEY is set, it
also asks `extract_action_items` to split each rewritten transcript into
tasks so we can eyeball that both Hanzi assignments come through with a
file_query for the Q1 Report.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALI_ROOT = HERE.parent
if str(ALI_ROOT) not in sys.path:
    sys.path.insert(0, str(ALI_ROOT))

from intent.pronoun_rewrite import HANZI_SELF_REWRITE_ENABLED, rewrite_self_pronouns  # noqa: E402


FIXTURES = [
    "Can you text me the Q1 Report. Also send it to my email",
    "text me I'll be late",
    "email me the pitch deck and book me a flight to LA",
    "what is my email",
    "find my resume",
    "send my cover letter to hanzi",
]


def _print_rewrites() -> None:
    print(f"HANZI_SELF_REWRITE_ENABLED = {HANZI_SELF_REWRITE_ENABLED}")
    print("=" * 72)
    for raw in FIXTURES:
        rewritten = rewrite_self_pronouns(raw)
        changed = "  (changed)" if rewritten != raw else ""
        print(f"IN : {raw!r}")
        print(f"OUT: {rewritten!r}{changed}")
        print("-" * 72)


async def _print_extraction() -> None:
    try:
        from intent.meeting_intelligence import extract_action_items
    except Exception as exc:
        print(f"[extract] skipped: import error {exc}")
        return

    if not os.environ.get("GEMINI_API_KEY"):
        print("[extract] skipped: GEMINI_API_KEY not set")
        return

    print("\n=== extract_action_items on rewritten transcripts ===")
    for raw in FIXTURES:
        rewritten = rewrite_self_pronouns(raw)
        print(f"\nTranscript: {rewritten!r}")
        try:
            items = await extract_action_items(rewritten, [])
        except Exception as exc:
            print(f"  [extract] error: {exc}")
            continue
        print(json.dumps(items, indent=2, ensure_ascii=False))


def main() -> None:
    _print_rewrites()
    try:
        asyncio.run(_print_extraction())
    except RuntimeError:
        # Some environments already have a running loop; just skip.
        pass


if __name__ == "__main__":
    main()
