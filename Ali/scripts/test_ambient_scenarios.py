"""Scenario matrix for ambient analyse().

Feeds each scenario's rolling transcript to the real Gemini-powered
analyser and asserts the tier + action_text + essential slots. Screen
context is intentionally empty — we're validating the classifier's
linguistic behaviour, not the multimodal path (covered elsewhere).

Run:
    cd Ali && .venv/bin/python -m scripts.test_ambient_scenarios
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from intent.ambient_analysis import analyse  # noqa: E402
from intent.action_safety import classify as classify_safety  # noqa: E402


@dataclass
class Scenario:
    label: str
    history: list[str]
    expect_tier: int
    # Accept any of these action_text values (model may pick synonyms).
    expect_action_text_any: list[str] = field(default_factory=list)
    expect_kind: str | None = None
    expect_slots_contain: list[str] = field(default_factory=list)
    expect_safety: str | None = None


SCENARIOS: list[Scenario] = [
    Scenario(
        label="tier 1 — factual question",
        history=[
            "hey i was reading the Anthropic paper",
            "it's about constitutional AI",
            "they train the model against a set of principles",
            "and they call it a constitution",
            "wait what is constitutional AI again",
        ],
        expect_tier=1,
    ),
    Scenario(
        label="tier 2 — jargon term",
        history=[
            "saw the transformer paper yesterday",
            "the one from vaswani",
            "changed deep learning entirely",
            "everyone talks about it",
            "never quite got the attention mechanism",
        ],
        expect_tier=2,
    ),
    Scenario(
        label="compose_mail — she/her",
        history=[
            "I should email Hanzi",
            "she wants the pitch deck",
            "we agreed to reply tonight",
            "need to send that email before Friday",
            "remind me to draft the reply to Hanzi now",
        ],
        expect_tier=3,
        expect_kind="local",
        expect_action_text_any=["compose_mail", "send_email"],
        expect_slots_contain=["body"],
        expect_safety="needs_confirm",
    ),
    Scenario(
        label="send_imessage",
        history=[
            "I'm running late for the call with Hanzi",
            "should text her",
            "tell her I'll be there in ten minutes",
            "she needs to know now",
            "send a quick iMessage to Hanzi saying I'll be ten minutes late",
        ],
        expect_tier=3,
        expect_kind="local",
        expect_action_text_any=["send_imessage", "send_message"],
        expect_slots_contain=["contact", "body"],
        expect_safety="needs_confirm",
    ),
    Scenario(
        label="create_calendar_event",
        history=[
            "Hanzi wants to do pitch prep",
            "she's free Friday afternoon",
            "let's block three pm for an hour",
            "I keep forgetting this stuff",
            "put pitch prep on my calendar for Friday 3pm",
        ],
        expect_tier=3,
        expect_kind="local",
        expect_action_text_any=["create_calendar_event", "add_calendar_event"],
        expect_slots_contain=["title"],
        expect_safety="needs_confirm",
    ),
    Scenario(
        label="find_file",
        history=[
            "where did I put my resume again",
            "I need it for this yc application",
            "it should be in downloads",
            "the pdf from march",
            "find my resume pdf",
        ],
        expect_tier=3,
        expect_kind="local",
        expect_action_text_any=["find_file"],
        expect_slots_contain=["file_query"],
        expect_safety="safe",
    ),
    Scenario(
        label="open_url",
        history=[
            "let me look at the YC site",
            "I want to check the apply page",
            "it's apply.ycombinator.com",
            "need to see the form",
            "open ycombinator for me",
        ],
        expect_tier=3,
        expect_kind="local",
        expect_action_text_any=["open_url"],
        expect_slots_contain=["url"],
        expect_safety="safe",
    ),
    Scenario(
        label="opencli — hackernews top",
        history=[
            "curious what's happening in tech",
            "anything new on the front page",
            "been out of the loop for a day",
            "should check hacker news",
            "show me the top hacker news stories",
        ],
        expect_tier=3,
        expect_kind="opencli",
        expect_action_text_any=["hackernews top"],
        expect_safety="safe",
    ),
    Scenario(
        label="opencli — wikipedia",
        history=[
            "so what does homomorphic encryption even mean",
            "I read something about it",
            "but never dug in",
            "it's some math concept",
            "tell me about homomorphic encryption on wikipedia",
        ],
        expect_tier=3,
        expect_kind="opencli",
        expect_action_text_any=["wikipedia search homomorphic encryption",
                                 "wikipedia search homomorphic"],
        expect_safety="safe",
    ),
    Scenario(
        label="tier 4 — chitchat",
        history=["yeah", "mhm", "ok", "sure", "haha"],
        expect_tier=4,
    ),
]


async def main() -> None:
    passed = failed = 0
    for scn in SCENARIOS:
        try:
            res = await analyse(scn.history, previous=None)
        except Exception as e:
            print(f"  ✗ {scn.label}  — analyse raised {e}")
            failed += 1
            continue

        fails: list[str] = []
        if res.tier != scn.expect_tier:
            fails.append(f"tier {res.tier} (wanted {scn.expect_tier})")
        if scn.expect_kind and (not res.action or res.action.get("kind") != scn.expect_kind):
            fails.append(f"kind={res.action.get('kind') if res.action else None!r} (wanted {scn.expect_kind!r})")
        if scn.expect_action_text_any:
            actual_text = (res.action or {}).get("text", "") if res.action else ""
            hit = any(a.lower() in actual_text.lower() or actual_text.lower().startswith(a.lower())
                      for a in scn.expect_action_text_any)
            if not hit:
                fails.append(f"action_text={actual_text!r} not in {scn.expect_action_text_any}")
        if scn.expect_slots_contain:
            slots = (res.action or {}).get("slots") or {}
            missing = [k for k in scn.expect_slots_contain if not slots.get(k)]
            if missing:
                fails.append(f"missing slots: {missing}")
        if scn.expect_safety:
            got = classify_safety(res.action)
            if got != scn.expect_safety:
                fails.append(f"safety={got} (wanted {scn.expect_safety})")

        if fails:
            print(f"  ✗ {scn.label}")
            for f in fails:
                print(f"     · {f}")
            print(f"     · headline={res.headline[:80]!r}")
            if res.action:
                print(f"     · action  ={res.action}")
            failed += 1
        else:
            summary = res.headline[:70] if res.headline else "(silent)"
            print(f"  ✓ {scn.label:32s} → {summary}")
            passed += 1

    total = passed + failed
    print(f"\n{passed}/{total} scenarios passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
