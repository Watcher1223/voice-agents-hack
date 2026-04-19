#!/usr/bin/env python3
"""
Headless simulation of the PTT find_flights multi-turn flow.

Drives the same code the live UI drives (parse_intent, _parse_when_phrase,
search_flights), but without Qt/audio — so we can batch-test scenarios.

Each scenario is a list of utterances:
  [0] the initial command (e.g. "find a flight from SFO to Tokyo")
  [1..] successive answers the user would speak in response to each
         follow-up prompt ("next weekend", "Boston", etc.)

The simulator walks the same slot-gating logic in main.py:
  origin missing        → "Where are you flying from?"
  destination missing   → "Where to?"
  depart_date missing   → "When do you want to fly?"
                          (answer goes through _parse_when_phrase)
…then calls search_flights and reports the top-3 chips the overlay
would render.
"""

from __future__ import annotations

import asyncio
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from executors.flights import FlightSearchError, search_flights
from intent.parser import _parse_when_phrase, parse_intent
from intent.schema import KnownGoal


SCENARIOS: list[dict] = [
    {
        "name": "all-slots-present (tomorrow, complete)",
        "turns": ["book a flight from Boston to London tomorrow"],
    },
    {
        "name": "missing date → 'next weekend'",
        "turns": ["find a flight from SFO to Tokyo", "next weekend"],
    },
    {
        "name": "missing date → 'tomorrow'",
        "turns": ["find a flight from Ontario to San Francisco", "tomorrow"],
    },
    {
        "name": "missing destination → 'Paris'",
        "turns": ["flight from Boston", "Paris", "tomorrow"],
    },
    {
        "name": "missing origin → 'SFO'",
        "turns": ["flights to Tokyo next weekend", "SFO"],
    },
    {
        "name": "LLM returns raw 'tomorrow' (parser-normalizes)",
        "turns": ["I need to fly from SFO to Tokyo tomorrow"],
    },
    {
        "name": "nonsense date answer → re-prompts",
        "turns": ["flights from SFO to Tokyo", "uhh whenever", "tomorrow"],
    },
    {
        "name": "hallucinated past year ('April 20' → LLM picks 2023)",
        "turns": ["Flight from San Francisco to Ontario on April 20"],
    },
]


def _fmt(slots: dict) -> str:
    return (
        f"origin={slots.get('origin')!r:18}"
        f"destination={slots.get('destination')!r:18}"
        f"depart_date={slots.get('depart_date')!r}"
    )


async def simulate(turns: list[str]) -> tuple[bool, str]:
    """Return (success, log) where log is a multi-line transcript of
    what the overlay/voice would have done during this scenario."""
    lines: list[str] = []
    log = lines.append

    intent = await parse_intent(turns[0])
    log(f"[parse]  goal={intent.goal.value}  {_fmt(intent.slots)}")
    if intent.goal != KnownGoal.FIND_FLIGHTS:
        return False, f"expected FIND_FLIGHTS, got {intent.goal.value}"

    slots = dict(intent.slots or {})
    answer_idx = 1
    today = datetime.date.today()
    ISO = re.compile(r"\d{4}-\d{2}-\d{2}")

    # Normalize initial depart_date if the LLM handed us "tomorrow"
    # verbatim. The real parser now does this, but we re-assert here to
    # mirror main.py's defensive check.
    raw = str(slots.get("depart_date") or "").strip()
    if raw and not ISO.fullmatch(raw):
        iso = _parse_when_phrase(raw, today)
        slots["depart_date"] = iso or ""

    while True:
        if not str(slots.get("origin") or "").strip():
            if answer_idx >= len(turns):
                return False, "ran out of turns before origin was supplied"
            log(f"[ASK]    'Where are you flying from?' → user: {turns[answer_idx]!r}")
            slots["origin"] = turns[answer_idx].strip()
            answer_idx += 1
            continue
        if not str(slots.get("destination") or "").strip():
            if answer_idx >= len(turns):
                return False, "ran out of turns before destination was supplied"
            log(f"[ASK]    'Where to?' → user: {turns[answer_idx]!r}")
            slots["destination"] = turns[answer_idx].strip()
            answer_idx += 1
            continue
        raw = str(slots.get("depart_date") or "").strip()
        if not raw or not ISO.fullmatch(raw):
            if raw and not ISO.fullmatch(raw):
                parsed = _parse_when_phrase(raw, today)
                if parsed:
                    slots["depart_date"] = parsed
                    continue
                log(f"[note]   depart_date {raw!r} not parseable — re-prompting")
                slots["depart_date"] = ""
            if answer_idx >= len(turns):
                return False, "ran out of turns before depart_date was supplied"
            log(f"[ASK]    'When do you want to fly?' → user: {turns[answer_idx]!r}")
            slots["depart_date"] = turns[answer_idx].strip()
            answer_idx += 1
            continue
        break

    log(f"[ready]  slots ready: {_fmt(slots)}")
    try:
        flights = await search_flights(slots)
    except FlightSearchError as exc:
        return False, "\n".join(lines + [f"[kiwi]   FlightSearchError: {exc}"])
    except Exception as exc:
        return False, "\n".join(lines + [f"[kiwi]   {type(exc).__name__}: {exc}"])
    if not flights:
        return False, "\n".join(lines + ["[kiwi]   zero flights returned"])

    top = flights[:3]
    log(f"[kiwi]   {len(flights)} flights; top-3 chips:")
    for f in top:
        price = int(f.get("price") or 0)
        chip = f"${price} {f.get('flyFrom', '')}→{f.get('flyTo', '')}"
        link = f.get("deepLink", "")
        log(f"         • {chip:24s} {link}")
    cheapest = top[0]
    log(
        f"[speak]  'Cheapest is {int(cheapest.get('price') or 0)} dollars "
        f"to {cheapest.get('flyTo')}.'"
    )
    return True, "\n".join(lines)


async def main_async() -> int:
    results: list[tuple[str, bool, str]] = []
    for sc in SCENARIOS:
        print(f"\n{'='*72}")
        print(f"SCENARIO: {sc['name']}")
        print(f"  turns: {sc['turns']}")
        print(f"{'='*72}")
        ok, log = await simulate(sc["turns"])
        print(log)
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
        results.append((sc["name"], ok, log))

    print(f"\n{'='*72}\nSUMMARY")
    for name, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
