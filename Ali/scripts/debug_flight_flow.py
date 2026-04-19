#!/usr/bin/env python3
"""
End-to-end smoke test for the FIND_FLIGHTS voice flow.

Mirrors exactly what main.py's _handle_transcript does for flight intents:
  transcript -> parse_intent -> search_flights (Kiwi MCP) -> open deeplink.

Skips STT (we pass the transcript in as a string) and skips the tkinter
overlay, but everything downstream of those is the real production code
path.

Usage:
  python3 scripts/debug_flight_flow.py
  python3 scripts/debug_flight_flow.py --transcript "flights from Ontario to SF on April 25"
  python3 scripts/debug_flight_flow.py --no-open   # don't actually launch browser
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from executors.flights import (
    FlightSearchError,
    format_flight_summary,
    search_flights,
)
from intent.parser import parse_intent
from intent.schema import KnownGoal


DEFAULT_TRANSCRIPTS = [
    "flights from Ontario to San Francisco on April 25",
    "book a flight from SFO to Tokyo next weekend",
    "fly from Boston to London tomorrow",
]


async def run_one(transcript: str, open_browser: bool) -> bool:
    print(f"\n{'='*70}\n>>> Transcript: {transcript!r}\n{'='*70}")

    # Stage 1 — parse intent (task creation)
    print("[1/3] Parsing intent...")
    intent = await parse_intent(transcript)
    print(f"      goal  = {intent.goal.value}")
    print(f"      slots = {intent.slots}")
    if intent.goal != KnownGoal.FIND_FLIGHTS:
        print(f"      FAIL: expected FIND_FLIGHTS, got {intent.goal.value}")
        return False

    # Stage 2 — search flights (task execution)
    print("[2/3] Calling Kiwi MCP...")
    try:
        flights = await search_flights(intent.slots)
    except FlightSearchError as e:
        print(f"      FAIL: {e}")
        return False
    if not flights:
        print("      FAIL: zero flights returned")
        return False
    print(f"      got {len(flights)} flights; top 3 by price:")
    for f in flights[:3]:
        print(f"        {format_flight_summary(f)}  ->  {f.get('deepLink')}")

    # Stage 3 — open cheapest deeplink (what the UI would do)
    top = flights[0]
    deeplink = top.get("deepLink")
    print(f"[3/3] Top pick: {format_flight_summary(top)}")
    print(f"      deeplink: {deeplink}")
    if open_browser and deeplink:
        subprocess.run(["open", deeplink], check=False)
        print("      (opened in browser)")
    elif not open_browser:
        print("      (skipped open — --no-open)")
    return True


async def main_async(args: argparse.Namespace) -> int:
    transcripts = [args.transcript] if args.transcript else DEFAULT_TRANSCRIPTS
    results = []
    for t in transcripts:
        ok = await run_one(t, open_browser=not args.no_open and t == transcripts[0])
        results.append((t, ok))

    print(f"\n{'='*70}\nSummary")
    for t, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {t!r}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=str, help="Custom transcript to test")
    parser.add_argument("--no-open", action="store_true", help="Skip opening the browser")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))
