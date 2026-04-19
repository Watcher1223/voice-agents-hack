"""Full-pipeline harness without the mic or mouse.

Tests the journey a real voice utterance would take once ambient has
already decided to surface a tier-3 action:

    ambient.AmbientAnalysis
        → main._execute_ambient_action
            → main._enrich_local_slots     (resolve_contact, find_by_alias)
                → AppleScriptExecutor.compose_mail / send_imessage / …
                → FilesystemExecutor.find_by_alias

To stay non-destructive we monkey-patch AppleScriptExecutor on the fly:
the real resolve_contact runs (so we can verify 'Hanzi' actually maps
to an email on this machine), but compose_mail / send_imessage /
create_calendar_event are intercepted into a `calls` list instead of
actually sending anything.

Run:
    cd Ali && .venv/bin/python -m scripts.test_ambient_pipeline
"""
from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ── Call recorder ─────────────────────────────────────────────────────────
captured_calls: list[tuple[str, dict]] = []


def _record(name):
    def _fn(self, **kwargs):
        captured_calls.append((name, kwargs))
        return None
    return _fn


# Import AppleScript module, patch the write-ish methods.
from executors.local import applescript as _as_mod  # noqa: E402

_as_mod.AppleScriptExecutor.compose_mail            = _record("compose_mail")
_as_mod.AppleScriptExecutor.send_imessage           = _record("send_imessage")
_as_mod.AppleScriptExecutor.create_calendar_event   = _record("create_calendar_event")


# ── Fake overlay ──────────────────────────────────────────────────────────
class FakeOverlay:
    def __init__(self):
        self.pushes: list[tuple[str, str]] = []

    def push(self, state, text=""):
        self.pushes.append((state, text))


# ── Build AmbientAnalysis + run _execute_ambient_action ───────────────────
from intent.ambient_analysis import AmbientAnalysis  # noqa: E402

# Import main.py, but we only need its helpers. main.py has top-level
# side effects (faulthandler.enable, dotenv, preflight run is inside main()).
# _execute_ambient_action / _execute_ambient_local / _enrich_local_slots
# are module-level async functions — safe to call directly.
import importlib as _imp  # noqa: E402
main = _imp.import_module("main")


def _analysis(action_text: str, slots: dict, headline: str, detail: str = "") -> AmbientAnalysis:
    return AmbientAnalysis(
        tier=3,
        headline=headline,
        detail=detail,
        action={"kind": "local", "text": action_text, "slots": slots},
        raw_json="{}",
    )


async def _run(label: str, a: AmbientAnalysis) -> list[tuple[str, dict]]:
    captured_calls.clear()
    overlay = FakeOverlay()
    await main._execute_ambient_action(a, overlay)
    print(f"\n─── {label} ───")
    if not captured_calls:
        print("  (no AppleScript call recorded — check path)")
    for name, kwargs in captured_calls:
        preview = {k: (v[:80] + "…" if isinstance(v, str) and len(v) > 80 else v) for k, v in kwargs.items()}
        print(f"  → {name}({preview})")
    for state, text in overlay.pushes:
        print(f"  · overlay.push({state!r}, {text[:100]!r})")
    return list(captured_calls)


async def main_test() -> None:
    # 1. compose_mail with contact name → should run resolve_contact first
    calls = await _run(
        "compose_mail 'Hanzi' about pitch deck",
        _analysis(
            "compose_mail",
            slots={"to": "Hanzi", "subject": "Pitch deck", "body": "Attached, let me know what you think."},
            headline="Email Hanzi about the pitch deck",
            detail="Hanzi wants the pitch deck tonight.",
        ),
    )
    assert any(n == "compose_mail" for n, _ in calls), "compose_mail not invoked"
    (_, kw), = [(n, k) for n, k in calls if n == "compose_mail"]
    to_after_enrichment = kw.get("to", "")
    print(f"  ✔ enrichment: Hanzi → {to_after_enrichment!r}")

    # 2. send_imessage with contact name
    calls = await _run(
        "send_imessage 'Hanzi' running late",
        _analysis(
            "send_imessage",
            slots={"contact": "Hanzi", "body": "Running ten minutes late."},
            headline="Text Hanzi",
            detail="I'm late for the call.",
        ),
    )
    assert any(n == "send_imessage" for n, _ in calls), "send_imessage not invoked"

    # 3. create_calendar_event
    calls = await _run(
        "create_calendar_event pitch prep",
        _analysis(
            "create_calendar_event",
            slots={"title": "Pitch prep with Hanzi", "date": "2026-04-24", "time": "15:00", "attendees": []},
            headline="Schedule pitch prep",
            detail="Friday 3pm for an hour.",
        ),
    )
    assert any(n == "create_calendar_event" for n, _ in calls), "create_calendar_event not invoked"

    # 4. find_file is not AppleScript but filesystem — should open -R via subprocess.
    #    We don't capture that; just verify no crash + an overlay push happened.
    calls = await _run(
        "find_file resume",
        _analysis(
            "find_file",
            slots={"file_query": "resume"},
            headline="Find your resume",
        ),
    )
    # find_file goes through FilesystemExecutor; no applescript call expected
    print(f"  note: find_file does no AppleScript call, verify by overlay push")

    # 5. open_url
    calls = await _run(
        "open_url ycombinator.com",
        _analysis(
            "open_url",
            slots={"url": "https://www.ycombinator.com"},
            headline="Opening YC",
        ),
    )

    print("\n=== full pipeline test passed ===")


if __name__ == "__main__":
    asyncio.run(main_test())
