"""Ambient-mode intent analysis.

Unlike `meeting_intelligence` (which only extracts action items), this module
runs glass-style: every N final transcripts, it analyses the rolling window
and decides whether to *surface something* to the user — a recent question's
answer, a term that just got dropped, a visible problem, or silence.

The decision hierarchy is ported from pickle-com/glass
(`src/features/listen/summary/summaryService.js` + `prompts/promptTemplates`).
Their insight: the prompt IS the gate — if the LLM sees nothing to say, it
emits the explicit 'stay silent' tier and we don't bother the user.

Output is JSON so we can route tier-3 suggestions into the existing
opencli / browser_task execution paths instead of just speaking text.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from config.settings import GEMINI_API_KEY

try:
    from google import genai as _genai  # type: ignore
    _AVAILABLE = bool(GEMINI_API_KEY)
except ImportError:
    _AVAILABLE = False


_SYSTEM = """\
You are an ambient AI assistant. You see a rolling transcript of what the
user is saying (and people they're talking to) and decide whether to
surface a short helpful note — or stay silent.

Apply this decision hierarchy IN ORDER. Stop at the first tier that fires.

  TIER 1 — answer a recent question.
    If the transcript contains an unanswered factual question the user
    would benefit from an answer to right now. Example triggers:
      - "what is IRR again?"
      - "when did OpenAI raise their Series C?"
      - "who founded YC?"

  TIER 2 — define a recent proper noun or jargon term.
    The term has to be non-obvious and freshly introduced. Prefer to
    surface over staying silent when a real term appeared. Examples that
    should fire tier 2:
      - "constitutional AI" (technique)
      - "Series C" if introduced without context
      - any acronym whose meaning isn't in the transcript

  TIER 3 — suggest a concrete action.
    The conversation implies the user should do something NOW. Emit the
    action ONLY if it maps to one of the two execution paths we ship:

    (A) kind='opencli' — for read-only lookups. action_text is the
        space-separated command. Available adapters:
          hackernews top | hackernews search <query>
          google search <query> | google news
          wikipedia search <query>
          producthunt today
          reddit hot
          linkedin timeline
          arxiv search <query>

    (B) kind='local' — for local desktop actions via AppleScript /
        Spotlight. Set action_text to exactly one of:
          compose_mail           — opens Mail.app with a draft (does
                                    NOT auto-send; user clicks Send).
                                    Emit slots: to, subject, body.
          send_imessage          — sends an iMessage now (IS
                                    auto-sent). Emit slots: contact,
                                    body.
          create_calendar_event  — creates a Calendar event.
                                    Emit slots: title, date, time,
                                    attendees (list, optional).
          find_file              — Spotlight search. Emit slots:
                                    file_query.
          open_url               — opens a URL in default browser.
                                    Emit slots: url.

    If the right answer is ANYTHING outside that list (post on
    LinkedIn, reply to a specific Gmail thread, schedule over
    Calendly, etc.), emit tier 4 instead. Do not invent capabilities
    we don't have — the demo depends on the agent only promising
    what it can deliver.

  TIER 4 — stay silent.
    Nothing above fires, or you would be repeating yourself.

OUTPUT SCHEMA — emit a SINGLE JSON object. No prose. No markdown fences.
Required keys:
  tier:        integer 1-4
  headline:    string, <=100 chars. MUST be non-empty for tiers 1-3,
               MUST be empty string for tier 4.
  detail:      string, 1-2 sentences. MUST be non-empty for tiers 1-2,
               may be empty for tiers 3/4.
  action_kind: string — one of "opencli", "local", "none". Use "none"
               for tiers 1/2/4.
  action_text: string — the concrete command/goal as described above.
               Empty string when action_kind="none".
  action_slots: object — per-goal parameters. Empty object when not
               applicable. Examples:
                 {"to":"hanzi@example.com","subject":"Re: pitch deck",
                  "body":"Got it, will send tonight."}     (compose_mail)
                 {"contact":"Hanzi","body":"running late"} (send_imessage)
                 {"title":"Pitch review","date":"2026-04-20","time":"15:00"}
                 {"file_query":"pitch deck pdf"}           (find_file)
                 {"url":"https://news.ycombinator.com"}    (open_url)

Rules:
- Do NOT repeat anything in the previous analysis if one is provided.
  When in doubt, emit tier 4 over repetition.
- Headlines are user-facing. No filler ("I think…", "It seems…").
- Keep to one JSON object on one line when practical."""


@dataclass
class AmbientAnalysis:
    tier: int = 4
    headline: str = ""
    detail: str = ""
    action: dict[str, Any] | None = None
    raw_json: str = ""

    def should_surface(self) -> bool:
        return self.tier in (1, 2, 3) and bool(self.headline.strip())


_PREVIOUS_JSON_PREAMBLE = "PREVIOUS ANALYSIS (do not repeat):"
_HISTORY_PREAMBLE       = "ROLLING TRANSCRIPT (oldest → newest):"


def _best_effort_json(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Strip markdown fences, then try strict json.loads. If that fails,
    find the outermost {…} block and parse just that. Returns (parsed,
    cleaned_text) or (None, cleaned_text) on complete failure."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned), cleaned
    except json.JSONDecodeError:
        pass
    # Find the first balanced {…} in the cleaned text.
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                snippet = cleaned[start:i + 1]
                try:
                    return json.loads(snippet), snippet
                except json.JSONDecodeError:
                    start = -1
                    continue
    return None, cleaned


def _contacts_block() -> str:
    """List the user's known contacts so Gemini can emit emails / iMessage
    addresses directly in action_slots instead of names that later need
    fuzzy matching. Loaded from config.resources.KNOWN_CONTACTS."""
    try:
        from config.resources import KNOWN_CONTACTS
        items = []
        seen: set[str] = set()
        for name, addr in KNOWN_CONTACTS.items():
            if addr in seen:
                continue  # skip secondary alias rows
            items.append(f"- {name.title()} → {addr}")
            seen.add(addr)
        if items:
            return "KNOWN CONTACTS (prefer emitting the email in action_slots):\n" + "\n".join(items)
    except Exception:
        pass
    return "KNOWN CONTACTS: (none configured)"


def _assemble_prompt(
    history: list[str],
    previous: AmbientAnalysis | None,
    screen_app: str = "",
    screen_window_title: str = "",
) -> str:
    hist_block = "\n".join(f"- {line}" for line in history) or "(empty)"
    prev_block = previous.raw_json if (previous and previous.raw_json) else "(none)"
    screen_block = "(none)"
    if screen_app or screen_window_title:
        screen_block = (
            f"Active app: {screen_app or 'unknown'}\n"
            f"Window title: {screen_window_title or 'unknown'}\n"
            "A screenshot of the user's current screen is attached to this "
            "request. Feel free to reference what's visible on it when it "
            "directly helps answer a question, define a term the user can "
            "see, or suggest an action about what's open."
        )
    return (
        f"{_SYSTEM}\n\n"
        f"{_contacts_block()}\n\n"
        f"{_HISTORY_PREAMBLE}\n{hist_block}\n\n"
        f"CURRENT SCREEN CONTEXT:\n{screen_block}\n\n"
        f"{_PREVIOUS_JSON_PREAMBLE}\n{prev_block}\n\n"
        "Emit your JSON object now."
    )


async def analyse(
    history: list[str],
    previous: AmbientAnalysis | None = None,
    screen_app: str = "",
    screen_window_title: str = "",
    screen_image_bytes: bytes = b"",
) -> AmbientAnalysis:
    """Run one pass of ambient analysis over the rolling transcript. Returns
    a tier-4 (silent) result on any error so the caller never has to
    exception-handle — ambient must fail quiet.

    Screen context is optional. When provided, the prompt tells the model
    that the image + app name + window title are the user's current focus;
    this lets tier 1-3 reference on-screen details ('the Gmail compose
    window', 'the arxiv paper you're reading').
    """
    if not _AVAILABLE or not history:
        return AmbientAnalysis()

    prompt = _assemble_prompt(history, previous, screen_app, screen_window_title)
    loop = asyncio.get_event_loop()

    def _call() -> str:
        client = _genai.Client(api_key=GEMINI_API_KEY)
        contents: list = [prompt]
        if screen_image_bytes:
            # google-genai Part with inline bytes + mime type. Gemini Flash
            # handles JPEG natively; no need to upload first.
            contents.append(
                _genai.types.Part.from_bytes(
                    data=screen_image_bytes,
                    mime_type="image/jpeg",
                )
            )
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=_genai.types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        )
        return (resp.text or "").strip()

    try:
        raw = await loop.run_in_executor(None, _call)
        data, cleaned = _best_effort_json(raw)
        if data is None:
            raise ValueError("no parseable JSON object in response")
    except Exception as e:
        snippet = (raw[:160] if "raw" in locals() else "").replace("\n", " ")
        print(f"[ambient] analyse failed: {e} — raw={snippet!r}")
        return AmbientAnalysis()

    tier = int(data.get("tier", 4))
    headline = str(data.get("headline") or "").strip()
    detail = str(data.get("detail") or "").strip()
    kind = str(data.get("action_kind") or "none").strip().lower()
    action_text = str(data.get("action_text") or "").strip()
    slots = data.get("action_slots") if isinstance(data.get("action_slots"), dict) else {}
    action: dict[str, Any] | None = None
    if kind != "none" and action_text:
        action = {"kind": kind, "text": action_text, "slots": slots or {}}
    return AmbientAnalysis(
        tier=tier if 1 <= tier <= 4 else 4,
        headline=headline,
        detail=detail,
        action=action,
        raw_json=cleaned,
    )
