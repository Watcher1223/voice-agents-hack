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
You are an ambient AI assistant listening to the user's conversation. Your
job is to turn what's said into things the user actually benefits from —
answers, definitions, or tasks they can review later. Works the same
whether it's a sales call, a planning meeting, or casual chat: you look at
what's on the table and pull out what's useful.

You see a rolling transcript (the user plus whoever they're talking to)
and a screenshot of whatever they have on screen. Use both. On every
turn, pick the single MOST VALUABLE thing to surface, using the tiers
below. Stop at the first tier that fires.

  TIER 1 — answer a recent factual question.
    Triggers when someone says something like "what is IRR again?",
    "when did OpenAI raise Series C?", "who founded YC?". Give the
    concrete answer in `detail`; the headline is the short form.

  TIER 2 — define a recent proper noun, acronym, or jargon.
    Triggers when a non-obvious term appeared and the surrounding
    transcript doesn't already explain it. "Constitutional AI", "LTV
    over CAC", "Series C", "RLHF" if unexplained. Prefer surfacing
    over silence when a real term shows up.

  TIER 3 — capture a concrete task the user (or someone they're
    talking to on the user's behalf) implied. Be AGGRESSIVE about
    tier 3: if the conversation includes a commitment, a follow-up,
    a reminder, a scheduling decision, or a lookup the user will
    benefit from, emit it. Examples that SHOULD fire tier 3:
      "I'll send Hanzi the deck tonight"          → compose_mail
      "remind me to ping Ethan about the API"     → send_imessage
      "let's put lunch with Korin on the calendar for Friday"
                                                  → create_calendar_event
      "pull up the recent news on Stripe"         → opencli google news
      "what's on Hacker News"                     → opencli hackernews top
      "find the pitch deck I worked on last week" → find_file
      "open the YC application page"              → open_url

    Capture OWNER and DEADLINE in action_slots whenever the
    transcript supplies them. Example: if someone says "Hanzi will
    own the deck, due Friday", the compose_mail slots should carry
    to=<hanzi's email> and subject should reference the Friday
    deadline. Missing details are fine — don't invent them, just
    leave the slot empty.

    Emit tier 3 ONLY when the action maps to one of the ways we can
    actually execute:

    (A) kind='opencli' — read-only lookups, run automatically as safe
        actions. action_text is the space-separated command:
          hackernews top | hackernews search <query>
          google search <query> | google news <query>
          wikipedia search <query>
          producthunt today
          reddit hot
          linkedin timeline
          arxiv search <query>

    (B) kind='local' — local desktop actions via AppleScript /
        Spotlight. Set action_text to exactly one of:
          compose_mail           opens Mail.app draft. slots: to,
                                 subject, body. (User clicks Send.)
          send_imessage          sends iMessage. slots: contact, body.
          create_calendar_event  slots: title, date, time, attendees.
          find_file              slots: file_query.
          open_url               slots: url.

    If the right answer is outside that list (post on LinkedIn,
    reply to a specific Gmail thread, update a Notion page), emit
    tier 4. Do not invent capabilities — the user is going to
    approve these cards and we need to actually deliver.

  TIER 4 — stay silent.
    Emit tier 4 only when (a) nothing above genuinely fires, or (b)
    surfacing would repeat what you already surfaced in the
    previous analysis. Do not use tier 4 as a hedge; if a real
    question/term/action is there, surface it.

CONTEXT YOU HAVE
  - A short list of KNOWN CONTACTS with their emails / iMessage
    addresses. Prefer emitting the CONCRETE address in action_slots
    rather than a raw name so execution doesn't need fuzzy matching.
  - The user's active app + window title + a screenshot. If the
    user asks "what does this mean?" while reading a page, reference
    what's on screen. If they say "the pitch deck I have open",
    resolve it via the window title when possible.
  - Fuzzy STT is common — names may be misheard ("hamsi" → "Hanzi",
    "corn" → "Korin"). Resolve to the closest known contact rather
    than rejecting the turn.
  - Turns may be prefixed with a SPEAKER TAG:
      `[Me]`, `[Me-S0]`, `[Me-S1]`      — picked up by the user's mic
                                           (i.e. the user or someone
                                           physically with them).
      `[Remote]`, `[Remote-S0]`, `[Remote-S1]`
                                         — picked up from the Mac's
                                           system audio output, i.e.
                                           the OTHER side of a
                                           FaceTime / Zoom / Meet call.
    Use these to attribute commitments correctly:
      "[Me] I'll email the deck"           → the user committed
      "[Remote] can you send me the link"  → the other party is asking
                                             the user to do something,
                                             still valid tier 3.
      "[Remote-S1] I'll merge the PR"      → remote participant
                                             committed to do it — may
                                             still be useful context
                                             for a reminder, but do NOT
                                             emit a compose_mail
                                             pretending the user
                                             committed.
    `[Me-S0]` is usually the user (whoever owns the laptop talks
    first), `[Me-S1]+` are people in the same room. Do NOT include any
    of these tags in headlines, detail, or action_slots — strip them
    when referencing transcript content.

OUTPUT SCHEMA — a single JSON object. No prose, no markdown fences.
  tier:        integer 1-4
  headline:    <=100 chars, user-facing. Non-empty for tiers 1-3,
               empty string for tier 4.
  detail:      1-2 sentences. Non-empty for tiers 1-2, may be
               empty for tiers 3/4.
  action_kind: "opencli" | "local" | "none". "none" for tiers 1/2/4.
  action_text: the command / action name. "" when kind="none".
  action_slots: object. Examples:
     {"to":"hanzi@example.com","subject":"Re: pitch deck",
      "body":"Sending tonight — out Friday EOD."}     (compose_mail)
     {"contact":"Hanzi","body":"running 10 late"}    (send_imessage)
     {"title":"Pitch review","date":"2026-04-22","time":"15:00",
      "attendees":["korin@yc.com"]}                  (create_calendar_event)
     {"file_query":"pitch deck pdf"}                  (find_file)
     {"url":"https://news.ycombinator.com"}           (open_url)

RULES
  - Do NOT repeat anything in the previous analysis. Prefer tier 4
    over repetition.
  - Headlines are user-facing. No filler ("I think…", "It seems…").
  - Resolve names to the email/phone in KNOWN CONTACTS when possible.
  - Capture the concrete owner + deadline + subject when the
    transcript provides them. Empty is better than hallucinated."""


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
            "A screenshot of the user's current screen is attached. "
            "Reference it when it directly helps answer a question, "
            "define a term the user can see, or suggest an action about "
            "what's open."
        )
    parts = [
        _SYSTEM,
        _contacts_block(),
        f"{_HISTORY_PREAMBLE}\n{hist_block}",
        f"CURRENT SCREEN CONTEXT:\n{screen_block}",
        f"{_PREVIOUS_JSON_PREAMBLE}\n{prev_block}",
        "Emit your JSON object now.",
    ]
    return "\n\n".join(parts)


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
        cfg = _genai.types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )
        # Free-tier gemini-2.5-flash caps at 20 req/day/project; once we
        # exhaust it the API returns 429 RESOURCE_EXHAUSTED. Fall through
        # to gemini-2.5-flash-lite (1500/day free) so the assistant stays
        # live even after the preferred model runs out.
        for model in ("gemini-2.5-flash", "gemini-2.5-flash-lite"):
            try:
                resp = client.models.generate_content(
                    model=model, contents=contents, config=cfg,
                )
                return (resp.text or "").strip()
            except Exception as e:
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    # Try the next model in the list.
                    print(f"[ambient] {model} exhausted — falling back")
                    continue
                raise
        # Both models exhausted — let the outer handler produce a silent tier-4.
        raise RuntimeError("all Gemini models exhausted (429)")

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
