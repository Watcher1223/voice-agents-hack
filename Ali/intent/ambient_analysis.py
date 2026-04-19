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
import shutil
from dataclasses import dataclass, field
from typing import Any

from config.settings import (
    AMBIENT_ANALYSE_RETRIES,
    AMBIENT_CACTUS_FALLBACK,
    AMBIENT_CACTUS_MODEL,
    AMBIENT_CACTUS_TIMEOUT_S,
    GEMINI_API_KEY,
)

_CACTUS_CLI = shutil.which("cactus")

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

  TIER 3 — suggest one or more concrete actions.
    The conversation implies the user should do something NOW. Emit
    every distinct action you hear. A single utterance like "email
    Hanzi that I want to book a trip to Hawaii" naturally contains TWO
    distinct actions:
      1. compose_mail → Hanzi: "I want to book a trip to Hawaii"
      2. browser_task → "book a flight to Hawaii"
    Both MUST appear in the `actions` array below — do not collapse
    them into one row, and do not drop the second one.

    Available action kinds:

    (A) kind='opencli' — read-only lookups. `text` is the space-
        separated command. Available adapters:
          hackernews top | hackernews search <query>
          google search <query> | google news
          wikipedia search <query>
          producthunt today
          reddit hot
          linkedin timeline
          arxiv search <query>

    (B) kind='local' — local desktop actions via AppleScript /
        Spotlight. `text` must be exactly one of:
          compose_mail           — opens Mail.app with a draft (does
                                    NOT auto-send; user clicks Send).
                                    slots: to, subject, body.
          send_imessage          — sends an iMessage now (IS
                                    auto-sent). slots: contact, body.
          create_calendar_event  — creates a Calendar event.
                                    slots: title, date, time,
                                    attendees (list, optional).
          find_file              — Spotlight search.
                                    slots: file_query.
          open_url               — opens a URL in default browser.
                                    slots: url.

    (C) kind='browser_task' — a real-world task best handled by a
        browser agent we will run later. Use this for booking
        flights/hotels, applying to jobs, anything that needs a real
        web session. `text` should be a natural-language goal (no
        slots required). Examples:
          text="book a flight from SFO to Cancun for next Tuesday"
          text="find a one-bedroom Airbnb in Lisbon under $150/night"
          text="apply to the Anthropic product engineer role"

    If an action doesn't map to A/B/C, simply omit it — do not
    fabricate capabilities. But if *any* action fits, emit the
    analysis as tier-3, even if other mentioned actions don't.

  TIER 4 — stay silent.
    Nothing above fires, or every candidate action is already on the
    checklist.

OUTPUT SCHEMA — emit a SINGLE JSON object. No prose. No markdown fences.
Required keys:
  tier:        integer 1-4
  headline:    string, <=100 chars. Short summary of what's happening.
               MUST be non-empty for tiers 1-3, empty string for tier 4.
  detail:      string, 1-2 sentences. MUST be non-empty for tiers 1-2,
               may be empty for tiers 3/4.
  actions:     array — ZERO or MORE action objects. Each object is
                 {"kind": "opencli"|"local"|"browser_task",
                  "text": "<command-or-goal>",
                  "slots": { ... per-kind, see above ... },
                  "label": "<<=60 char user-facing task label>"}
               The `label` is what the user sees on their checklist
               row. Use a verb phrase like "Email Hanzi about Hawaii
               trip" or "Book flight to Hawaii". Omit `actions` (or
               leave it empty) for tiers 1/2/4.

Example for "email Hanzi that I want to book a grad trip to Hawaii":
{
  "tier": 3,
  "headline": "Email Hanzi + book Hawaii flight",
  "detail": "",
  "actions": [
    {"kind":"local","text":"compose_mail",
     "slots":{"to":"Hanzi","subject":"Grad trip to Hawaii",
              "body":"I want to book a grad trip to Hawaii."},
     "label":"Email Hanzi about Hawaii grad trip"},
    {"kind":"browser_task","text":"book a flight to Hawaii for the grad trip",
     "slots":{}, "label":"Book flight to Hawaii"}
  ]
}

Rules:
- TASKS ALREADY ON CHECKLIST lists rows the user has not yet executed.
  Do not re-emit those exact items. Any *new* distinct action still
  surfaces.
- PREVIOUS SURFACED ANALYSIS is for context; avoid re-emitting the
  same action but feel free to surface *new* ones.
- Headlines are user-facing. No filler.
- Keep output compact JSON."""


@dataclass
class AmbientAnalysis:
    tier: int = 4
    headline: str = ""
    detail: str = ""
    # Zero or more actions. A single utterance can produce multiple
    # (e.g. "email Hanzi + book a flight" → compose_mail + browser_task).
    actions: list[dict[str, Any]] = field(default_factory=list)
    raw_json: str = ""

    @property
    def action(self) -> dict[str, Any] | None:
        """Back-compat accessor — first action, or None. Callers that
        iterate use .actions; the checklist-execution path recreates an
        AmbientAnalysis per Task with a single action in the list."""
        return self.actions[0] if self.actions else None

    def should_surface(self) -> bool:
        return self.tier in (1, 2, 3) and bool(self.headline.strip())


_PREVIOUS_JSON_PREAMBLE = "PREVIOUS SURFACED ANALYSIS (avoid repeating this exact action only):"
_CHECKLIST_PREAMBLE = (
    "TASKS ALREADY ON THE USER'S CHECKLIST — pending, not yet executed "
    "(do not duplicate these rows; new distinct actions still surface):"
)
_HISTORY_PREAMBLE = "ROLLING TRANSCRIPT (oldest → newest):"


def _is_quota_error(exc: BaseException) -> bool:
    """Hard quota exhaustion — retries won't help, escalate to Cactus."""
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    needles = (
        "resource_exhausted",
        "resource exhausted",
        "quota exceeded",
        "quotafailure",
        "free_tier_requests",
        "exceeded your current quota",
    )
    if any(n in msg for n in needles):
        return True
    if "resourceexhausted" in name:
        return True
    return False


async def _run_cactus_analyse(prompt: str) -> str:
    """Invoke `cactus run <model> --prompt <prompt>` and return the
    Assistant block from stdout. Mirrors the pattern in
    intent/parser.py and executors/local/disk_index/answer.py so a
    single codepath in the Cactus workflow stays consistent."""
    if not _CACTUS_CLI:
        return ""
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            _CACTUS_CLI,
            "run",
            AMBIENT_CACTUS_MODEL,
            "--prompt",
            prompt,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=b"exit\n"),
            timeout=AMBIENT_CACTUS_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        print(
            f"[ambient] cactus fallback timed out after "
            f"{AMBIENT_CACTUS_TIMEOUT_S:.0f}s (model={AMBIENT_CACTUS_MODEL})"
        )
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        return ""
    except (OSError, asyncio.CancelledError) as exc:
        print(f"[ambient] cactus subprocess failed: {exc}")
        return ""
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="ignore").strip()[:240]
        print(f"[ambient] cactus rc={proc.returncode} stderr={err}")
        return ""
    raw = stdout.decode("utf-8", errors="ignore")
    return _extract_cactus_reply(raw)


def _extract_cactus_reply(output: str) -> str:
    """Pull the 'Assistant:' block out of cactus's chat-style stdout.
    Duplicated here (rather than imported) so the module doesn't take a
    hard dep on disk_index, which pulls in sentence-transformers."""
    if not output:
        return ""
    marker = "Assistant:"
    idx = output.find(marker)
    tail = output[idx + len(marker):] if idx >= 0 else output
    lines: list[str] = []
    for raw in tail.splitlines():
        stripped = raw.strip()
        # Stop at the token-stats line, e.g. "[66 tokens | latency: …]".
        if stripped.startswith("[") and "tokens" in stripped:
            break
        if stripped.lower().startswith("you:"):
            break
        lines.append(raw)
    return "\n".join(lines).strip()


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """503 / overload / rate-limit — worth sleeping and retrying."""
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    needles = (
        "503",
        "429",
        "unavailable",
        "overloaded",
        "deadline exceeded",
        "try again",
        "resource exhausted",
        "rate limit",
        "temporarily",
        "high demand",
        "503 unavailable",
    )
    if any(n in msg for n in needles):
        return True
    if any(n in name for n in ("resourceexhausted", "serviceunavailable", "deadline", "aborted")):
        return True
    return False


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


def _format_pending_checklist_block() -> str:
    """Surface pending checklist rows so Gemini won't re-suggest duplicates
    but still allows *new* distinct tier-3 actions."""
    try:
        from observer.task_checklist import checklist

        pending = checklist().pending()
        if not pending:
            return "(none)"
        return "\n".join(f"- {t.label[:200]}" for t in pending[:20])
    except Exception:
        return "(none)"


def _assemble_prompt(
    history: list[str],
    previous: AmbientAnalysis | None,
    screen_app: str = "",
    screen_window_title: str = "",
) -> str:
    hist_block = "\n".join(f"- {line}" for line in history) or "(empty)"
    prev_block = previous.raw_json if (previous and previous.raw_json) else "(none)"
    checklist_block = _format_pending_checklist_block()
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
        f"{_HISTORY_PREAMBLE}\n{hist_block}\n\n"
        f"CURRENT SCREEN CONTEXT:\n{screen_block}\n\n"
        f"{_PREVIOUS_JSON_PREAMBLE}\n{prev_block}\n\n"
        f"{_CHECKLIST_PREAMBLE}\n{checklist_block}\n\n"
        "Emit your JSON object now."
    )


# Compact prompt for the small-model fallback (1B Gemma class). No
# few-shot example — smaller models parrot names and places from
# examples into outputs. Skips the screen block (Cactus is text-only).
_CACTUS_SYSTEM = """\
Read the CURRENT TRANSCRIPT at the bottom and output ONE JSON object
describing what (if anything) the user just said they need to do.

Rules:
1. Output ONLY valid JSON. No prose, no markdown, no fences.
2. Only use names, places, topics, and words that appear in the
   CURRENT TRANSCRIPT. Do not invent details. Do not use names from
   these instructions.
3. If the transcript describes a task the user should do, set tier=3.
   Otherwise set tier=4.

JSON keys:
  tier:      integer, 3 or 4.
  headline:  short summary, empty string if tier=4.
  detail:    empty string.
  actions:   array of action objects (empty array if tier=4).

Each action object:
  {"kind": <"local"|"opencli"|"browser_task">,
   "text": <see below>,
   "slots": <see below>,
   "label": <short verb phrase for the task>}

kind="local" — text is one of:
  compose_mail           slots: {"to":<name>, "subject":<topic>, "body":<message>}
  send_imessage          slots: {"contact":<name>, "body":<message>}
  create_calendar_event  slots: {"title":<what>, "date":<YYYY-MM-DD>, "time":<HH:MM>}
  find_file              slots: {"file_query":<query>}
  open_url               slots: {"url":<url>}

kind="opencli" — text is an opencli command (hackernews top, google search X, wikipedia search Y, arxiv search Z). slots: {}.

kind="browser_task" — text is a natural-language web goal (book flight, apply to job, research company). slots: {}.

A single sentence can contain multiple actions. Each distinct action
becomes its own entry in the actions array. Only include an action if
the transcript actually implies it.

If an action matching the transcript already appears under TASKS
ALREADY ON CHECKLIST, do not re-emit it."""


def _assemble_cactus_prompt(
    history: list[str], previous: AmbientAnalysis | None
) -> str:
    # Only the most recent few turns — the 1B model tends to over-weight
    # older lines when the context gets long, and the interesting signal
    # is always in the last utterance or two.
    tail = history[-4:] if history else []
    hist_block = "\n".join(f"- {line}" for line in tail) or "(empty)"
    checklist_block = _format_pending_checklist_block()
    return (
        f"{_CACTUS_SYSTEM}\n\n"
        f"TASKS ALREADY ON CHECKLIST:\n{checklist_block}\n\n"
        f"CURRENT TRANSCRIPT:\n{hist_block}\n\n"
        "JSON output:"
    )


async def _analyse_with_gemini(
    prompt: str, screen_image_bytes: bytes
) -> tuple[dict[str, Any] | None, str, BaseException | None]:
    """Call Gemini with retries. Returns (parsed, cleaned_text, last_exc).
    On quota-exhausted / repeated transient errors the caller escalates
    to Cactus."""
    loop = asyncio.get_event_loop()

    def _call() -> str:
        client = _genai.Client(api_key=GEMINI_API_KEY)
        contents: list = [prompt]
        if screen_image_bytes:
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

    raw = ""
    last_exc: BaseException | None = None
    for attempt in range(AMBIENT_ANALYSE_RETRIES):
        try:
            raw = await loop.run_in_executor(None, _call)
            data, cleaned = _best_effort_json(raw)
            if data is None:
                raise ValueError("no parseable JSON object in response")
            return data, cleaned, None
        except Exception as e:
            last_exc = e
            # Hard quota → don't waste more attempts here; jump to Cactus.
            if _is_quota_error(e):
                snippet = (raw[:160] if raw else "").replace("\n", " ")
                print(
                    f"[ambient] gemini quota exhausted: {type(e).__name__} "
                    f"— falling through to on-device fallback. raw={snippet!r}"
                )
                return None, "", e
            if attempt < AMBIENT_ANALYSE_RETRIES - 1 and _is_retryable_gemini_error(e):
                delay = min(12.0, 1.25 * (2**attempt))
                print(
                    f"[ambient] analyse retry {attempt + 1}/{AMBIENT_ANALYSE_RETRIES} "
                    f"after {e!s} — sleeping {delay:.1f}s",
                    flush=True,
                )
                await asyncio.sleep(delay)
                continue
            snippet = (raw[:160] if raw else "").replace("\n", " ")
            print(f"[ambient] analyse failed: {e} — raw={snippet!r}")
            return None, "", e

    return None, "", last_exc


async def _analyse_with_cactus(
    history: list[str],
    previous: AmbientAnalysis | None,
) -> tuple[dict[str, Any] | None, str]:
    """Run the ambient analysis locally via the Cactus CLI with a lean
    prompt tuned for the 270M function-gemma model. Returns (parsed,
    cleaned_text) or (None, '') on failure. Never raises."""
    if not AMBIENT_CACTUS_FALLBACK or not _CACTUS_CLI:
        return None, ""
    print(
        f"[ambient] cactus fallback running (model={AMBIENT_CACTUS_MODEL}, "
        f"timeout={AMBIENT_CACTUS_TIMEOUT_S:.0f}s)…",
        flush=True,
    )
    prompt = _assemble_cactus_prompt(history, previous)
    raw = await _run_cactus_analyse(prompt)
    if not raw:
        return None, ""
    data, cleaned = _best_effort_json(raw)
    if data is None:
        snippet = raw[:160].replace("\n", " ")
        print(f"[ambient] cactus fallback: unparseable JSON — raw={snippet!r}")
        return None, cleaned
    return data, cleaned


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

    Priority order: Gemini Flash (fast, multimodal) → Cactus CLI on-device
    (slower, no image support) → tier-4 silent. The Cactus fallback keeps
    the checklist flowing when Gemini's free-tier quota is exhausted.
    """
    if not history:
        return AmbientAnalysis()
    if not _AVAILABLE and not (AMBIENT_CACTUS_FALLBACK and _CACTUS_CLI):
        return AmbientAnalysis()

    prompt = _assemble_prompt(history, previous, screen_app, screen_window_title)

    data: dict[str, Any] | None = None
    cleaned = ""
    last_exc: BaseException | None = None
    if _AVAILABLE:
        data, cleaned, last_exc = await _analyse_with_gemini(prompt, screen_image_bytes)

    # Escalate to Cactus when Gemini is unavailable OR when it failed
    # (quota / repeated transient / parse). Cactus uses a leaner prompt
    # tuned for the small function-gemma model — assembled internally,
    # not the full multimodal prompt above.
    if data is None and AMBIENT_CACTUS_FALLBACK and _CACTUS_CLI:
        cactus_data, cactus_cleaned = await _analyse_with_cactus(history, previous)
        if cactus_data is not None:
            data = cactus_data
            cleaned = cactus_cleaned

    if data is None:
        # Surface an explicit hint when Gemini quota hit and the local
        # fallback is off — otherwise the checklist silently stops
        # filling and the user has no idea why.
        if last_exc is not None and _is_quota_error(last_exc):
            if not AMBIENT_CACTUS_FALLBACK:
                print(
                    "[ambient] Gemini quota exhausted and local fallback is "
                    "OFF. Set VOICE_AGENT_AMBIENT_CACTUS_FALLBACK=1 to use "
                    "Cactus on-device (slower, sometimes noisy), upgrade the "
                    "Gemini key, or wait for the daily quota reset."
                )
        elif last_exc is not None:
            print(f"[ambient] no analysis produced (last error: {last_exc!s})")
        return AmbientAnalysis()

    tier = int(data.get("tier", 4))
    headline = str(data.get("headline") or "").strip()
    detail = str(data.get("detail") or "").strip()

    # Preferred: new-schema `actions` array. Fallback: legacy
    # single-action fields. Keep the fallback because older cached
    # prompts and rare one-shot outputs still use the old shape.
    actions: list[dict[str, Any]] = []
    raw_actions = data.get("actions")
    if isinstance(raw_actions, list):
        for a in raw_actions:
            if not isinstance(a, dict):
                continue
            kind = str(a.get("kind") or "").strip().lower()
            text = str(a.get("text") or "").strip()
            slots = a.get("slots") if isinstance(a.get("slots"), dict) else {}
            label = str(a.get("label") or "").strip()
            if not kind or kind == "none" or not text:
                continue
            actions.append(
                {"kind": kind, "text": text, "slots": slots or {}, "label": label}
            )
    if not actions:
        legacy_kind = str(data.get("action_kind") or "none").strip().lower()
        legacy_text = str(data.get("action_text") or "").strip()
        legacy_slots = data.get("action_slots") if isinstance(data.get("action_slots"), dict) else {}
        if legacy_kind != "none" and legacy_text:
            actions.append(
                {
                    "kind": legacy_kind,
                    "text": legacy_text,
                    "slots": legacy_slots or {},
                    "label": headline,
                }
            )

    return AmbientAnalysis(
        tier=tier if 1 <= tier <= 4 else 4,
        headline=headline,
        detail=detail,
        actions=actions,
        raw_json=cleaned,
    )
