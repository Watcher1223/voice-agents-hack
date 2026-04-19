"""
Layer 2 — Intent Parser

Priority order:
  1. Gemini API (via google-generativeai) — fast, reliable, sub-second
  2. Cactus CLI (on-device Gemma 4) — private, but slow on CPU without ANE
  3. Rule-based fallback — covers the 3 core demo flows with no network

The demo story: audio STT runs on-device (Cactus/Gemma 4 audio tower).
Intent parsing uses Gemini because text generation needs a GPU/ANE to be
fast enough for real-time use. Privacy note: the transcript (text) goes to
Gemini, but the raw audio never leaves the device.
"""

import asyncio
import json
import os
import re
import shutil
from urllib.parse import quote_plus

from intent.schema import IntentObject, KnownGoal
from config.settings import CACTUS_INTENT_MODEL, GEMINI_API_KEY


# #region agent log
def _dlog(loc: str, msg: str, data: dict, hid: str = "H12") -> None:
    try:
        import json as _j, os as _o, time as _t
        _p = "/Users/alspenceramitojr/Desktop/Ali/.cursor/debug-4ea166.log"
        _o.makedirs(_o.path.dirname(_p), exist_ok=True)
        with open(_p, "a") as _f:
            _f.write(_j.dumps({
                "sessionId": "4ea166",
                "hypothesisId": hid,
                "location": loc,
                "message": msg,
                "data": data,
                "timestamp": int(_t.time() * 1000),
            }) + "\n")
            _f.flush()
    except Exception:
        pass
# #endregion

# ── Backend availability checks ───────────────────────────────────────────────
try:
    from google import genai as _genai  # type: ignore
    GEMINI_AVAILABLE = bool(GEMINI_API_KEY)
except ImportError:
    GEMINI_AVAILABLE = False

CACTUS_CLI = shutil.which("cactus")
CACTUS_AVAILABLE = CACTUS_CLI is not None

# Sticky: once Gemini says the account is out of quota / over rate limit,
# skip it for the rest of the session instead of paying the round-trip
# (and dumping a multi-line error) on every command.
_GEMINI_DISABLED_THIS_SESSION = False

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intent classifier for a voice agent called Ali.
Given a voice transcript, output ONE JSON object with EXACTLY these fields:
{
  "goal": one of [apply_to_job, send_message, send_email, add_calendar_event, open_url, find_file, capture_meeting, ask_knowledge, unknown],
  "target": {"type": "url|contact|file|question|calendar", "value": "..."},
  "uses_local_data": list of strings drawn from [resume, cover_letter, attachment, document, deck, contacts, calendar, index],
  "requires_browser": true|false,
  "requires_submission": true|false,
  "slots": { goal-specific key/value pairs extracted from the transcript }
}

Goal definitions (read carefully — pick the most specific one):

- apply_to_job: the user wants to SUBMIT an application to a job/program/accelerator (e.g. "apply to YC", "apply to the Stripe role"). NOT for "apply a filter", "apply settings", "apply the patch".
  slots: {"company": str, "role": str?, "batch": str?}
  requires_browser: true, requires_submission: true.

- send_message: send an iMessage/SMS/chat to a specific person (e.g. "text Hanzi I'm late", "message Corinne saying hi", "text Hanzi the Q1 Report").
  Trigger words ONLY count when they're verbs ("text X...", "message X..."). NOT for "the next message said...", "text on the slide", "the text of the doc".
  slots: {"contact": str, "body": str?, "file_query": str?}
  If the user references a file to send alongside or instead of a body ("text Hanzi the Q1 Report"), extract the natural-language file name into "file_query".
  uses_local_data: ["contacts"]; include "attachment" too when a file_query is set. requires_submission: true.

- send_email: compose/send an email, usually with a file attachment (e.g. "email Sam the deck", "send the Q1 doc to my boss", "email Hanzi the Q1 Report").
  slots: {"to": str?, "subject": str?, "body": str?, "file_query": str?}
  uses_local_data: include "attachment" when a file is referenced. requires_submission: true.

- add_calendar_event: create a calendar event/meeting/reminder (e.g. "schedule a meeting with Sam Tuesday at 3", "add dentist Friday noon").
  slots: {"title": str, "when": str?, "duration_minutes": int?, "attendees": list[str]?}
  uses_local_data: ["calendar"], requires_submission: true.

- open_url: open a website or web service (e.g. "open my linkedin", "go to docs.google.com", "open gmail").
  slots: {"url": str}  (use https://www.<service>.com if only a service name is given)
  requires_browser: false (we just launch the URL), requires_submission: false.

- find_file: locate/reveal a LOCAL file or folder (e.g. "find my resume", "where is my 2024 tax return", "open my Q1 deck", "show me my cover letter").
  "open my <thing>" is find_file when <thing> is a document/file type; it is open_url when <thing> is a web service (linkedin, gmail, github, etc.).
  slots: {"file_query": str}

- capture_meeting: start live meeting transcription/notes (e.g. "start meeting capture", "take notes for this meeting", "listen to this meeting").
  slots: {}

- ask_knowledge: the user is asking a question that should be answered from their local files/identity/notes (e.g. "who am I", "what's my email", "when did I last update my resume", "summarize my OKR notes", "what did my contract say about termination").
  Question-shaped utterances ending in "?" usually belong here unless they are clearly an imperative.
  slots: {"question": str}
  uses_local_data: ["index"].

- find_flights: search for flights between two cities (e.g. "find flights from SF to Tokyo next weekend", "flights to Paris May 5th", "show me tickets from Boston to London").
  slots: {"origin": str, "destination": str, "depart_date": "YYYY-MM-DD"?, "return_date": "YYYY-MM-DD"?}
  requires_browser: false, requires_submission: false.

- unknown: only when none of the above fit.

Examples:

Transcript: "apply to YC with my resume"
{"goal":"apply_to_job","target":{"type":"url","value":"apply.ycombinator.com"},"uses_local_data":["resume"],"requires_browser":true,"requires_submission":true,"slots":{"company":"YC"}}

Transcript: "apply the filter to these photos"
{"goal":"unknown","target":{},"uses_local_data":[],"requires_browser":false,"requires_submission":false,"slots":{}}

Transcript: "the next message says the deadline is Friday"
{"goal":"unknown","target":{},"uses_local_data":[],"requires_browser":false,"requires_submission":false,"slots":{}}

Transcript: "text Corinne I'll be ten minutes late"
{"goal":"send_message","target":{"type":"contact","value":"Corinne"},"uses_local_data":["contacts"],"requires_browser":false,"requires_submission":true,"slots":{"contact":"Corinne","body":"I'll be ten minutes late"}}

Transcript: "text hanzi the Q1 Report"
{"goal":"send_message","target":{"type":"contact","value":"hanzi"},"uses_local_data":["contacts","attachment"],"requires_browser":false,"requires_submission":true,"slots":{"contact":"hanzi","file_query":"Q1 Report"}}

Transcript: "schedule a meeting with Sam Tuesday at 3"
{"goal":"add_calendar_event","target":{"type":"calendar","value":""},"uses_local_data":["calendar"],"requires_browser":false,"requires_submission":true,"slots":{"title":"Meeting with Sam","when":"Tuesday at 3","attendees":["Sam"]}}

Transcript: "open my resume"
{"goal":"find_file","target":{"type":"file","value":"resume"},"uses_local_data":[],"requires_browser":false,"requires_submission":false,"slots":{"file_query":"resume"}}

Transcript: "open my linkedin"
{"goal":"open_url","target":{"type":"url","value":"https://www.linkedin.com"},"uses_local_data":[],"requires_browser":false,"requires_submission":false,"slots":{"url":"https://www.linkedin.com"}}

Transcript: "email hanzi the Q1 Report"
{"goal":"send_email","target":{"type":"contact","value":"hanzi"},"uses_local_data":["attachment"],"requires_browser":false,"requires_submission":true,"slots":{"to":"hanzi","subject":"Q1 Report","file_query":"Q1 Report"}}

Transcript: "who am I"
{"goal":"ask_knowledge","target":{"type":"question","value":"who am I"},"uses_local_data":["index"],"requires_browser":false,"requires_submission":false,"slots":{"question":"who am I"}}

Transcript: "start meeting capture"
{"goal":"capture_meeting","target":{},"uses_local_data":[],"requires_browser":false,"requires_submission":false,"slots":{}}

Output ONLY the JSON object. No prose, no markdown fences, no explanation."""


async def parse_intent(transcript: str) -> IntentObject:
    """
    Parse a raw transcript into an IntentObject.

    Priority:
      1. Gemini  — primary classifier (fast, schema-aware, handles nuance)
      2. Cactus  — on-device fallback when Gemini is unavailable
      3. Rule-based  — last-ditch offline fallback (keyword heuristics)
    """
    global _GEMINI_DISABLED_THIS_SESSION
    if GEMINI_AVAILABLE and not _GEMINI_DISABLED_THIS_SESSION:
        try:
            gem = await _parse_with_gemini(transcript)
            # #region agent log
            _dlog(
                "intent:parse_intent:final",
                "gemini intent selected",
                {"transcript": transcript, "final_goal": gem.goal.value, "source": "gemini"},
                "H12",
            )
            # #endregion
            return gem
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "quota" in err_str.lower():
                _GEMINI_DISABLED_THIS_SESSION = True
                print("[intent] Gemini quota exhausted → Cactus for rest of session")
            else:
                print(f"[intent] Gemini failed ({e}), trying Cactus")
            # #region agent log
            _dlog(
                "intent:parse_intent:gemini_error",
                "gemini parse failed",
                {"transcript": transcript, "err": err_str[:180],
                 "disabled_session": _GEMINI_DISABLED_THIS_SESSION},
                "H12",
            )
            # #endregion

    if CACTUS_AVAILABLE:
        try:
            cat = await _parse_with_cactus(transcript)
            # #region agent log
            _dlog(
                "intent:parse_intent:final",
                "cactus intent selected",
                {"transcript": transcript, "final_goal": cat.goal.value, "source": "cactus"},
                "H12",
            )
            # #endregion
            return cat
        except Exception as e:
            print(f"[intent] Cactus failed ({e}), using rule fallback")
            # #region agent log
            _dlog(
                "intent:parse_intent:cactus_error",
                "cactus parse failed",
                {"transcript": transcript, "err": str(e)[:180]},
                "H12",
            )
            # #endregion

    rule = _rule_based_parse(transcript)
    # #region agent log
    _dlog(
        "intent:parse_intent:final",
        "rule-based intent selected (offline fallback)",
        {"transcript": transcript, "final_goal": rule.goal.value, "source": "rule"},
        "H12",
    )
    # #endregion
    return rule


async def _parse_with_gemini(transcript: str) -> IntentObject:
    prompt = f"{SYSTEM_PROMPT}\n\nTranscript: {transcript}"
    loop = asyncio.get_event_loop()

    def _call():
        client = _genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=_genai.types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=384,
                response_mime_type="application/json",
            ),
        )
        return response.text

    raw = await loop.run_in_executor(None, _call)
    return _parse_json_response(raw, transcript)


_CACTUS_PARSE_TIMEOUT_S = float(os.environ.get("ALI_CACTUS_INTENT_TIMEOUT_S", "8.0"))


async def _parse_with_cactus(transcript: str) -> IntentObject:
    prompt = f"{SYSTEM_PROMPT}\n\nTranscript: {transcript}"
    # Keep CLI args minimal for broad cactus version compatibility.
    # Some installs reject "--max-tokens"/"--temperature" for `cactus run`.
    proc = await asyncio.create_subprocess_exec(
        CACTUS_CLI, "run", CACTUS_INTENT_MODEL, "--prompt", prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_CACTUS_PARSE_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        # Cactus on CPU with a 2B model can take 30s+. Kill and fall through
        # to the rule-based parser so the user gets *something* quickly.
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        raise RuntimeError(
            f"cactus timed out after {_CACTUS_PARSE_TIMEOUT_S:.0f}s "
            f"(model={CACTUS_INTENT_MODEL})"
        )
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode().strip())
    return _parse_json_response(stdout.decode(), transcript)


def _parse_json_response(raw: str, transcript: str) -> IntentObject:
    """
    Parse an LLM JSON response into an IntentObject with defensive coercion.

    Raises RuntimeError on hard failures (malformed JSON, non-object payload)
    so the caller can fall through to the next backend.
    """
    if not raw:
        raise RuntimeError("empty response from LLM")

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object, got {type(data).__name__}")

    raw_goal = str(data.get("goal", "unknown")).strip().lower()
    goal = KnownGoal._value2member_map_.get(raw_goal, KnownGoal.UNKNOWN)

    target = data.get("target", {})
    if not isinstance(target, dict):
        target = {}

    uses_local_data = data.get("uses_local_data", [])
    if not isinstance(uses_local_data, list):
        uses_local_data = []
    uses_local_data = [str(x) for x in uses_local_data if isinstance(x, (str, int, float))]

    slots = data.get("slots", {})
    if not isinstance(slots, dict):
        slots = {}

    # LLMs love to return "tomorrow" / "next weekend" verbatim in
    # depart_date, but Kiwi needs YYYY-MM-DD. They also hallucinate the
    # year ("April 20" → "2023-04-20") since training data skews past.
    # Normalise both issues so every caller sees a clean future date.
    if goal == KnownGoal.FIND_FLIGHTS:
        import datetime as _dt
        today = _dt.date.today()
        for key in ("depart_date", "return_date"):
            raw_date = str(slots.get(key) or "").strip()
            if not raw_date:
                continue
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
                try:
                    d = _dt.date.fromisoformat(raw_date)
                except ValueError:
                    continue
                if d < today:
                    # Roll forward to the next occurrence of (month, day).
                    try:
                        candidate = d.replace(year=today.year)
                    except ValueError:
                        candidate = d  # Feb 29 on non-leap year; leave as-is.
                    if candidate < today:
                        candidate = candidate.replace(year=today.year + 1)
                    slots[key] = candidate.isoformat()
            else:
                iso = _parse_when_phrase(raw_date, today)
                if iso:
                    slots[key] = iso

    return IntentObject(
        goal=goal,
        target=target,
        uses_local_data=uses_local_data,
        requires_browser=bool(data.get("requires_browser", False)),
        requires_submission=bool(data.get("requires_submission", False)),
        slots=slots,
        raw_transcript=transcript,
    )


def _extract_contact_and_body(transcript: str) -> tuple[str, str]:
    """
    Extract contact name and message body from natural language like:
      "Text Hanzi I'll be late"
      "Can you text Corinne and tell her what's up"
      "Send a message to Ethan saying I'm on my way"
    """
    # Skip noise words before we look for a name
    SKIP = {
        "text", "message", "imessage", "send", "a", "an", "the", "to",
        "can", "you", "please", "hey", "hi", "and", "saying", "say",
        "tell", "him", "her", "them", "that", "i", "me", "my",
    }
    words = transcript.split()

    # Find the first word after a trigger keyword that looks like a name
    trigger_indices = [
        i for i, w in enumerate(words)
        if w.lower() in ("text", "message", "imessage")
    ]

    contact = "unknown"
    contact_idx = -1

    if trigger_indices:
        # Look at words immediately after the trigger
        start = trigger_indices[-1] + 1
        for i in range(start, min(start + 5, len(words))):
            w = words[i]
            if w.lower() not in SKIP and len(w) > 1:
                contact = w.rstrip(".,!?")
                contact_idx = i
                break

    # Body = everything after the contact name
    if contact_idx >= 0 and contact_idx + 1 < len(words):
        body_words = words[contact_idx + 1:]
        # Strip connector words at the start of the body ("and tell him", "saying", etc.)
        while body_words and body_words[0].lower() in ("and", "saying", "that", "to"):
            body_words = body_words[1:]
        body = " ".join(body_words).strip()
    else:
        body = transcript

    return contact, body


_FIND_FILE_TRIGGERS = (
    "find my ",
    "find the ",
    "where is my ",
    "where's my ",
    "show me my ",
    "open my ",
    "reveal my ",
    "locate my ",
)

_ATTACHMENT_TRIGGERS = (
    "attach ",
    "attachment",
    "send me the ",
    "email me the ",
    "email me my ",
    "send the file",
)

_FILE_HINT_WORDS = {
    "resume", "cv", "cover", "letter", "deck", "slides", "document", "doc",
    "docx", "pdf", "file", "folder", "finder", "download", "downloads",
}
_FILE_EXT_HINTS = (".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".pages", ".ppt", ".pptx")


def _infer_open_url_target(transcript: str) -> str | None:
    """
    Infer a web destination from phrases like:
      - "open my linkedin"
      - "open gmail"
      - "go to docs.google.com"
    Returns None if this sounds like a local file/folder request.
    """
    t = transcript.lower().strip()
    cue = None
    for c in ("open my ", "open ", "go to ", "visit ", "launch "):
        if t.startswith(c):
            cue = c
            break
    if cue is None:
        return None

    query = transcript[len(cue):].strip().rstrip(".?!")
    if not query:
        return None
    ql = query.lower()
    tokens = [tok for tok in re.findall(r"[a-z0-9._-]+", ql) if tok]
    if not tokens:
        return None

    # If this looks file-like, let FIND_FILE handle it.
    if any(w in tokens for w in _FILE_HINT_WORDS):
        return None
    if any(ext in ql for ext in _FILE_EXT_HINTS):
        return None

    # Explicit URL/domain
    if ql.startswith("http://") or ql.startswith("https://"):
        return query
    if "." in tokens[0] and " " not in query:
        return f"https://{tokens[0]}" if not ql.startswith("http") else query

    # Single service token (linkedin, github, notion, etc.) -> direct domain.
    if len(tokens) == 1 and len(tokens[0]) >= 3:
        return f"https://www.{tokens[0]}.com"

    # Fallback to web search for multi-word destinations.
    return f"https://www.google.com/search?q={quote_plus(query)}"


def _extract_file_query(transcript: str, trigger: str) -> str:
    lower = transcript.lower()
    idx = lower.find(trigger)
    if idx < 0:
        return transcript.strip()
    tail = transcript[idx + len(trigger) :].strip()
    # Trim trailing punctuation.
    return tail.rstrip(".?! ").strip() or transcript.strip()


_KNOWLEDGE_QUESTION_STARTS = (
    "who ",
    "whose ",
    "what ",
    "what's ",
    "whats ",
    "when ",
    "where ",
    "why ",
    "how ",
    "do i ",
    "am i ",
    "is my ",
    "are my ",
    "was my ",
    "were my ",
    "tell me about ",
    "summarize ",
    "summarise ",
)


def _is_knowledge_question(transcript: str) -> bool:
    """Question-shaped utterances that should route through RAG over the disk index."""
    t = (transcript or "").strip().lower()
    if not t:
        return False
    # Explicit question mark is a strong signal.
    if t.endswith("?"):
        return True
    # Keep imperative file-reveal phrasing ("find/open my X") off this path —
    # those are handled above.
    for kw in ("find my", "open my", "show me my", "reveal my", "locate my"):
        if kw in t:
            return False
    return any(t.startswith(prefix) for prefix in _KNOWLEDGE_QUESTION_STARTS)


_FLIGHT_TRIGGERS = ("flight", "flights", "fly to", "fly from", "ticket to", "tickets to")


def _parse_when_phrase(phrase: str, today) -> str | None:
    """
    Convert a natural-language date phrase into an absolute YYYY-MM-DD string.

    `today` is a datetime.date (passed in so this stays testable).

    Handle at minimum:
      - "tomorrow"              → today + 1 day
      - "next weekend"          → the upcoming Saturday
      - "in N days" / "in N weeks"
      - "may 5", "may 5th", "december 2"  (current year, or next year if already past)

    Return None for anything you don't recognise — the caller will just
    omit the date and Kiwi will show "any date".
    """
    import datetime as _dt
    p = phrase.strip().lower().removeprefix("on ").strip()
    if p == "tomorrow":
        return (today + _dt.timedelta(days=1)).isoformat()
    if "next weekend" in p or p == "weekend":
        return (today + _dt.timedelta(days=(5 - today.weekday()) % 7 or 7)).isoformat()
    m = re.match(r"in\s+(\d+)\s+(day|days|week|weeks)", p)
    if m:
        n = int(m.group(1)) * (7 if "week" in m.group(2) else 1)
        return (today + _dt.timedelta(days=n)).isoformat()
    months = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    m = re.search(r"\b([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?\b", p)
    if m and m.group(1)[:3] in months:
        mo, day = months[m.group(1)[:3]], int(m.group(2))
        year = today.year if (mo, day) >= (today.month, today.day) else today.year + 1
        try:
            return _dt.date(year, mo, day).isoformat()
        except ValueError:
            return None
    return None


def _extract_flight_slots(transcript: str) -> dict:
    """
    Pull origin/destination/dates out of phrases like:
      "find flights from SF to Tokyo next weekend"
      "flights to Paris on May 5"
      "fly from Boston to London tomorrow"
    """
    import datetime
    t = transcript.lower()

    # Origin: "from X to Y" — capture everything between.
    m = re.search(r"from\s+([a-z][a-z\s.]+?)\s+to\s+([a-z][a-z\s.]+?)(?:\s+(?:on|next|tomorrow|in|this|for)|[,.?!]|$)", t)
    if m:
        origin, destination = m.group(1).strip(), m.group(2).strip()
    else:
        # "flights to Tokyo" — no explicit origin.
        m2 = re.search(r"(?:to|for)\s+([a-z][a-z\s.]+?)(?:\s+(?:on|next|tomorrow|in|this|for)|[,.?!]|$)", t)
        origin = ""
        destination = m2.group(1).strip() if m2 else ""

    slots: dict = {"origin": origin, "destination": destination}

    # Date phrase: grab what follows "on"/"next"/"tomorrow"/"in" and hand to
    # the parser. Fine if nothing matches — date is optional.
    d = re.search(r"(?:\b(?:on|next|tomorrow|in this|in)\b\s*)([a-z0-9\s]+?)(?:[,.?!]|$)", t)
    if d:
        date_str = _parse_when_phrase(d.group(0).strip(), datetime.date.today())
        if date_str:
            slots["depart_date"] = date_str

    return slots


def _rule_based_parse(transcript: str) -> IntentObject:
    """Keyword fallback covering the three core demo flows."""
    t = transcript.lower()

    if any(kw in t for kw in _FLIGHT_TRIGGERS):
        slots = _extract_flight_slots(transcript)
        if slots.get("destination"):
            return IntentObject(
                goal=KnownGoal.FIND_FLIGHTS,
                target={"type": "url", "value": "kiwi.com"},
                uses_local_data=[],
                requires_browser=False,
                requires_submission=False,
                slots=slots,
                raw_transcript=transcript,
            )

    url_target = _infer_open_url_target(transcript)
    if url_target is not None:
        return IntentObject(
            goal=KnownGoal.OPEN_URL,
            target={"type": "url", "value": url_target},
            uses_local_data=[],
            requires_browser=False,
            requires_submission=False,
            slots={"url": url_target},
            raw_transcript=transcript,
        )

    if any(kw in t for kw in [
        "start meeting", "capture meeting", "meeting capture",
        "listen to meeting", "take notes", "record meeting",
        "start capture", "capture this",
    ]):
        return IntentObject(
            goal=KnownGoal.CAPTURE_MEETING,
            target={},
            uses_local_data=[],
            requires_browser=False,
            requires_submission=False,
            slots={},
            raw_transcript=transcript,
        )

    if any(kw in t for kw in ["apply", "yc", "y combinator", "application"]):
        return IntentObject(
            goal=KnownGoal.APPLY_TO_JOB,
            target={"type": "url", "value": "apply.ycombinator.com"},
            uses_local_data=["resume"],
            requires_browser=True,
            requires_submission=True,
            slots={"company": "YC", "batch": "Fall 2026"},
            raw_transcript=transcript,
        )

    for trigger in _FIND_FILE_TRIGGERS:
        if trigger in t:
            query = _extract_file_query(transcript, trigger)
            return IntentObject(
                goal=KnownGoal.FIND_FILE,
                target={"type": "file", "value": query},
                uses_local_data=[],
                requires_browser=False,
                requires_submission=False,
                slots={"file_query": query},
                raw_transcript=transcript,
            )

    if _is_knowledge_question(transcript):
        return IntentObject(
            goal=KnownGoal.ASK_KNOWLEDGE,
            target={"type": "question", "value": transcript.strip()},
            uses_local_data=["index"],
            requires_browser=False,
            requires_submission=False,
            slots={"question": transcript.strip()},
            raw_transcript=transcript,
        )

    if any(kw in t for kw in ["text", "message", "imessage"]):
        contact, body = _extract_contact_and_body(transcript)
        return IntentObject(
            goal=KnownGoal.SEND_MESSAGE,
            target={"type": "contact", "value": contact},
            uses_local_data=["contacts"],
            requires_browser=False,
            requires_submission=True,
            slots={"contact": contact, "body": body},
            raw_transcript=transcript,
        )

    if any(kw in t for kw in ["meeting", "calendar", "schedule", "event"]):
        return IntentObject(
            goal=KnownGoal.ADD_CALENDAR_EVENT,
            target={"type": "calendar", "value": ""},
            uses_local_data=["calendar"],
            requires_browser=False,
            requires_submission=True,
            slots={"title": transcript},
            raw_transcript=transcript,
        )

    if any(kw in t for kw in ["email", "mail"]) and any(trig in t for trig in _ATTACHMENT_TRIGGERS):
        file_query = transcript.strip()
        for trig in _ATTACHMENT_TRIGGERS:
            if trig in t:
                file_query = _extract_file_query(transcript, trig)
                break
        return IntentObject(
            goal=KnownGoal.SEND_EMAIL,
            target={"type": "contact", "value": ""},
            uses_local_data=["attachment"],
            requires_browser=False,
            requires_submission=True,
            slots={"file_query": file_query},
            raw_transcript=transcript,
        )

    return IntentObject.unknown(transcript)
