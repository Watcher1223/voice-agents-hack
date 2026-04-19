"""Local Gemma-4 (Cactus sidecar) silence gate.

One tiny HTTP call, hitting CACTUS_SIDECAR_URL at /v1/complete:

    should_surface_gemma(window) -> bool | None

Returns True if Gemma thinks the window contains a factual question, a term
worth defining, or a concrete action; False if it's small talk/filler; None
on any failure (sidecar down, timeout, unparseable output) so callers can
transparently fall through to the cloud Gemini call. Timeout is kept tight
(default 1.5s) — if Gemma can't answer in that window, we'd rather skip it
than stall the ambient loop.

The prompt mirrors the one used in scripts/gemma_eval.py, so numbers from
the eval (93.3% accuracy, 100% valid, p50 660ms, skips 53% of Gemini calls
with 5% FN) transfer directly. Don't edit the prompt without re-running.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from config.settings import CACTUS_SIDECAR_URL


_COMPLETE_URL = CACTUS_SIDECAR_URL.rstrip("/") + "/v1/complete"
_DEFAULT_TIMEOUT_S = float(os.environ.get("VOICE_AGENT_GEMMA_TIMEOUT_S", "1.5"))


_SILENCE_SYSTEM = """You are a "should I interrupt?" gate for an ambient AI assistant.
The user is in a conversation and you see a short rolling transcript.
Decide whether there's anything worth surfacing to them right now.

Output EXACTLY one word:
  surface  - there is a factual question to answer, a jargon term to define, or a concrete action to take (send email, add calendar event, lookup)
  silent   - small talk, filler, greetings, repetition, or nothing actionable

Output one of: surface, silent"""


_VALID_SILENCE = {"surface", "silent"}


def _complete(system: str, user: str, max_tokens: int, timeout_s: float) -> str | None:
    """POST to the sidecar. Returns the response text on success, None on failure."""
    body = json.dumps({
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        _COMPLETE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data: dict[str, Any] = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError):
        return None
    if not data.get("success"):
        return None
    return (data.get("response") or "").strip()


def _parse_one_of(raw: str, valid: set[str]) -> str | None:
    """Extract the first recognised label from a Gemma response — tolerant of
    JSON wrappers, quotes, code fences, or bare words."""
    if not raw:
        return None
    cleaned = raw.strip().strip("`").strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            for k in ("label", "answer", "result", "decision"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip().lower() in valid:
                    return v.strip().lower()
        if isinstance(obj, str) and obj.strip().lower() in valid:
            return obj.strip().lower()
    except (json.JSONDecodeError, ValueError):
        pass
    low = cleaned.lower()
    for v in valid:
        if v in low:
            return v
    return None


def should_surface_gemma(window: list[str], timeout_s: float = _DEFAULT_TIMEOUT_S) -> bool | None:
    """True if Gemma thinks there's something worth surfacing in the
    transcript window, False for small talk/filler, None on failure."""
    if not window:
        return None
    user = "\n".join(f"- {line}" for line in window[-15:])
    raw = _complete(_SILENCE_SYSTEM, user, max_tokens=8, timeout_s=timeout_s)
    if raw is None:
        return None
    label = _parse_one_of(raw, _VALID_SILENCE)
    if label is None:
        return None
    return label == "surface"
