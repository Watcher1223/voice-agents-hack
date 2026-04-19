"""
Domain vocabulary that biases STT and post-corrects common mis-hearings.

Why this exists:
  Whisper reliably mishears "Korin" → "Corinne" and "LAX" → "Alex". The
  initial_prompt fed to Whisper steers its language model toward these
  proper nouns; the CORRECTIONS list is a narrow, context-qualified
  safety net for the handful of cases the prompt doesn't catch.

Edit this file to add your contacts, project names, and airport codes.
"""

from __future__ import annotations

import re
from typing import Iterable


# ── Vocabulary ────────────────────────────────────────────────────────────────

CONTACTS: list[str] = [
    "Korin",
    "Hanzi",
    "Alspencer",
]

ACRONYMS: list[str] = [
    "YC", "Y Combinator",
    "LAX", "SFO", "JFK", "SJC",
    "LinkedIn", "GitHub", "Gmail", "Ali",
]


# Hard cap on the merged bias list so we stay under Deepgram's ~100 keyterm
# ceiling. Leaves room for ACRONYMS + the static CONTACTS without truncation.
_KEYTERMS_MAX = 95


def keyterms() -> list[str]:
    """All terms Deepgram Nova-3 should bias toward as `keyterm` params.

    Merge order: static CONTACTS first (never dropped), then contact-derived
    unusual names (auto-pruned at 80), then ACRONYMS. Order-preserved dedupe
    so the cap trims low-priority tail entries, not hand-curated ones.
    """
    extra: list[str] = []
    try:
        from config.contact_vocab import get_unusual_first_names
        extra = get_unusual_first_names()
    except Exception as exc:  # contacts permission denied, osascript failure, …
        print(f"[vocab] contact-derived bias unavailable: {exc}")
    return list(dict.fromkeys(CONTACTS + extra + ACRONYMS))[:_KEYTERMS_MAX]

# Context-qualified corrections. For each tuple:
#   - `wrong` is the set of mis-hearings (lowercase, whole-word)
#   - `right` is the canonical spelling
#   - `hints` must contain at least one token present in the transcript
#     for the replacement to fire — prevents "Alex" → "LAX" when the
#     user really did mean a person named Alex.
_Correction = tuple[set[str], str, set[str]]
CORRECTIONS: list[_Correction] = [
    ({"corinne", "corin", "koreen", "karen"}, "Korin",
     {"email", "mail", "message", "text", "call", "reach", "ping", "dm", "draft"}),
    ({"hans", "hansi", "hanz"}, "Hanzi",
     {"email", "mail", "message", "text", "call", "reach", "ping", "dm", "draft"}),
    ({"alex"}, "LAX",
     {"flight", "flights", "fly", "airport", "plane", "travel", "book"}),
]


# ── Whisper initial_prompt ────────────────────────────────────────────────────

def whisper_initial_prompt() -> str:
    """
    A short free-form hint string that Whisper uses as linguistic context.
    Listing proper nouns makes Whisper far more likely to emit them correctly.
    Kept under ~200 chars — Whisper treats it as lightweight bias, not ground truth.
    """
    contacts = ", ".join(CONTACTS)
    acros    = " ".join(ACRONYMS)
    return (
        f"Voice command. Common people: {contacts}. "
        f"Common acronyms and airports: {acros}."
    )


# ── Post-correction ───────────────────────────────────────────────────────────

def _has_any(text_lower: str, hints: Iterable[str]) -> bool:
    return any(h in text_lower for h in hints)


def _all_corrections() -> list[_Correction]:
    """Static hand-curated rules plus auto-derived ones from Contacts.

    Auto-derived rules carry an empty `hints` set, which `apply_corrections`
    treats as "fire unconditionally" — safe because the canonical RHS is
    always a proper noun we've opted-in to recognise.
    """
    rules: list[_Correction] = list(CORRECTIONS)
    try:
        from config.contact_vocab import get_mis_split_rules
        rules.extend(get_mis_split_rules())
    except Exception as exc:
        print(f"[vocab] contact-derived corrections unavailable: {exc}")
    return rules


def apply_corrections(text: str) -> str:
    """
    Replace mis-heard tokens with canonical forms when context supports it.
    Idempotent — safe to call on already-correct text.

    Rules with a non-empty `hints` set only fire when at least one hint
    token is present in the transcript (prevents e.g. "Alex" -> "LAX"
    outside a travel context). Rules with an empty `hints` set fire
    unconditionally — reserved for proper-noun corrections derived from
    the user's own Contacts.
    """
    if not text:
        return text
    lower = text.lower()
    out = text
    for wrong_set, right, hints in _all_corrections():
        if hints and not _has_any(lower, hints):
            continue
        for w in wrong_set:
            # Whole-word, case-insensitive replace.
            pattern = re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE)
            out = pattern.sub(right, out)
    return out
