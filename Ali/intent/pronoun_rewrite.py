"""
Temporary demo helper: rewrite first-person pronouns ("me", "my") in a
spoken utterance so they read as the fixed contact name "hanzi".

The demo story is: when the user says "text me the Q1 Report. Also send
it to my email", both tasks should assign Hanzi as the recipient. The
downstream intent parser, multi-action extractor, and ambient analyser
all receive the rewritten text so they uniformly route to Hanzi.

Flip ``HANZI_SELF_REWRITE_ENABLED`` to False (or set the env var
``VOICE_AGENT_HANZI_SELF_REWRITE=0``) to disable the rewrite.
"""
from __future__ import annotations

import os
import re

HANZI_SELF_REWRITE_ENABLED: bool = (
    os.environ.get("VOICE_AGENT_HANZI_SELF_REWRITE", "1").strip() not in {"0", "false", "False", ""}
)

_SELF_TARGET = "hanzi"

# Word-boundary match so we don't touch "myself", "meme", "email", etc.
# Case-insensitive — keep the replacement lowercase since it's a name token
# the contact resolver normalises anyway.
_PRONOUN_RE = re.compile(r"\b(?:me|my)\b", re.IGNORECASE)


def rewrite_self_pronouns(text: str) -> str:
    """Replace standalone ``me``/``my`` tokens with ``hanzi``.

    Returns the input unchanged when the rewrite is disabled or when the
    string contains no match — callers can log-compare by identity.
    """
    if not HANZI_SELF_REWRITE_ENABLED:
        return text
    if not text:
        return text
    return _PRONOUN_RE.sub(_SELF_TARGET, text)


__all__ = ["rewrite_self_pronouns", "HANZI_SELF_REWRITE_ENABLED"]
