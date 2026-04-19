"""Conversation-mode classifier + per-mode guidance prompts.

The ambient analyser's generic prompt is decent but treats every
conversation the same. A sales call needs real-time fact-checking; a
planning meeting needs action-item extraction; casual chat needs
neither. Mode detection lets us swap in specialized guidance so the
same rolling transcript produces different (and better) suggestions.

Two-stage design:
  1. Cheap keyword heuristic runs every N turns (free, ~O(n)).
  2. Detected mode is injected into the ambient prompt as an extra
     guidance block ahead of the tier classifier.

Keep the keyword lists tight — false-positives (accidentally flipping
into sales mode during a casual chat) make the agent's behaviour feel
random.
"""
from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    GENERIC    = "generic"
    SALES_CALL = "sales_call"
    MEETING    = "meeting"


# Phrases that are strongly associated with a sales / discovery call.
# Match is case-insensitive substring — keep phrases long enough that
# they don't trip on casual chat.
_SALES_CALL_PHRASES = (
    "our product",
    "our tool",
    "our platform",
    "our solution",
    "pricing tier",
    "how do you currently",
    "how are you currently",
    "would you use",
    "would you pay",
    "use case",
    "discovery call",
    "sales call",
    "what's your budget",
    "which stakeholders",
    "decision maker",
    "roi",
    "evaluate",
    "competitor",
    "prospect",
    "what problem",
)

# Phrases that indicate we're in a planning / action-item-heavy meeting.
_MEETING_PHRASES = (
    "agenda",
    "action item",
    "action items",
    "okr",
    "okrs",
    "kickoff",
    "kick off",
    "stand up",
    "standup",
    "retrospective",
    "retro meeting",
    "who's the owner",
    "owner is",
    "by friday",
    "by monday",
    "by eod",
    "by end of",
    "decided that",
    "let's decide",
    "follow-up",
    "follow up next week",
)


def detect_mode(history: list[str]) -> Mode:
    """Inspect the most recent ~15 turns and classify.
    Requires at least 2 keyword hits and a clear lead to avoid flapping
    — otherwise falls back to GENERIC."""
    if not history:
        return Mode.GENERIC
    text = " ".join(history[-15:]).lower()
    sales_hits = sum(1 for phrase in _SALES_CALL_PHRASES if phrase in text)
    meeting_hits = sum(1 for phrase in _MEETING_PHRASES if phrase in text)
    # Need at least 2 hits AND a clear winner (>= 2× the loser + 1).
    if sales_hits >= 2 and sales_hits > meeting_hits + 1:
        return Mode.SALES_CALL
    if meeting_hits >= 2 and meeting_hits > sales_hits + 1:
        return Mode.MEETING
    return Mode.GENERIC


_SALES_CALL_GUIDANCE = """\
CONVERSATION MODE: SALES / DISCOVERY CALL
The user is on a live call with a prospect or customer. Their attention
is on the conversation — they can't click pills mid-sentence.

Your job changes:

1. REAL-TIME FACT-CHECKING. When the prospect says a claim, number, or
   company name, surface a verification immediately as tier-3 with
   kind='opencli'. These auto-execute as SAFE reads — the result shows
   on the overlay without user action:
     - "google search <claim>"          — fact-check a statement
     - "google news <company>"          — recent news about a company
     - "wikipedia search <topic>"       — background on a topic
     - "google search <name> linkedin"  — look up a named person

2. PROSPECT / CONTACT LOOKUP. When a person is named (by either side),
   trigger opencli "google search <full name> linkedin" so the rep can
   glance at their background.

3. DO NOT queue calendar or email drafts during the call. The rep will
   handle post-call follow-ups themselves; mid-call task cards are a
   distraction. Skip tier-3 compose_mail / create_calendar_event.

4. TIER-2 DEFINITIONS are still useful when the prospect uses jargon
   the rep may not recognize — surface them briefly.

5. Prioritize information density. The rep's attention is on the
   prospect; your overlay should give them a fact, not ask for a
   decision."""

_MEETING_GUIDANCE = """\
CONVERSATION MODE: MEETING / PLANNING SESSION
The user is in a planning discussion with action items being decided.

Your job:

1. AGGRESSIVELY extract tier-3 action items (emails, messages,
   calendar events, follow-ups). These go into the right-column task
   list; the user reviews and approves after the meeting.

2. Include the action OWNER and DEADLINE in slots. Example:
     {to: 'hanzi', subject: 'Q3 deck', body: 'drop-dead by Friday'}

3. Tier-1/2 answers should be SHORT — the user is focused on the
   meeting, not reading a paragraph.

4. Dedupe aggressively: the same action item mentioned twice in one
   meeting should not create two cards. The store already handles the
   (kind, text, slots) dedupe — just don't fight it by slightly
   rephrasing."""


def guidance_for(mode: Mode) -> str:
    """Return a prompt-guidance block for a mode, or empty string for
    GENERIC. The ambient analyser concatenates this after the main
    system prompt and before the rolling history."""
    if mode == Mode.SALES_CALL:
        return _SALES_CALL_GUIDANCE
    if mode == Mode.MEETING:
        return _MEETING_GUIDANCE
    return ""
