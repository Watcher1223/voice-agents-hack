"""
Task-specific browser-use executors for meeting capture mode.

All actions share ONE injected LocalAgentClient (the same Node MCP server
and Chrome session that the voice-mode persistent browser uses). An
asyncio.Lock serializes access so concurrent action items queue rather
than fight over the single Chrome tab.

The UX still feels parallel because the overlay shows each item flip to
"Running" → "Done" in rapid succession, and extraction happens ahead of
execution.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

# `mcp` (the Anthropic Model Context Protocol SDK) is pulled in transitively
# by agent_client; keeping the import lazy means loading this module doesn't
# require the SDK — only calling the executors does.
if TYPE_CHECKING:
    from executors.browser.agent_client import LocalAgentClient


@dataclass
class TaskResult:
    success: bool
    summary: str        # short, spoken-friendly: "$189 on Delta" / "Draft ready for Korin"
    detail: str = ""    # longer detail or error message
    url: str = ""       # deeplink shown in overlay
    # End-of-meeting confirmation. If confirm_prompt is set, main.py will
    # TTS it and listen for yes/no. On yes, it runs confirm_task through
    # the shared browser client.
    confirm_prompt: str = ""
    confirm_task: str = ""

    def status_label(self) -> str:
        """Status string consumed by the meeting overlay."""
        if self.success:
            short = self.summary[:40]
            return f"done:{short}"
        return "error"


# Demo default — override via env MEETING_HOME_CITY if needed
_DEFAULT_ORIGIN = "San Francisco"


async def _run_through_shared_client(
    client: "LocalAgentClient",
    lock: asyncio.Lock,
    task: str,
    session_id: str,
    url: str | None,
    max_wait: float,
    confirmation_followup_wait: float = 45.0,
):
    """One serialized round-trip through the shared browser. Returns TaskStatus."""
    async with lock:
        handle = await client.run_task(task, session_id, url=url)
        status = await client.poll_until_paused_or_terminal(handle.id, max_wait=max_wait)
        if status.state == "awaiting_confirmation":
            await client.send_message(handle.id, "yes, proceed")
            status = await client.poll_until_paused_or_terminal(
                handle.id, max_wait=confirmation_followup_wait
            )
        return status


async def search_flight(
    destination: str,
    date: str,
    origin: str = "",
    # Legacy browser args kept for call-site compatibility but ignored.
    client: "LocalAgentClient | None" = None,
    lock: "asyncio.Lock | None" = None,
) -> TaskResult:
    """Find the cheapest one-way flight via Kiwi MCP. No browser required."""
    from executors.flights import search_flights, format_flight_summary, build_kiwi_url

    origin = (origin.strip() or _DEFAULT_ORIGIN)
    destination = destination.strip() or "Los Angeles"

    slots: dict = {"origin": origin, "destination": destination}
    if date:
        # Accept YYYY-MM-DD or natural language; pass through — Kiwi MCP
        # validates and the parser already normalised dates upstream.
        slots["depart_date"] = date

    try:
        flights = await search_flights(slots)
    except Exception as e:
        return TaskResult(False, "search failed", detail=str(e))

    if not flights:
        return TaskResult(False, "no flights found",
                          url=build_kiwi_url(slots) if slots.get("destination") else "")

    top = flights[0]
    summary = format_flight_summary(top)
    deeplink = top.get("deepLink", build_kiwi_url(slots))
    return TaskResult(True, summary, url=deeplink)


async def draft_email_in_gmail(
    client: "LocalAgentClient",
    lock: asyncio.Lock,
    recipient: str,
    subject: str,
    body: str,
) -> TaskResult:
    """Open Gmail, compose a draft. Does NOT send — user confirms at meeting end."""
    recipient = recipient.strip() or "the recipient"
    subject   = subject.strip()   or "Follow-up from our meeting"
    body      = body.strip()      or f"Hi {recipient},\n\nFollowing up on our meeting. Let me know your thoughts."

    task = (
        f"Open https://mail.google.com. "
        f"Click the Compose button. "
        f"In the To field enter: {recipient}. "
        f"In the Subject field enter: {subject}. "
        f"In the body type: {body}. "
        f"Do NOT click Send — just save or leave as a draft. "
        f"Reply: 'Draft saved for {recipient}' when done."
    )
    session_id = f"email-{uuid.uuid4().hex[:8]}"

    try:
        status = await _run_through_shared_client(
            client, lock, task, session_id,
            url="https://mail.google.com",
            max_wait=90.0,
        )
        if status.state == "complete":
            recipient_label = recipient if recipient != "the recipient" else "the recipient"
            return TaskResult(
                success=True,
                summary=f"Draft ready for {recipient_label}",
                url="https://mail.google.com",
                confirm_prompt=f"I drafted the email to {recipient_label}. Want me to send it?",
                confirm_task=(
                    f"Open https://mail.google.com. "
                    f"Go to Drafts. Open the most recent draft addressed to {recipient_label}"
                    + (f" with subject '{subject}'" if subject else "")
                    + ". Click the Send button. Reply: 'Email sent to "
                    f"{recipient_label}' when done."
                ),
            )
        err = status.error or status.state
        return TaskResult(False, "draft failed", detail=err)
    except Exception as e:
        return TaskResult(False, "unavailable", detail=str(e))
