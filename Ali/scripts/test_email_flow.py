"""End-to-end spike for a realistic chained voice flow: open gmail, read
the top unread message, draft a reply (never send). Exercises session
continuity across three user turns, like:

    "open gmail"
    "what's in my inbox?"
    "reply to that one saying ..."

Uses the user's logged-in Chrome profile via the extension. Nothing
leaves the drafts folder — the spike explicitly tells the agent not to
send.

    cd Ali && .venv/bin/python -m scripts.test_email_flow

If gmail is not logged in, the agent will stop at the sign-in page and
report back; that's a signal to log in once, not a bug.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from executors.browser.agent_client import LocalAgentClient, TaskStatus  # noqa: E402

UTTERANCES = [
    "open gmail.com",
    "tell me the sender and subject of the most recent email at the top of my inbox",
    (
        "open that email and compose a short one-line reply saying "
        "'got it, will take a look tomorrow' — then SAVE AS DRAFT and close. "
        "Do NOT click send."
    ),
]


async def _drain_confirmations(
    client: LocalAgentClient, handle: str, status: TaskStatus
) -> TaskStatus:
    """For this spike, auto-approve anything the agent flags; the
    'don't click send' guard lives in the prompt itself, not in a
    confirmation dialog."""
    while status.state == "awaiting_confirmation":
        summary = status.confirmation.summary if status.confirmation else "?"
        print(f"[spike] auto-approving confirmation: {summary[:120]}")
        status = await client.send_message(handle, "yes, proceed")
    return status


async def main() -> None:
    async with LocalAgentClient() as client:
        handle: str | None = None
        try:
            for idx, text in enumerate(UTTERANCES, start=1):
                print(f"\n─── Utterance {idx}: {text} ───")
                if idx == 1:
                    status = await client.run_task(task=text, session_id="")
                    handle = status.id
                    print(f"[u{idx}] session id = {handle}")
                else:
                    assert handle is not None
                    status = await client.send_message(handle, text)
                    if status.id != handle:
                        raise SystemExit(
                            f"session id drifted: got {status.id!r}, expected {handle!r}"
                        )

                status = await _drain_confirmations(client, handle or status.id, status)

                print(f"[u{idx}] state={status.state}")
                if status.answer:
                    print(f"[u{idx}] answer: {status.answer[:500]}")
                if status.error:
                    print(f"[u{idx}] error:  {status.error[:300]}")

                if status.state != "complete":
                    print(f"\n❌ utterance {idx} did not complete; stopping here")
                    return

            print("\n✅ 3-utterance email flow completed on one session")
        finally:
            if handle is not None:
                try:
                    await client.cancel(handle)
                except Exception:
                    pass


if __name__ == "__main__":
    asyncio.run(main())
