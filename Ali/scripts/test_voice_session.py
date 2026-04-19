"""Session-continuity spike for the browser sub-agent.

Simulates two consecutive voice utterances that share one long-lived session,
mirroring the llm-in-chrome CLI's `start` → `message` pattern. Run this while
the Chrome extension is loaded and connected to the relay on :7862.

    cd Ali && .venv/bin/python -m scripts.test_voice_session

Utterance 1 drives the agent from a cold start; we capture the
server-generated session id from its returned status.

Utterance 2 ("now click the more information link") only makes sense if the
agent is on the page from utterance 1 — so if it succeeds, session reuse
works end-to-end.
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
    "open example.com",
    "now click the More information link and tell me the page title",
]


async def _drain_confirmations(client: LocalAgentClient, handle: str, status: TaskStatus) -> TaskStatus:
    """Auto-approve any confirmation loops so the spike doesn't hang."""
    while status.state == "awaiting_confirmation":
        summary = status.confirmation.summary if status.confirmation else "?"
        print(f"[spike] auto-approving confirmation: {summary[:80]}")
        status = await client.send_message(handle, "yes, proceed")
    return status


async def main() -> None:
    async with LocalAgentClient() as client:
        handle: str | None = None
        try:
            for idx, text in enumerate(UTTERANCES, start=1):
                print(f"\n─── Utterance {idx}: {text!r} ───")
                if idx == 1:
                    status = await client.run_task(task=text, session_id="")
                    handle = status.id
                    print(f"[u1] server-generated session id = {handle}")
                else:
                    assert handle is not None
                    status = await client.send_message(handle, text)
                    assert status.id == handle, (
                        f"session id drifted between turns: got {status.id!r}, expected {handle!r}"
                    )

                status = await _drain_confirmations(client, handle or status.id, status)

                if status.state != "complete":
                    print(f"[fail] utterance {idx} ended {status.state}: {status.error}")
                    raise SystemExit(1)

                print(f"[u{idx}] ✓ state={status.state}")
                if status.answer:
                    print(f"[u{idx}] answer: {status.answer[:300]}")

            print("\n✅ session reuse works — both utterances shared one session and both completed")
        finally:
            if handle is not None:
                try:
                    await client.cancel(handle)
                except Exception:
                    pass


if __name__ == "__main__":
    asyncio.run(main())
