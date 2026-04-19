"""Unified append-only log for everything the agent does.

Every path (push-to-talk routing, opencli execution, browser sub-agent
tool calls, ambient analyses, screen focus changes, confirmations)
writes a one-line event to `~/.ali/agent.log` via `log(tag, msg)`.
Stdout still gets the same line so existing terminal workflows are
unchanged.

Why a single file:
- Post-mortem debugging: "what happened at 02:43?" with grep.
- Demo polish: `tail -f ~/.ali/agent.log` is a single pane of glass.
- Replay: feed a log back into a test fixture.

The path is overridable via VOICE_AGENT_LOG for CI / headless runs.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

_LOG_PATH = Path(
    os.environ.get("VOICE_AGENT_LOG", "~/.ali/agent.log")
).expanduser()


def log(tag: str, msg: str) -> None:
    """One-line event: timestamp + tag + message. Never raises — a broken
    log must never crash the agent."""
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{tag}] {msg}"
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line)


def path() -> Path:
    return _LOG_PATH
