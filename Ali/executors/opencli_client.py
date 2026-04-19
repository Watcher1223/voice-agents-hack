"""OpenCLI subprocess wrapper.

OpenCLI (https://github.com/jackwener/opencli) exposes 100+ sites as
deterministic CLI commands (e.g. `opencli hackernews top --limit 5`). We
shell out, capture JSON, and hand the rows back to main.py for voice
readback.

No LLM in this path — the adapter does the DOM work deterministically via
its bridge extension + daemon on :19825.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import OPENCLI_NODE_BIN, OPENCLI_ENTRY

_INTENTS_PATH = Path(__file__).resolve().parent.parent / "config" / "opencli_intents.json"


@dataclass
class OpenCliIntent:
    name: str
    match: re.Pattern[str]
    cmd: list[str]
    description: str = ""
    speak_template: str = "{top3_titles}"


@dataclass
class OpenCliResult:
    ok: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    raw_stdout: str = ""
    raw_stderr: str = ""
    returncode: int = 0
    command: list[str] = field(default_factory=list)

    def error_message(self) -> str:
        # Surface the most useful one-liner from stderr/stdout when something
        # goes wrong (extension not connected, site blocked, etc.).
        text = (self.raw_stderr or self.raw_stdout or "").strip()
        first_line = text.splitlines()[0] if text else ""
        return first_line or f"opencli exited {self.returncode}"


def _load_intents() -> list[OpenCliIntent]:
    data = json.loads(_INTENTS_PATH.read_text())
    intents = []
    for entry in data.get("intents", []):
        intents.append(
            OpenCliIntent(
                name=entry["name"],
                match=re.compile(entry["match"]),
                cmd=list(entry["cmd"]),
                description=entry.get("description", ""),
                speak_template=entry.get("speak_template", "{top3_titles}"),
            )
        )
    return intents


_INTENTS: list[OpenCliIntent] | None = None


def _intents() -> list[OpenCliIntent]:
    global _INTENTS
    if _INTENTS is None:
        _INTENTS = _load_intents()
    return _INTENTS


def match_intent(transcript: str) -> tuple[OpenCliIntent, list[str]] | None:
    """Return (intent, capture_groups) if the transcript matches an opencli
    adapter, else None. Capture groups are indexed from 1 (i.e. groups[0] is
    group 1). The caller substitutes them into the intent's cmd template."""
    t = (transcript or "").strip().rstrip(".!?")
    for intent in _intents():
        m = intent.match.match(t)
        if m:
            return intent, list(m.groups())
    return None


def _render_cmd(cmd_template: list[str], groups: list[str]) -> list[str]:
    """Substitute $1, $2, … inside command args from regex capture groups."""
    out: list[str] = []
    for arg in cmd_template:
        def _sub(m: re.Match[str]) -> str:
            idx = int(m.group(1))
            if 1 <= idx <= len(groups) and groups[idx - 1] is not None:
                return groups[idx - 1]
            return ""
        out.append(re.sub(r"\$(\d+)", _sub, arg))
    return out


async def run_intent(intent: OpenCliIntent, groups: list[str]) -> OpenCliResult:
    args = _render_cmd(intent.cmd, groups)
    # Invoke node + entry directly (skip the opencli shim) so an old
    # `env node` resolution in PATH can't pick an incompatible Node version.
    command = [OPENCLI_NODE_BIN, OPENCLI_ENTRY, *args, "-f", "json"]
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    if proc.returncode == 0 and stdout.strip():
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, list):
                rows = [r for r in parsed if isinstance(r, dict)]
            elif isinstance(parsed, dict):
                rows = [parsed]
        except json.JSONDecodeError:
            # Some adapters stream YAML despite -f json; leave rows empty and
            # let the caller speak the raw stdout instead.
            pass
    return OpenCliResult(
        ok=proc.returncode == 0,
        rows=rows,
        raw_stdout=stdout,
        raw_stderr=stderr,
        returncode=proc.returncode or 0,
        command=command,
    )


def summarize(result: OpenCliResult, intent: OpenCliIntent, groups: list[str]) -> str:
    """Turn opencli's row output into a one-sentence voice reply by filling
    the intent's speak_template. Falls back to a generic summary if rows
    aren't present (e.g. YAML output)."""
    titles = []
    for row in result.rows[:3]:
        t = row.get("title") or row.get("name") or row.get("headline") or ""
        if t:
            titles.append(str(t))
    top3_titles = "; ".join(titles) if titles else (result.raw_stdout[:300].strip() or "no results")
    template_vars = {
        "top3_titles": top3_titles,
        "count": str(len(result.rows)),
    }
    for i, g in enumerate(groups, start=1):
        template_vars[f"match_{i}"] = g or ""
    try:
        return intent.speak_template.format(**template_vars)
    except Exception:
        return top3_titles
