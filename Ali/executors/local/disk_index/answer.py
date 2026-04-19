"""
Retrieval-augmented answering backed by Gemma 4 (Cactus).

Local-first: by default we only shell out to the Cactus CLI. Gemini is an
opt-in fallback guarded by ALI_ALLOW_CLOUD_FALLBACK so nothing leaves the
laptop unless the user explicitly permits it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .retrieve import Hit

_CACTUS_CLI = shutil.which("cactus")


@dataclass(frozen=True)
class AnswerResult:
    text: str
    cited_paths: list[str]
    backend: str  # "cactus" | "gemini" | "stub"
    snippets_used: int


async def answer_question(
    transcript: str,
    *,
    profile: dict[str, Any] | None,
    hits: list[Hit],
    cactus_model: str,
    allow_cloud_fallback: bool,
    gemini_key: str | None,
) -> AnswerResult:
    """Produce a short spoken answer grounded in retrieved snippets."""
    transcript = (transcript or "").strip()
    if not transcript:
        return AnswerResult(
            text="I didn't catch that — could you say it again?",
            cited_paths=[],
            backend="stub",
            snippets_used=0,
        )

    prompt = _build_prompt(transcript=transcript, profile=profile, hits=hits)

    if _CACTUS_CLI:
        reply = await _call_cactus(prompt, cactus_model)
        if reply:
            return AnswerResult(
                text=reply,
                cited_paths=[h.path for h in hits],
                backend="cactus",
                snippets_used=len(hits),
            )

    if allow_cloud_fallback and gemini_key:
        reply = await _call_gemini(prompt, gemini_key)
        if reply:
            return AnswerResult(
                text=reply,
                cited_paths=[h.path for h in hits],
                backend="gemini",
                snippets_used=len(hits),
            )

    fallback = _fallback_answer(transcript, profile, hits)
    return AnswerResult(
        text=fallback,
        cited_paths=[h.path for h in hits],
        backend="stub",
        snippets_used=len(hits),
    )


# ─── Prompt shaping ───────────────────────────────────────────────────────────


_SYSTEM = (
    "You are Ali, an on-device assistant. Answer the user's question in ONE or TWO "
    "spoken sentences. Use ONLY the facts in the context below. If the context "
    "doesn't answer the question, say so plainly. No lists, no markdown."
)


def _build_prompt(
    *,
    transcript: str,
    profile: dict[str, Any] | None,
    hits: list[Hit],
) -> str:
    parts: list[str] = [_SYSTEM, ""]

    if profile:
        parts.append("User profile:")
        parts.append(_profile_block(profile))
        parts.append("")

    if hits:
        parts.append("Excerpts from the user's files (most relevant first):")
        for i, hit in enumerate(hits, 1):
            mtime = _fmt_mtime(hit.mtime)
            parts.append(f"[{i}] {hit.path}  (modified {mtime})")
            parts.append(hit.snippet.strip())
            parts.append("")

    parts.append(f"Question: {transcript}")
    parts.append("Answer:")
    return "\n".join(parts)


def _profile_block(profile: dict[str, Any]) -> str:
    lines: list[str] = []
    for key in ("name", "git_email", "hostname", "platform", "home"):
        value = profile.get(key)
        if value:
            lines.append(f"- {key}: {value}")
    me = profile.get("contacts_me")
    if isinstance(me, dict):
        if me.get("emails"):
            lines.append(f"- emails: {', '.join(me['emails'])}")
        if me.get("phones"):
            lines.append(f"- phones: {', '.join(me['phones'])}")
        if me.get("organization"):
            lines.append(f"- organization: {me['organization']}")
    snippet = profile.get("resume_snippet")
    if isinstance(snippet, str) and snippet:
        lines.append("- resume_excerpt:")
        lines.append(snippet[:800])
    return "\n".join(lines) if lines else "(no profile information cached)"


def _fmt_mtime(mtime: float | None) -> str:
    if not mtime:
        return "unknown"
    try:
        return time.strftime("%Y-%m-%d", time.localtime(float(mtime)))
    except (TypeError, ValueError):
        return "unknown"


# ─── Backends ─────────────────────────────────────────────────────────────────


async def _call_cactus(prompt: str, model: str) -> str:
    if not _CACTUS_CLI:
        return ""
    try:
        proc = await asyncio.create_subprocess_exec(
            _CACTUS_CLI,
            "run",
            model,
            "--prompt",
            prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
    except (OSError, asyncio.TimeoutError, asyncio.CancelledError):
        return ""
    if proc.returncode != 0:
        return ""
    return _clean_reply(stdout.decode("utf-8", errors="ignore"))


async def _call_gemini(prompt: str, api_key: str) -> str:
    try:
        from google import genai as _genai  # type: ignore
    except ImportError:
        return ""

    loop = asyncio.get_event_loop()

    def _sync() -> str:
        client = _genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=_genai.types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=200,
            ),
        )
        return (response.text or "").strip()

    try:
        raw = await loop.run_in_executor(None, _sync)
    except Exception:
        return ""
    return _clean_reply(raw)


def _clean_reply(raw: str) -> str:
    text = (raw or "").strip()
    # Strip model echoes of prompt scaffolding.
    text = re.sub(r"^```(?:json|text)?|```$", "", text, flags=re.MULTILINE).strip()
    # Some Cactus builds prepend the prompt — drop up to the first "Answer:" line.
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1].strip()
    # Clip to two sentences for spoken output.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 2:
        text = " ".join(sentences[:2])
    return text.strip()


# ─── Fallback when every backend fails ────────────────────────────────────────


def _fallback_answer(
    transcript: str,
    profile: dict[str, Any] | None,
    hits: list[Hit],
) -> str:
    lowered = transcript.lower()
    if profile and any(kw in lowered for kw in ("who am i", "my name", "what is my name")):
        name = profile.get("name") or (profile.get("contacts_me", {}) or {}).get("name")
        if name:
            return f"You're {name}."
    if profile and "email" in lowered:
        email = profile.get("git_email")
        emails = (profile.get("contacts_me", {}) or {}).get("emails") or []
        candidate = email or (emails[0] if emails else None)
        if candidate:
            return f"Your email is {candidate}."
    if hits:
        top = hits[0]
        return f"I can't reach the model, but {top.name} looks most relevant."
    return "I can't reach the model right now, and I don't have that in my index."
