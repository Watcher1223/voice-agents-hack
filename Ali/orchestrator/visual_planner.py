"""
Vision-first planner.
Given intent + latest visual observation, decide the next atomic action.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from dataclasses import dataclass
from typing import Any

from config.settings import CACTUS_GEMMA4_MODEL
from intent.schema import IntentObject, KnownGoal

ALLOWED_ACTION_TYPES = {
    "navigate",
    "click_text",
    "click_selector",
    "type_selector",
    "upload_file",
    "scroll",
    "wait_for_text",
    "ask_user",
    "run_script",
    "author_script",
    "compose_mail",
    "complete",
    "abort",
}

_ATTACHMENT_ROLES = ("attachment", "deck", "document")

# Actions that must not be gated behind the confirmation prompt even if the
# planner accidentally marks them irreversible — safe by construction.
_ALWAYS_SAFE_ACTION_TYPES = {"run_script", "author_script"}

CACTUS_CLI = shutil.which("cactus")


@dataclass
class NextAction:
    action_type: str
    reason: str
    expected_outcome: str
    safety_level: str
    confirm_required: bool
    params: dict[str, Any]

    def validate(self) -> None:
        if self.action_type not in ALLOWED_ACTION_TYPES:
            raise ValueError(f"Unsupported action_type: {self.action_type}")
        if self.safety_level not in {"safe", "irreversible"}:
            raise ValueError(f"Invalid safety_level: {self.safety_level}")
        if self.action_type in _ALWAYS_SAFE_ACTION_TYPES:
            if self.safety_level != "safe":
                raise ValueError(
                    f"Action {self.action_type} must be safe"
                )
        elif self.safety_level == "irreversible" and not self.confirm_required:
            raise ValueError("Irreversible actions must require confirmation")


async def choose_next_action(
    intent: IntentObject,
    observation: dict[str, Any],
    collected_data: dict[str, Any],
    step_index: int,
    max_steps: int,
) -> NextAction:
    """Return the next action in the observe-decide-act loop."""
    if step_index >= max_steps:
        return NextAction(
            action_type="abort",
            reason="Exceeded configured visual action step limit.",
            expected_outcome="Execution stops safely.",
            safety_level="safe",
            confirm_required=False,
            params={},
        )

    if CACTUS_CLI:
        try:
            action = await _choose_with_cactus(intent, observation, collected_data)
            action.validate()
            return action
        except Exception as exc:
            print(f"[visual-planner] Cactus decision failed ({exc}); using deterministic fallback.")

    action = _fallback_action(intent, observation, collected_data, step_index)
    action.validate()
    return action


async def _choose_with_cactus(
    intent: IntentObject,
    observation: dict[str, Any],
    collected_data: dict[str, Any],
) -> NextAction:
    prompt = _build_prompt(intent, observation, collected_data)
    proc = await asyncio.create_subprocess_exec(
        CACTUS_CLI,
        "run",
        CACTUS_GEMMA4_MODEL,
        "--prompt",
        prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode().strip())
    return _parse_next_action(stdout.decode())


def _build_prompt(intent: IntentObject, observation: dict[str, Any], collected_data: dict[str, Any]) -> str:
    disk_context = _disk_context(collected_data)
    script_catalog = _script_catalog(collected_data)

    instructions = (
        "You are a strict action planner for a local desktop voice agent.\n"
        "Based on the goal, current observation, and state data, return exactly one JSON object.\n"
        "Allowed action_type: navigate, click_text, click_selector, type_selector, upload_file, "
        "scroll, wait_for_text, ask_user, run_script, author_script, complete, abort.\n"
        "Required fields: action_type, reason, expected_outcome, safety_level, confirm_required, params.\n"
        "safety_level must be safe or irreversible.\n"
        "If action is irreversible, confirm_required must be true.\n"
        "upload_file.params.file_role must match one of disk_context.role when disk_context is non-empty.\n"
        "Prefer an existing script from script_catalog via run_script "
        "(params={name, args: {...}}). Only when no catalog entry fits, emit "
        "author_script (params={name, runtime, description, params, body}); "
        "the orchestrator will validate and persist it. "
        "run_script and author_script are always safe.\n"
        "Output only JSON.\n\n"
    )
    return (
        instructions
        + f"intent_goal={intent.goal.value}\n"
        + f"intent_target={json.dumps(intent.target, ensure_ascii=True)}\n"
        + f"intent_slots={json.dumps(intent.slots, ensure_ascii=True)}\n"
        + f"observation={json.dumps(observation, ensure_ascii=True)}\n"
        + f"disk_context={json.dumps(disk_context, ensure_ascii=True)}\n"
        + f"script_catalog={json.dumps(script_catalog, ensure_ascii=True)}\n"
        + f"collected_data={json.dumps(_slim_collected_data(collected_data), ensure_ascii=True)}\n"
    )


def _disk_context(collected_data: dict[str, Any]) -> list[dict[str, str]]:
    resolved = collected_data.get("resolved_local_files") or {}
    if not isinstance(resolved, dict):
        return []
    entries: list[dict[str, str]] = []
    from pathlib import Path as _Path

    for role, path in list(resolved.items())[:6]:
        if not isinstance(path, str) or not path:
            continue
        p = _Path(path)
        entries.append({"role": role, "basename": p.name, "parent": p.parent.name})
    return entries


def _script_catalog(collected_data: dict[str, Any]) -> list[dict[str, Any]]:
    catalog = collected_data.get("script_catalog")
    if isinstance(catalog, list):
        return catalog[:12]
    return []


def _slim_collected_data(collected_data: dict[str, Any]) -> dict[str, Any]:
    # Avoid repeating the catalog and resolved_local_files inside collected_data
    # since we emit them as dedicated blocks.
    slim = dict(collected_data)
    slim.pop("script_catalog", None)
    slim.pop("last_observation", None)
    return slim


def _parse_next_action(raw: str) -> NextAction:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    return NextAction(
        action_type=data.get("action_type", ""),
        reason=data.get("reason", ""),
        expected_outcome=data.get("expected_outcome", ""),
        safety_level=data.get("safety_level", "safe"),
        confirm_required=bool(data.get("confirm_required", False)),
        params=data.get("params", {}) or {},
    )


def _fallback_action(
    intent: IntentObject,
    observation: dict[str, Any],
    collected_data: dict[str, Any],
    step_index: int,
) -> NextAction:
    url = (observation.get("url") or "").lower()
    scope = observation.get("scope", "")

    if intent.goal == KnownGoal.APPLY_TO_JOB:
        if "apply.ycombinator.com" not in url:
            return NextAction(
                action_type="navigate",
                reason="Ensure browser is at YC Apply before form actions.",
                expected_outcome="YC Apply page is open.",
                safety_level="safe",
                confirm_required=False,
                params={"url": "https://apply.ycombinator.com"},
            )
        if not collected_data.get("yc_form_filled"):
            return NextAction(
                action_type="upload_file",
                reason="Run YC form fill phase using known slots and resume.",
                expected_outcome="Known fields become populated.",
                safety_level="safe",
                confirm_required=False,
                params={"mode": "yc_apply_fill"},
            )
        return NextAction(
            action_type="click_text",
            reason="Submit application once form fill stage completed.",
            expected_outcome="YC submission completes.",
            safety_level="irreversible",
            confirm_required=True,
            params={"text": "Submit", "mode": "yc_apply_submit"},
        )

    if intent.goal == KnownGoal.FIND_FILE:
        script_result = collected_data.get("script_result")
        if isinstance(script_result, dict) and script_result.get("name") == "reveal_in_finder":
            return NextAction(
                action_type="complete",
                reason="Revealed the file in Finder.",
                expected_outcome="User sees the file highlighted in a Finder window.",
                safety_level="safe",
                confirm_required=False,
                params={},
            )
        resolved = collected_data.get("resolved_local_files") or {}
        found_path = resolved.get("found") if isinstance(resolved, dict) else None
        if found_path:
            return NextAction(
                action_type="run_script",
                reason="Reveal the located file in Finder for the user.",
                expected_outcome="A Finder window opens highlighting the file.",
                safety_level="safe",
                confirm_required=False,
                params={"name": "reveal_in_finder", "args": {"path": found_path}},
            )
        # No resolution. First step: ask for clarification. Any later step
        # means we already asked and still have nothing — abort so the loop
        # doesn't burn through max_steps re-asking the same question.
        if step_index > 0:
            query = (collected_data.get("slots") or {}).get("file_query") or "(unknown)"
            return NextAction(
                action_type="abort",
                reason=f"No local file matched query={query!r}.",
                expected_outcome="Orchestration stops so the user can refine.",
                safety_level="safe",
                confirm_required=False,
                params={},
            )
        return NextAction(
            action_type="ask_user",
            reason="Could not resolve a local file matching the request.",
            expected_outcome="User clarifies or names the file.",
            safety_level="safe",
            confirm_required=False,
            params={"question": "Which file are you looking for? A more specific name helps."},
        )

    if intent.goal == KnownGoal.SEND_EMAIL:
        resolved = collected_data.get("resolved_local_files") or {}
        attachments: list[str] = []
        if isinstance(resolved, dict):
            for role in _ATTACHMENT_ROLES:
                value = resolved.get(role)
                if isinstance(value, str) and value:
                    attachments.append(value)
        slots = collected_data.get("slots") or {}
        to = str(slots.get("to") or intent.target.get("value") or "")
        subject = str(slots.get("subject") or slots.get("title") or "")
        body = str(slots.get("body") or "")
        return NextAction(
            action_type="compose_mail",
            reason="Draft a mail message with any resolved attachments.",
            expected_outcome="Mail opens a new draft addressed and optionally attached.",
            safety_level="safe",
            confirm_required=False,
            params={
                "to": to,
                "subject": subject,
                "body": body,
                "attachments": attachments,
                "send": False,
            },
        )

    if intent.goal == KnownGoal.SEND_MESSAGE:
        if not collected_data.get("contact_resolved"):
            return NextAction(
                action_type="ask_user",
                reason="Resolve contact before drafting a send action.",
                expected_outcome="Contact address available.",
                safety_level="safe",
                confirm_required=False,
                params={"question": "Resolve contact via local contacts now?"},
            )
        return NextAction(
            action_type="abort",
            reason="Generic fallback does not send messages automatically.",
            expected_outcome="Wait for explicit user command.",
            safety_level="safe",
            confirm_required=False,
            params={},
        )

    if scope == "desktop" and step_index == 0:
        return NextAction(
            action_type="ask_user",
            reason="Need user context to progress local desktop action.",
            expected_outcome="User clarifies next desktop action.",
            safety_level="safe",
            confirm_required=False,
            params={"question": "I captured your current screen. Proceed with the next action?"},
        )

    return NextAction(
        action_type="complete",
        reason="No further action inferred for this goal/state.",
        expected_outcome="Orchestration exits cleanly.",
        safety_level="safe",
        confirm_required=False,
        params={},
    )
