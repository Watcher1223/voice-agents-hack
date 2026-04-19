"""
Layer 3 — Orchestrator
Vision-first state machine: observe, decide, act, verify.
"""

import asyncio
import time

from config.settings import DRY_RUN, VISION_FIRST_ENABLED, VISION_MAX_ACTION_STEPS
from intent.file_resolve import enrich_intent_with_resolved_files
from intent.schema import IntentObject, KnownGoal
from orchestrator.state import TaskState, TaskStatus
from orchestrator.visual_planner import NextAction, choose_next_action
from ui.confirmation import ask_confirmation

MAX_STEPS = 20
MAX_RETRIES = 2


class Orchestrator:
    def __init__(self):
        from executors.local.applescript import AppleScriptExecutor
        from executors.local.filesystem import FilesystemExecutor
        from executors.local.script_runtime import catalog_summary, load_catalog

        self._local_fs = FilesystemExecutor()
        self._local_as = AppleScriptExecutor()
        self._browser = None
        self._script_catalog = load_catalog()
        self._catalog_summary = catalog_summary
        self._reload_catalog = load_catalog

    async def run(self, intent: IntentObject):
        if intent.goal == KnownGoal.UNKNOWN:
            print(f"[orchestrator] Unknown intent: '{intent.raw_transcript}' — cannot act.")
            return

        await enrich_intent_with_resolved_files(intent, intent.raw_transcript)

        resolved = dict(intent.slots.get("resolved_local_files", {}) or {})
        state = TaskState(
            goal=intent.goal,
            plan_name="vision-first-observe-loop",
            steps=[],
            collected_data={
                **intent.slots,
                "slots": intent.slots,
                **{k: k for k in intent.uses_local_data},
                "resolved_local_files": resolved,
                "script_catalog": self._catalog_summary(self._script_catalog),
            },
        )
        if "resume" in resolved:
            state.collected_data.setdefault("resume_path", resolved["resume"])
        state.status = TaskStatus.RUNNING

        if not VISION_FIRST_ENABLED:
            raise RuntimeError("VISION_FIRST_ENABLED is false; this build requires vision-first mode.")

        print(f"[orchestrator] Starting plan: {state.plan_name}")
        observation = await self._observe(intent, "initial")
        state.collected_data["last_observation"] = observation
        print(f"[orchestrator][observe] initial scope={observation.get('scope')} path={observation.get('screenshot_path')}")

        while state.status == TaskStatus.RUNNING:
            if state.step_index >= min(MAX_STEPS, VISION_MAX_ACTION_STEPS):
                state.fail("Exceeded max steps — aborting for safety.")
                break

            step_started = time.perf_counter()
            action = await choose_next_action(
                intent=intent,
                observation=observation,
                collected_data=state.collected_data,
                step_index=state.step_index,
                max_steps=VISION_MAX_ACTION_STEPS,
            )
            print(
                f"[orchestrator][decision] step_index={state.step_index} "
                f"action_type={action.action_type} safety={action.safety_level}"
            )

            if action.action_type == "complete":
                state.status = TaskStatus.COMPLETED
                print("[orchestrator] Done: planner reported complete.")
                break
            if action.action_type == "abort":
                state.fail(action.reason or "Planner aborted.")
                break

            if action.confirm_required:
                state.status = TaskStatus.AWAITING_CONFIRMATION
                msg = _confirmation_message(action)
                approved = await ask_confirmation(msg)
                if not approved:
                    state.status = TaskStatus.ABORTED
                    print("[orchestrator] User aborted.")
                    return
                state.status = TaskStatus.RUNNING

            if self._is_dry_run_skip_action(action):
                elapsed = time.perf_counter() - step_started
                print(
                    f"[orchestrator][dry-run] Skipping irreversible action: "
                    f"{action.action_type} elapsed={elapsed:.2f}s"
                )
                state.step_index += 1
                observation = await self._observe(intent, f"post_skip_{state.step_index}")
                state.collected_data["last_observation"] = observation
                continue

            retries = 0
            while retries <= MAX_RETRIES:
                try:
                    result = await self._execute_action(intent, action, state.collected_data)
                    if isinstance(result, dict):
                        state.collected_data.update(result)
                    elapsed = time.perf_counter() - step_started
                    print(
                        f"[orchestrator][step] status=ok retries={retries} elapsed={elapsed:.2f}s "
                        f"action_type={action.action_type}"
                    )
                    state.step_index += 1
                    observation = await self._observe(intent, f"post_action_{state.step_index}")
                    state.collected_data["last_observation"] = observation
                    print(
                        f"[orchestrator][observe] scope={observation.get('scope')} "
                        f"path={observation.get('screenshot_path')}"
                    )
                    break
                except Exception as e:
                    retries += 1
                    elapsed = time.perf_counter() - step_started
                    print(
                        f"[orchestrator][step] status=failed retries={retries} "
                        f"elapsed={elapsed:.2f}s action_type={action.action_type} error={e}"
                    )

                    if retries <= MAX_RETRIES:
                        await asyncio.sleep(1)
                        continue
                    approved = await ask_confirmation(
                        f"Action '{action.action_type}' failed: {e}\n\nRetry?"
                    )
                    if approved:
                        retries = 0
                        continue
                    state.fail(str(e))
                    return

        if state.status == TaskStatus.FAILED:
            print(f"[orchestrator] Failed: {state.error}")
        elif state.status == TaskStatus.COMPLETED:
            print(f"[orchestrator] Done: {state.goal}")

    async def _execute_action(self, intent: IntentObject, action: NextAction, data: dict):
        if action.action_type == "navigate":
            return await self._run_browser("navigate", {"url": action.params["url"]})
        if action.action_type == "upload_file":
            if action.params.get("mode") != "yc_apply_fill":
                raise ValueError("Unsupported upload_file mode")
            resume_path = _path_for_file_action(data, action.params, self._local_fs)
            await self._run_browser(
                "yc_apply_fill",
                {"resume_path": resume_path, "slots": data.get("slots", {})},
            )
            return {"resume_path": resume_path, "yc_form_filled": True}
        if action.action_type == "click_text":
            if action.params.get("mode") != "yc_apply_submit":
                raise ValueError("Unsupported click_text mode")
            return await self._run_browser("yc_apply_submit", {})
        if action.action_type == "ask_user":
            approved = await ask_confirmation(action.params.get("question", action.reason))
            return {"user_confirmed": approved}
        if action.action_type == "run_script":
            return await self._handle_run_script(action, data)
        if action.action_type == "author_script":
            return await self._handle_author_script(action, data)
        if action.action_type == "compose_mail":
            return self._handle_compose_mail(action, data)
        if action.action_type in {"wait_for_text", "scroll", "click_selector", "type_selector"}:
            return {"noop_action": action.action_type}
        raise ValueError(f"Unsupported action_type: {action.action_type}")

    async def _handle_run_script(self, action: NextAction, data: dict) -> dict:
        from executors.local.script_runtime import run_script

        name = str(action.params.get("name") or "").strip()
        if not name:
            raise ValueError("run_script requires params.name")
        args = dict(action.params.get("args") or {})
        resolved = data.get("resolved_local_files") or {}
        for key, value in list(args.items()):
            if isinstance(value, str) and value.startswith("$"):
                role = value[1:]
                if isinstance(resolved, dict) and role in resolved:
                    args[key] = resolved[role]
                elif role in data and isinstance(data[role], str):
                    args[key] = data[role]
        result = await run_script(name, args, self._script_catalog)
        print(
            f"[orchestrator][script] name={name} returncode={result.returncode} "
            f"duration_ms={result.duration_ms}"
        )
        if not result.ok():
            snippet = (result.stderr or result.stdout).strip().splitlines()
            detail = snippet[-1] if snippet else "script failed"
            raise RuntimeError(f"script {name!r} exited {result.returncode}: {detail}")
        return {
            "script_result": {
                "name": result.name,
                "returncode": result.returncode,
                "duration_ms": result.duration_ms,
                "stdout_snippet": result.stdout[:200],
            }
        }

    def _handle_compose_mail(self, action: NextAction, data: dict) -> dict:
        params = action.params or {}
        to = str(params.get("to") or "")
        subject = str(params.get("subject") or "")
        body = str(params.get("body") or "")
        send = bool(params.get("send", False))

        attachments = params.get("attachments")
        if not attachments:
            resolved = data.get("resolved_local_files") or {}
            attachments = []
            if isinstance(resolved, dict):
                for role in ("attachment", "deck", "document"):
                    value = resolved.get(role)
                    if isinstance(value, str) and value:
                        attachments.append(value)
        self._local_as.compose_mail(
            to=to,
            subject=subject,
            body=body,
            send=send,
            attachments=attachments or None,
        )
        return {"mail_composed": True, "attachments_used": list(attachments or [])}

    async def _handle_author_script(self, action: NextAction, data: dict) -> dict:
        from executors.local.script_runtime import (
            ScriptParam,
            ScriptValidationError,
            persist_script,
        )

        params = action.params or {}
        name = str(params.get("name") or "").strip()
        runtime = str(params.get("runtime") or "").strip()
        description = str(params.get("description") or "").strip()
        body = str(params.get("body") or "")
        raw_params = params.get("params") or []
        parsed_params: list[ScriptParam] = []
        for entry in raw_params:
            if not isinstance(entry, dict):
                continue
            try:
                parsed_params.append(
                    ScriptParam(
                        name=str(entry.get("name", "")).strip(),
                        type=str(entry.get("type", "string")).strip() or "string",
                        required=bool(entry.get("required", True)),
                        default=(None if entry.get("default") is None else str(entry.get("default"))),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid author_script param: {exc}") from exc

        try:
            spec = persist_script(
                name=name,
                runtime=runtime,
                description=description,
                params=tuple(parsed_params),
                body=body,
            )
        except ScriptValidationError as exc:
            print(f"[orchestrator][script] author rejected name={name} reason={exc}")
            raise RuntimeError(f"author_script rejected: {exc}") from exc

        self._script_catalog = self._reload_catalog()
        catalog_list = self._catalog_summary(self._script_catalog)
        print(
            f"[orchestrator][script] authored name={spec.name} runtime={spec.runtime} "
            f"params={len(spec.params)}"
        )
        return {
            "authored_script": spec.name,
            "script_catalog": catalog_list,
        }

    async def _run_local(self, action: str, params: dict, data: dict | None = None):
        if action == "find_file":
            data = data or {}
            path = _path_for_file_action(data, params, self._local_fs)
            role = params.get("file_role") or params.get("alias") or "resume"
            return {"resume_path": path, "resolved_role": role, "resolved_path": path}
        if action == "resolve_contact":
            address = self._local_as.resolve_contact(params["name"])
            return {"contact": address, "contact_resolved": True}
        if action == "send_imessage":
            self._local_as.send_imessage(params["contact"], params["body"])
            return {}
        if action == "compose_mail":
            self._local_as.compose_mail(params["to"], params["subject"], params["body"])
            return {}
        if action == "create_calendar_event":
            self._local_as.create_calendar_event(
                params["title"], params.get("date", ""), params.get("time", ""), params.get("attendees", [])
            )
            return {}
        raise ValueError(f"Unknown local action: {action}")

    async def _run_browser(self, action: str, params: dict):
        if self._browser is None:
            try:
                from executors.browser.browser import BrowserExecutor
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Browser executor is unavailable. Install Playwright to run browser actions."
                ) from e
            self._browser = BrowserExecutor()

        if action == "navigate":
            await self._browser.navigate(params["url"])
            return {}
        if action == "yc_apply_fill":
            await self._browser.yc_apply_fill(params["resume_path"], params.get("slots", {}))
            return {}
        if action == "yc_apply_submit":
            await self._browser.yc_apply_submit()
            return {}
        if action == "capture_observation":
            return await self._browser.capture_observation(label=params.get("label", "browser"))
        raise ValueError(f"Unknown browser action: {action}")

    async def _observe(self, intent: IntentObject, label: str) -> dict:
        if intent.requires_browser:
            return await self._run_browser("capture_observation", {"label": label})
        return self._local_as.capture_observation(label=label)

    def _is_dry_run_skip_action(self, action: NextAction) -> bool:
        return DRY_RUN and action.safety_level == "irreversible"


def _path_for_file_action(data: dict, params: dict, local_fs) -> str:
    """
    Resolve an absolute file path for actions like upload_file / find_file /
    run_script. Preference order:
      1. resolved_local_files[params.file_role]
      2. resolved_local_files[params.alias]  (legacy callers)
      3. data["resume_path"] when the role is "resume" (or unspecified)
      4. FilesystemExecutor.find_by_alias(role)
    """
    resolved = data.get("resolved_local_files") or {}
    role = params.get("file_role") or params.get("alias") or "resume"
    if isinstance(resolved, dict):
        candidate = resolved.get(role)
        if isinstance(candidate, str) and candidate:
            return candidate
    if role == "resume":
        existing = data.get("resume_path")
        if isinstance(existing, str) and existing:
            return existing
    return local_fs.find_by_alias(role)


def _resolve_params(params: dict, data: dict) -> dict:
    resolved = {}
    for k, v in params.items():
        if isinstance(v, str) and v.startswith("$"):
            key = v[1:]
            resolved[k] = data.get(key, v)
        else:
            resolved[k] = v
    return resolved


def _confirmation_message(action: NextAction) -> str:
    safe_params = {k: v for k, v in action.params.items() if k != "slots"}
    detail = "\n".join(f"  {k}: {v}" for k, v in safe_params.items())
    return (
        f"About to execute: {action.action_type}\n"
        f"Reason: {action.reason}\n"
        f"{detail}\n\nProceed?"
    )
