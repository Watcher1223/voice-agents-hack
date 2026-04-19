"""Persistent task list for ambient-mode suggestions.

The agent surfaces tier-3 action suggestions (email Hanzi, schedule
pitch prep, etc.) into this store rather than forcing an immediate
yes/no decision. Tasks live at ~/.ali/tasks.json and show up in the
right-edge side panel; the user approves or dismisses each on their
own timeline.

Dedupe rule: if there's already a pending task with the same action
kind + action text whose key slots match, we don't add a duplicate.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_STORE_PATH = Path(os.environ.get("VOICE_AGENT_TASKS_FILE", "~/.ali/tasks.json")).expanduser()
_MAX_TASKS = 40


@dataclass
class Task:
    id: str
    created_at: float
    headline: str
    detail: str
    action_kind: str          # "local" | "opencli"
    action_text: str           # e.g. "compose_mail", "hackernews top"
    slots: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"    # pending | approved | dismissed | executing | done | failed
    progress: list[str] = field(default_factory=list)  # per-step log for long-horizon runs


class TasksStore:
    """Thread-safe-ish JSON-backed list. Readers/writers coordinate
    through the UI's own event loop; worst case we miss a refresh.
    """

    def __init__(self, path: Path = _STORE_PATH) -> None:
        self._path = path
        self.tasks: list[Task] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self.tasks = [Task(**t) for t in data.get("tasks", [])]
        except Exception:
            self.tasks = []

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            blob = {"tasks": [asdict(t) for t in self.tasks]}
            self._path.write_text(json.dumps(blob, indent=2))
        except Exception:
            pass

    # ── API ──────────────────────────────────────────────────────────────

    def add(
        self,
        headline: str,
        detail: str,
        action_kind: str,
        action_text: str,
        slots: dict[str, Any] | None = None,
    ) -> Task:
        slots = dict(slots or {})
        existing = self._find_matching_pending(action_kind, action_text, slots)
        if existing is not None:
            return existing
        task = Task(
            id=uuid.uuid4().hex[:8],
            created_at=time.time(),
            headline=headline.strip(),
            detail=detail.strip(),
            action_kind=action_kind,
            action_text=action_text,
            slots=slots,
        )
        self.tasks.insert(0, task)   # newest first
        self.tasks = self.tasks[:_MAX_TASKS]
        self._save()
        return task

    def mark(self, task_id: str, status: str) -> Task | None:
        for t in self.tasks:
            if t.id == task_id:
                t.status = status
                self._save()
                return t
        return None

    def append_progress(self, task_id: str, line: str) -> None:
        for t in self.tasks:
            if t.id == task_id:
                t.progress.append(line)
                t.progress = t.progress[-8:]    # keep recent
                self._save()
                return

    def update_slots(self, task_id: str, slots: dict[str, Any]) -> None:
        for t in self.tasks:
            if t.id == task_id:
                t.slots = dict(slots)
                self._save()
                return

    def get(self, task_id: str) -> Task | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def pending(self) -> list[Task]:
        return [t for t in self.tasks if t.status in ("pending", "executing")]

    def recent(self, limit: int = 10) -> list[Task]:
        return self.tasks[:limit]

    # ── dedupe ───────────────────────────────────────────────────────────

    def _find_matching_pending(
        self, kind: str, text: str, slots: dict[str, Any]
    ) -> Task | None:
        for t in self.tasks:
            if t.status != "pending":
                continue
            if t.action_kind != kind or t.action_text != text:
                continue
            if _key_slots_match(t.action_text, t.slots, slots):
                return t
        return None


# Which slot keys distinguish two compose_mail drafts from being the same task?
_DEDUPE_KEYS = {
    "compose_mail":          ("to", "subject"),
    "send_email":            ("to", "subject"),
    "send_imessage":         ("contact", "body"),
    "send_message":          ("contact", "body"),
    "create_calendar_event": ("title", "date"),
    "add_calendar_event":    ("title", "date"),
    "find_file":             ("file_query",),
    "open_url":              ("url",),
}


def _key_slots_match(action_text: str, a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = _DEDUPE_KEYS.get(action_text)
    if not keys:
        # Unknown action — compare all slot keys verbatim.
        return _raw_eq(a, b)
    for k in keys:
        va = str(a.get(k, "")).strip().lower()
        vb = str(b.get(k, "")).strip().lower()
        if va != vb:
            return False
    return True


def _raw_eq(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return {k: str(v).strip().lower() for k, v in a.items()} == {
        k: str(v).strip().lower() for k, v in b.items()
    }
