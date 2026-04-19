"""Persisted task checklist.

Ambient-listen mode (utterances not prefixed with "Ali…") no longer fires
actions directly. Instead, every tier-3 suggestion is appended to a
persistent checklist the user reviews and ticks at their leisure — by
clicking a checkbox in the overlay or saying "run 1" / "run all" / "skip
2" / "clear tasks".

Persistence lives at ``~/.ali/tasks.json`` so unfinished tasks survive a
restart. Writes are atomic (temp file + ``os.replace``) so a crash mid-
write can't corrupt the list.

All access goes through a process-wide singleton guarded by an RLock. The
UI thread (Qt) and the asyncio agent loop both call into this module, so
concurrent reads and writes are expected.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


# Status vocabulary. `pending` is the default after add(); the rest are
# set by the executor as the task moves through its lifecycle.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

_VALID_STATUSES = {
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
}

# Tasks that aren't "live" anymore; hidden from the default pending view.
_TERMINAL_STATUSES = {STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED}

# Keep the stored file bounded. Done/failed/skipped older than this get
# trimmed on save. Pending and running rows are always retained.
_MAX_TERMINAL_TASKS = 50


_STORE_PATH = Path(
    os.environ.get("ALI_TASKS_PATH", "~/.ali/tasks.json")
).expanduser()


@dataclass
class Task:
    """One entry on the checklist.

    ``action`` carries the ambient-analysis payload (kind/text/slots) so
    the executor can reconstruct an AmbientAnalysis without re-running
    the LLM. ``result`` is filled in post-execution — a short human-
    readable summary the overlay can show.
    """

    id: str
    label: str
    detail: str
    action: dict[str, Any]
    status: str = STATUS_PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Task":
        return cls(
            id=str(raw.get("id") or uuid.uuid4().hex[:8]),
            label=str(raw.get("label", "")),
            detail=str(raw.get("detail", "")),
            action=dict(raw.get("action") or {}),
            status=str(raw.get("status", STATUS_PENDING)),
            created_at=float(raw.get("created_at", time.time())),
            updated_at=float(raw.get("updated_at", time.time())),
            result=str(raw.get("result", "")),
        )


class TaskChecklist:
    """Thread-safe in-memory checklist backed by a JSON file."""

    def __init__(self, path: Path = _STORE_PATH) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._tasks: list[Task] = []
        self._load()

    # ── IO ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            with self._path.open("r") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                return
            with self._lock:
                self._tasks = [Task.from_dict(r) for r in raw if isinstance(r, dict)]
        except Exception:
            # Corrupt file or unreadable permissions — start fresh rather
            # than crash the whole agent at boot.
            self._tasks = []

    def _save(self) -> None:
        with self._lock:
            # Trim the terminal tail so the file doesn't grow forever.
            pending = [t for t in self._tasks if t.status not in _TERMINAL_STATUSES]
            terminal = [t for t in self._tasks if t.status in _TERMINAL_STATUSES]
            terminal.sort(key=lambda t: t.updated_at, reverse=True)
            self._tasks = pending + terminal[:_MAX_TERMINAL_TASKS]

            payload = [t.to_dict() for t in self._tasks]

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            with tmp.open("w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._path)
        except Exception:
            pass

    # ── Mutators ─────────────────────────────────────────────────────────

    def add(
        self,
        label: str,
        detail: str,
        action: dict[str, Any],
    ) -> Task:
        task = Task(
            id=uuid.uuid4().hex[:8],
            label=label.strip() or "(untitled task)",
            detail=detail.strip(),
            action=dict(action or {}),
        )
        with self._lock:
            # Skip obvious duplicates so the same suggestion doesn't land
            # twice when the ambient analyser repeats itself across turns.
            for existing in self._tasks:
                if (
                    existing.status == STATUS_PENDING
                    and existing.label.lower() == task.label.lower()
                    and existing.action.get("text") == task.action.get("text")
                ):
                    return existing
            self._tasks.append(task)
        self._save()
        return task

    def update_status(
        self,
        task_id: str,
        status: str,
        result: str | None = None,
    ) -> Task | None:
        if status not in _VALID_STATUSES:
            return None
        with self._lock:
            for t in self._tasks:
                if t.id == task_id:
                    t.status = status
                    t.updated_at = time.time()
                    if result is not None:
                        t.result = result[:400]
                    break
            else:
                return None
        self._save()
        return self.get(task_id)

    def remove(self, task_id: str) -> bool:
        with self._lock:
            before = len(self._tasks)
            self._tasks = [t for t in self._tasks if t.id != task_id]
            changed = len(self._tasks) != before
        if changed:
            self._save()
        return changed

    def clear(self, include_terminal: bool = True) -> int:
        """Remove tasks. By default everything goes; pass
        ``include_terminal=False`` to keep pending tasks and only sweep
        done/failed/skipped rows."""
        with self._lock:
            before = len(self._tasks)
            if include_terminal:
                self._tasks = []
            else:
                self._tasks = [
                    t for t in self._tasks if t.status not in _TERMINAL_STATUSES
                ]
            removed = before - len(self._tasks)
        self._save()
        return removed

    # ── Accessors ────────────────────────────────────────────────────────

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            for t in self._tasks:
                if t.id == task_id:
                    return t
        return None

    def pending(self) -> list[Task]:
        with self._lock:
            return [t for t in self._tasks if t.status == STATUS_PENDING]

    def all(self) -> list[Task]:
        with self._lock:
            return list(self._tasks)

    def find_by_index(self, one_based: int) -> Task | None:
        """Map a spoken ordinal ("run 1") to the Nth pending task."""
        if one_based < 1:
            return None
        pending = self.pending()
        if one_based > len(pending):
            return None
        return pending[one_based - 1]


# Process-wide singleton. All callers should use this rather than
# constructing their own — the JSON file is a shared resource.
_singleton: TaskChecklist | None = None
_singleton_lock = threading.Lock()


def checklist() -> TaskChecklist:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = TaskChecklist()
    return _singleton


__all__ = [
    "Task",
    "TaskChecklist",
    "checklist",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
]
