"""Unit tests for TasksStore: add, dedupe, mark, persistence."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from executors.local.tasks_store import TasksStore  # noqa: E402


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        store = TasksStore(path=tmp_path)
        assert store.pending() == []

        # add 3 tasks
        a = store.add("Email Hanzi", "details", "local", "compose_mail",
                      {"to": "hanzili@gmail.com", "subject": "Pitch"})
        b = store.add("Schedule pitch prep", "friday 3pm", "local",
                      "create_calendar_event", {"title": "Pitch prep", "date": "2026-04-24"})
        c = store.add("Show HN top", "", "opencli", "hackernews top", {})
        assert len(store.pending()) == 3

        # dedupe: same compose_mail action with same to+subject
        d = store.add("Email Hanzi (v2)", "again", "local", "compose_mail",
                      {"to": "hanzili@gmail.com", "subject": "Pitch"})
        assert d.id == a.id, "expected dedupe hit on to+subject"
        assert len(store.pending()) == 3

        # different subject → new task
        e = store.add("Email Hanzi follow-up", "", "local", "compose_mail",
                      {"to": "hanzili@gmail.com", "subject": "Follow-up"})
        assert e.id != a.id
        assert len(store.pending()) == 4

        # persistence
        store2 = TasksStore(path=tmp_path)
        assert len(store2.pending()) == 4
        ids2 = {t.id for t in store2.pending()}
        assert {a.id, b.id, c.id, e.id} == ids2

        # mark done removes from pending
        store.mark(a.id, "done")
        assert a.id not in {t.id for t in store.pending()}
        assert len(store.pending()) == 3

        # progress log
        store.append_progress(b.id, "step 1: found calendar")
        store.append_progress(b.id, "step 2: added event")
        got = store.get(b.id)
        assert got is not None and got.progress == ["step 1: found calendar", "step 2: added event"]

        print(f"✓ 6/6 assertions passed — store at {tmp_path}")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
