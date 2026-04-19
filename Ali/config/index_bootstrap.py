"""
Startup bootstrap for the disk index.

Invoked once from `main.py` right after preflight. If the index is missing
(or the user asked for a rebuild), we spawn `scripts/build_index.py` as a
subprocess so the heavyweight embedding step runs in a clean interpreter
and doesn't share state with the agent's asyncio loop.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
_BUILD_SCRIPT = _ROOT / "scripts" / "build_index.py"


def ensure_index(*, force_rebuild: bool = False) -> None:
    """Ensure the disk index exists; rebuild in a subprocess if not."""
    from executors.local.disk_index import index_exists, index_stats, reset_handle

    if not force_rebuild and index_exists():
        stats = index_stats()
        if stats is not None:
            print(
                f"[index] ready — {stats.files} files, {stats.chunks} chunks, "
                f"built {stats.age}"
            )
        return

    if force_rebuild:
        print("[index] rebuild requested — indexing your laptop…")
    else:
        print(
            "[index] first run — indexing your laptop (one-time, "
            "can take a few minutes)"
        )

    proc = subprocess.run(
        [sys.executable, str(_BUILD_SCRIPT)],
        cwd=str(_ROOT),
    )

    if proc.returncode != 0:
        print(
            "[index][warn] build failed; continuing with whatever partial "
            "index exists (you can retry later from the menu bar)."
        )
        return

    reset_handle()
    stats = index_stats()
    if stats is not None:
        print(
            f"[index] built — {stats.files} files, {stats.chunks} chunks."
        )


def parse_build_summary(stdout: bytes | str) -> dict | None:
    """Pull the final JSON line the build script emits (for callers that
    capture stdout instead of streaming it)."""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="ignore")
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None
