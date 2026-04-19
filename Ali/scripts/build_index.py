#!/usr/bin/env python3
"""
Build or rebuild the Ali laptop-wide disk index.

Used by:
  * the startup bootstrap (runs once, first launch)
  * the menu bar "Rebuild Index…" item
  * manual invocation:  python scripts/build_index.py

Emits tqdm-style progress to stderr and prints a JSON summary on the last
line of stdout, so it's easy to wrap from Python.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running as `python scripts/build_index.py` from the project root.
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the Ali disk index")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress output (final JSON summary still emitted)",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="skip the embedding pass (metadata + FTS only)",
    )
    args = parser.parse_args(argv)

    # Env tweak so tokenizers don't spam warnings in subprocess output.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from config.resources import FILE_ALIASES
    from config.settings import (
        INDEX_CHUNK_TOKENS,
        INDEX_DIR,
        INDEX_EMBED_MODEL,
        INDEX_ENABLE_EMBEDDINGS,
        INDEX_MAX_FILE_BYTES,
        INDEX_SCAN_ROOTS,
    )
    from executors.local.disk_index.build import BuildConfig, run_build

    resume_raw = FILE_ALIASES.get("resume")
    resume_path = os.path.expanduser(resume_raw) if resume_raw else None
    if resume_path and not Path(resume_path).is_file():
        resume_path = None

    cfg = BuildConfig(
        index_dir=Path(INDEX_DIR),
        scan_roots=list(INDEX_SCAN_ROOTS),
        max_file_bytes=INDEX_MAX_FILE_BYTES,
        embed_model=INDEX_EMBED_MODEL,
        enable_embeddings=INDEX_ENABLE_EMBEDDINGS and not args.no_embeddings,
        chunk_tokens=INDEX_CHUNK_TOKENS,
        resume_path=resume_path,
    )

    started = time.time()
    if not args.quiet:
        roots_display = ", ".join(str(r) for r in cfg.scan_roots) or "(none)"
        print(f"[build_index] scan roots: {roots_display}", file=sys.stderr)
        print(
            f"[build_index] embed_model={cfg.embed_model} "
            f"embeddings={'on' if cfg.enable_embeddings else 'off'}",
            file=sys.stderr,
        )

    def progress(event: str, data: dict) -> None:
        if args.quiet:
            return
        if event == "progress":
            files = data.get("files", 0)
            chunks = data.get("chunks", 0)
            print(
                f"[build_index] extract… files={files} chunks={chunks}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "extract_done":
            print(
                f"[build_index] extract done: files={data.get('files')} "
                f"chunks={data.get('chunks')}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "embed_progress":
            print(
                f"[build_index] embed… {data.get('done')}/{data.get('total')}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "vector_build_start":
            print(
                f"[build_index] building HNSW (count={data.get('count')})",
                file=sys.stderr,
                flush=True,
            )
        elif event == "profile_start":
            print("[build_index] building user profile…", file=sys.stderr, flush=True)
        elif event == "profile_error":
            print(
                f"[build_index][warn] profile error: {data.get('err')}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "done":
            print(
                f"[build_index] done in {data.get('duration_s')}s",
                file=sys.stderr,
                flush=True,
            )

    try:
        result = run_build(cfg, progress=progress)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "duration_s": round(time.time() - started, 2),
                }
            )
        )
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "files": result.files,
                "chunks": result.chunks,
                "embedded": result.embedded,
                "duration_s": round(result.duration_s, 2),
                "index_dir": str(cfg.index_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
