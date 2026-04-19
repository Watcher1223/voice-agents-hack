"""
Orchestrator for a full index build.

Pipeline:
  1. Discover candidate files (bounded walk, deny-list).
  2. Extract text (best-effort, per-extension).
  3. Chunk + insert into SQLite; FTS5 picks up rows via triggers.
  4. Embed all chunks (MiniLM, batch).
  5. Build hnswlib HNSW index, save to disk.
  6. Build user profile JSON.

Runs in a subprocess (see `scripts/build_index.py`) so the agent loop never
shares the embedder's big tensors with the main event loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import discovery, embed, extract, profile, store, vectors


ProgressFn = Callable[[str, dict], None]


@dataclass
class BuildConfig:
    index_dir: Path
    scan_roots: list[Path]
    max_file_bytes: int
    embed_model: str
    enable_embeddings: bool
    chunk_tokens: int
    resume_path: str | None


@dataclass
class BuildResult:
    files: int
    chunks: int
    embedded: int
    duration_s: float


def run_build(
    cfg: BuildConfig,
    *,
    progress: ProgressFn | None = None,
) -> BuildResult:
    started = time.time()
    _emit(progress, "start", {"index_dir": str(cfg.index_dir)})

    db_path = cfg.index_dir / "index.db"
    vec_bin = cfg.index_dir / "vectors.bin"
    vec_meta = cfg.index_dir / "vectors_meta.json"
    profile_path = cfg.index_dir / "profile.json"

    # Rebuild: drop the old DB so triggers/FTS rebuild cleanly.
    if db_path.exists():
        db_path.unlink()
    if vec_bin.exists():
        vec_bin.unlink()
    if vec_meta.exists():
        vec_meta.unlink()

    conn = store.connect(db_path, create=True)
    try:
        conn.execute("BEGIN")
        file_count = 0
        chunk_count = 0
        for cand in discovery.iter_candidates(
            cfg.scan_roots, max_file_bytes=cfg.max_file_bytes
        ):
            file_count += 1
            content = extract.extract_text(cand.path)
            chunks = (
                extract.chunk_text(content, chunk_tokens=cfg.chunk_tokens)
                if content
                else []
            )
            file_id = store.upsert_file(
                conn,
                path=str(cand.path),
                name=cand.path.name,
                ext=cand.ext or None,
                size=cand.size,
                mtime=cand.mtime,
                mime=extract.guess_mime(cand.path),
                content_ok=bool(chunks),
            )
            if chunks:
                store.clear_chunks(conn, file_id)
                store.insert_chunks(conn, file_id, chunks)
                chunk_count += len(chunks)
            if file_count % 500 == 0:
                _emit(
                    progress,
                    "progress",
                    {
                        "files": file_count,
                        "chunks": chunk_count,
                        "stage": "extract",
                    },
                )
                conn.execute("COMMIT")
                conn.execute("BEGIN")
        conn.execute("COMMIT")
        _emit(
            progress,
            "extract_done",
            {"files": file_count, "chunks": chunk_count},
        )

        embedded = 0
        if cfg.enable_embeddings and chunk_count > 0:
            embedded = _build_vectors(
                conn,
                vec_bin=vec_bin,
                vec_meta=vec_meta,
                model_name=cfg.embed_model,
                progress=progress,
            )
        else:
            _emit(progress, "embed_skipped", {})

        store.set_manifest(conn, "built_at", str(time.time()))
        store.set_manifest(conn, "files", str(file_count))
        store.set_manifest(conn, "chunks", str(chunk_count))
        store.set_manifest(conn, "embedded", str(embedded))
        store.set_manifest(conn, "embed_model", cfg.embed_model)

    finally:
        conn.close()

    _emit(progress, "profile_start", {})
    try:
        profile.build_profile(resume_path=cfg.resume_path, output_path=profile_path)
    except Exception as exc:
        _emit(progress, "profile_error", {"err": str(exc)[:200]})
    _emit(progress, "profile_done", {})

    duration = time.time() - started
    result = BuildResult(
        files=file_count,
        chunks=chunk_count,
        embedded=embedded,
        duration_s=duration,
    )
    _emit(
        progress,
        "done",
        {
            "files": result.files,
            "chunks": result.chunks,
            "embedded": result.embedded,
            "duration_s": round(duration, 1),
        },
    )
    return result


def _build_vectors(
    conn,
    *,
    vec_bin: Path,
    vec_meta: Path,
    model_name: str,
    progress: ProgressFn | None,
) -> int:
    ids: list[int] = []
    texts: list[str] = []
    for chunk_id, text in store.iter_chunks_for_vector_build(conn):
        ids.append(chunk_id)
        texts.append(text)

    if not ids:
        return 0

    _emit(progress, "embed_start", {"count": len(ids)})
    batch_size = 64
    import numpy as np

    all_vecs = np.zeros((len(ids), embed.EMBED_DIM), dtype="float32")
    for start in range(0, len(ids), batch_size):
        chunk_texts = texts[start : start + batch_size]
        vecs = embed.embed_texts(
            chunk_texts,
            model_name=model_name,
            batch_size=batch_size,
            show_progress=False,
        )
        all_vecs[start : start + len(chunk_texts)] = vecs
        if (start // batch_size) % 20 == 0:
            _emit(
                progress,
                "embed_progress",
                {"done": start + len(chunk_texts), "total": len(ids)},
            )

    _emit(progress, "vector_build_start", {"count": len(ids)})
    vectors.build_index(
        vec_bin,
        vec_meta,
        ids=ids,
        vectors=all_vecs,
        model_name=model_name,
    )
    _emit(progress, "vector_build_done", {"count": len(ids)})
    return len(ids)


def _emit(progress: ProgressFn | None, event: str, data: dict) -> None:
    if progress is not None:
        try:
            progress(event, data)
        except Exception:
            pass
    else:
        print(f"[disk-index] {event} {data}")
