"""
Hybrid retrieval over the disk index.

Combines:
  * SQLite FTS5 BM25 on chunk text + filename
  * hnswlib cosine search on chunk embeddings

Results are fused with reciprocal rank fusion (RRF). The caller gets back
ranked `Hit` objects with enough metadata to cite a source file and to
embed a short snippet in a RAG prompt.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import embed, store, vectors


@dataclass(frozen=True)
class Hit:
    path: str
    name: str
    snippet: str
    score: float
    mtime: float | None
    source: str  # "fts" | "vector" | "hybrid"


@dataclass(frozen=True)
class FileHit:
    path: str
    name: str
    score: float
    mtime: float | None


_RRF_K = 60


class IndexHandle:
    """Bundled read-only handles for query-time access.

    Kept thread-safe by serialising access to both the SQLite connection and
    the hnswlib index behind a single lock — the queries are fast enough that
    a mutex is cheaper than re-opening connections per request.
    """

    def __init__(
        self,
        *,
        db: sqlite3.Connection,
        vec_index,
        vec_meta,
        index_dir: Path,
    ) -> None:
        self._db = db
        self._vec_index = vec_index
        self._vec_meta = vec_meta
        self._index_dir = index_dir
        self._lock = threading.Lock()

    @property
    def vectors_available(self) -> bool:
        return self._vec_index is not None

    @property
    def embed_model(self) -> str | None:
        if self._vec_meta is None:
            return None
        return self._vec_meta.model or None

    def search_files(self, query: str, *, limit: int = 20) -> list[FileHit]:
        terms = _extract_terms(query)
        if not terms:
            return []
        match_expr = _fts_match_expression(terms, prefix=True)
        # bm25() must be called in a query that scans content_fts directly.
        # Rank chunks first, then join to files and pick the best rank per file.
        with self._lock:
            rows = self._db.execute(
                """
                WITH matched AS (
                    SELECT rowid AS chunk_id, bm25(content_fts) AS rank
                    FROM content_fts
                    WHERE content_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                )
                SELECT files.path AS path, files.name AS name, files.mtime AS mtime,
                       MIN(matched.rank) AS rank
                FROM matched
                JOIN chunks ON chunks.id = matched.chunk_id
                JOIN files  ON files.id  = chunks.file_id
                GROUP BY files.id
                ORDER BY rank
                LIMIT ?
                """,
                (match_expr, limit * 4, limit),
            ).fetchall()
        out: list[FileHit] = []
        for row in rows:
            out.append(
                FileHit(
                    path=str(row["path"]),
                    name=str(row["name"]),
                    score=float(row["rank"] or 0.0),
                    mtime=row["mtime"],
                )
            )
        return out

    def search_content(
        self,
        query: str,
        *,
        k: int = 6,
    ) -> list[Hit]:
        """Hybrid top-k retrieval with RRF."""
        fts_hits = self._fts_hits(query, k=k * 4)
        vec_hits = self._vector_hits(query, k=k * 4)
        fused = _reciprocal_rank_fusion(fts_hits, vec_hits, limit=k)
        ids = [hit_id for hit_id, _ in fused]
        details = store.lookup_chunks_by_id(self._db, ids)
        hits: list[Hit] = []
        for chunk_id, score in fused:
            if chunk_id not in details:
                continue
            path, text, mtime = details[chunk_id]
            name = Path(path).name
            snippet = _trim_snippet(text, query)
            source = _source_for(chunk_id, fts_hits, vec_hits)
            hits.append(
                Hit(
                    path=path,
                    name=name,
                    snippet=snippet,
                    score=score,
                    mtime=mtime,
                    source=source,
                )
            )
        return hits

    def _fts_hits(self, query: str, *, k: int) -> list[tuple[int, float]]:
        terms = _extract_terms(query)
        if not terms:
            return []
        match_expr = _fts_match_expression(terms, prefix=True)
        with self._lock:
            rows = self._db.execute(
                """
                SELECT rowid AS id, bm25(content_fts) AS rank
                FROM content_fts
                WHERE content_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_expr, k),
            ).fetchall()
        return [(int(r["id"]), float(r["rank"] or 0.0)) for r in rows]

    def _vector_hits(self, query: str, *, k: int) -> list[tuple[int, float]]:
        if self._vec_index is None:
            return []
        model = self.embed_model
        if not model:
            return []
        vec = embed.embed_query(query, model_name=model)
        with self._lock:
            raw = vectors.query(self._vec_index, vec, k=k)
        return raw

    def close(self) -> None:
        try:
            with self._lock:
                self._db.close()
        except Exception:
            pass


# ─── Module-level handle cache ────────────────────────────────────────────────

_handle_lock = threading.Lock()
_handle: IndexHandle | None = None
_handle_dir: Path | None = None


def get_handle(index_dir: Path) -> IndexHandle | None:
    """Open the index at `index_dir` (once per process)."""
    global _handle, _handle_dir
    with _handle_lock:
        if _handle is not None and _handle_dir == index_dir:
            return _handle
        if _handle is not None:
            _handle.close()
            _handle = None
            _handle_dir = None
        db_path = index_dir / "index.db"
        if not db_path.exists():
            return None
        conn = store.connect(db_path, create=False)
        vec_index, vec_meta = vectors.load_index(
            index_dir / "vectors.bin",
            index_dir / "vectors_meta.json",
        )
        _handle = IndexHandle(
            db=conn, vec_index=vec_index, vec_meta=vec_meta, index_dir=index_dir
        )
        _handle_dir = index_dir
        return _handle


def reset_handle() -> None:
    """Force the next `get_handle` call to reopen the DB (used after rebuild)."""
    global _handle, _handle_dir
    with _handle_lock:
        if _handle is not None:
            _handle.close()
        _handle = None
        _handle_dir = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]{1,}")
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "and", "or", "is", "are",
        "was", "were", "be", "my", "me", "i", "you", "your",
        "what", "whats", "where", "when", "who", "whom", "how",
        "do", "does", "did", "can", "could", "should", "would",
        "that", "this", "these", "those", "have", "has", "had",
        "on", "in", "for", "with", "about", "from",
    }
)


def _extract_terms(query: str) -> list[str]:
    tokens = [tok.lower() for tok in _WORD_RE.findall(query or "")]
    return [tok for tok in tokens if tok not in _STOPWORDS and len(tok) >= 2]


def _fts_match_expression(terms: list[str], *, prefix: bool) -> str:
    if not terms:
        return ""
    # Escape double quotes for FTS5 string literals, then wrap each term.
    parts: list[str] = []
    for term in terms:
        safe = term.replace('"', '""')
        quoted = f'"{safe}"'
        if prefix:
            quoted += "*"
        parts.append(quoted)
    return " OR ".join(parts)


def _reciprocal_rank_fusion(
    fts: list[tuple[int, float]],
    vec: list[tuple[int, float]],
    *,
    limit: int,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, (chunk_id, _) in enumerate(fts):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, (chunk_id, _) in enumerate(vec):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:limit]


def _source_for(
    chunk_id: int,
    fts: list[tuple[int, float]],
    vec: list[tuple[int, float]],
) -> str:
    in_fts = any(cid == chunk_id for cid, _ in fts)
    in_vec = any(cid == chunk_id for cid, _ in vec)
    if in_fts and in_vec:
        return "hybrid"
    if in_fts:
        return "fts"
    if in_vec:
        return "vector"
    return "hybrid"


def _trim_snippet(text: str, query: str, *, width: int = 320) -> str:
    text = (text or "").strip()
    if len(text) <= width:
        return text
    terms = _extract_terms(query)
    low = text.lower()
    for term in terms:
        idx = low.find(term)
        if idx >= 0:
            start = max(0, idx - width // 3)
            end = min(len(text), start + width)
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(text) else ""
            return prefix + text[start:end].strip() + suffix
    return text[:width].strip() + "…"
