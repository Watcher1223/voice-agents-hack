"""
Layer 4A — Local Executor: Filesystem
Named alias lookup so the agent can reference "resume" instead of full paths.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path

from config.resources import FILE_ALIASES


# Stopwords / suffixes that aren't useful as a Spotlight term.
_QUERY_STOPWORDS = {
    "find", "my", "the", "a", "an", "of", "to", "for", "and", "or",
    "please", "can", "you", "file", "files", "document", "doc", "docx",
    "pdf", "attach", "attachment", "send", "email", "mail", "text",
    "message", "imessage", "hanzi",
}

# Coarse alias aliases for fuzzy resume/cv/cover-letter matches.
_ALIAS_SYNONYMS = (
    (("resume", "cv"), "resume"),
    (("cover letter", "coverletter", "cover_letter"), "cover_letter"),
)


class FilesystemExecutor:
    def find_by_alias(self, alias: str) -> str:
        """
        Return the absolute path for a named file alias.
        Raises FileNotFoundError if the alias is not configured or the file is missing.
        """
        path = FILE_ALIASES.get(alias)
        if not path:
            raise FileNotFoundError(
                f"No file alias '{alias}' configured. Add it to config/resources.py."
            )
        expanded = os.path.expanduser(path)
        if not os.path.exists(expanded):
            raise FileNotFoundError(
                f"File alias '{alias}' points to '{expanded}' but the file does not exist."
            )
        return expanded

    def read_text(self, alias: str) -> str:
        path = self.find_by_alias(alias)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def list_aliases(self) -> dict[str, str]:
        return dict(FILE_ALIASES)


def _match_alias_synonym(query: str) -> str | None:
    q = query.lower()
    for phrases, alias_key in _ALIAS_SYNONYMS:
        for phrase in phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", q):
                return alias_key
    return None


def _extract_query_terms(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    kept: list[str] = []
    for tok in tokens:
        if len(tok) < 2:
            continue
        if tok in _QUERY_STOPWORDS:
            continue
        if tok not in kept:
            kept.append(tok)
    kept.sort(key=len, reverse=True)
    return kept


def _try_disk_index(query: str) -> str | None:
    try:
        from executors.local.disk_index import index_exists, search_files
    except Exception:
        return None
    try:
        if not index_exists():
            return None
    except Exception:
        return None
    try:
        hits = search_files(query, limit=6)
    except Exception:
        return None
    for hit in hits or []:
        path = getattr(hit, "path", "")
        if not path or path.startswith("ali://"):
            continue
        if os.path.isfile(path):
            return path
    return None


def _try_mdfind(query: str) -> str | None:
    mdfind_bin = shutil.which("mdfind")
    if not mdfind_bin:
        return None
    terms = _extract_query_terms(query)
    if not terms:
        return None

    home = os.path.expanduser("~")
    roots = [
        os.path.join(home, "Documents"),
        os.path.join(home, "Desktop"),
        os.path.join(home, "Downloads"),
        home,
    ]

    def _run(predicate: str, root: str) -> list[str]:
        try:
            result = subprocess.run(
                [mdfind_bin, "-onlyin", root, predicate],
                capture_output=True,
                text=True,
                timeout=4.0,
            )
        except (subprocess.TimeoutExpired, OSError):
            return []
        if result.returncode != 0:
            return []
        return [ln for ln in result.stdout.splitlines() if ln.strip()]

    # Try a combined AND predicate first; fall back to the single most
    # distinctive term.
    candidates: list[str] = []
    if len(terms) >= 2:
        pred = " && ".join(f'kMDItemDisplayName == "*{t}*"c' for t in terms[:3])
        for root in roots:
            candidates.extend(_run(pred, root))
            if candidates:
                break
    if not candidates:
        pred = f'kMDItemDisplayName == "*{terms[0]}*"c'
        for root in roots:
            candidates.extend(_run(pred, root))
            if candidates:
                break
    if not candidates:
        return None

    doc_exts = (".pdf", ".docx", ".doc", ".pages", ".rtf", ".txt", ".md",
                ".pptx", ".ppt", ".key", ".xlsx", ".xls", ".csv", ".numbers")

    def _score(path: str) -> tuple:
        p = Path(path)
        ext = p.suffix.lower()
        ext_rank = doc_exts.index(ext) if ext in doc_exts else 999
        name = p.stem.lower()
        term_hits = sum(1 for t in terms if t in name)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (ext_rank, -term_hits, -mtime, len(p.name))

    valid = [c for c in candidates if os.path.isfile(c)]
    if not valid:
        return None
    valid.sort(key=_score)
    return valid[0]


def resolve_file_query_to_path(query: str) -> str | None:
    """Resolve a natural-language file reference ("Q1 Report", "my resume")
    to an absolute path on disk.

    Strategy (fastest first):
      1. Alias-synonym match against ``FILE_ALIASES`` (resume / cover letter).
      2. Pre-built SQLite/FTS disk index (``search_files``).
      3. Spotlight ``mdfind`` with a best-effort predicate.

    Returns ``None`` when nothing plausible is found. Never raises."""
    if not query or not query.strip():
        return None
    query = query.strip()

    alias_key = _match_alias_synonym(query)
    if alias_key is not None:
        raw = FILE_ALIASES.get(alias_key)
        if raw:
            expanded = os.path.expanduser(raw)
            if os.path.isfile(expanded):
                return expanded

    indexed = _try_disk_index(query)
    if indexed is not None:
        return indexed

    return _try_mdfind(query)


async def resolve_file_query_to_path_async(query: str) -> str | None:
    """Async wrapper — runs the blocking resolver in a thread so callers on
    the asyncio loop (orchestrator, ambient executor) don't stall on
    Spotlight or SQLite I/O."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, resolve_file_query_to_path, query)
