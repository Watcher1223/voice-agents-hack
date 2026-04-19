"""
Content extraction for the disk index.

Cheap, best-effort, bounded. Any extractor that fails or times out returns
an empty string so the file is still recorded in the metadata table but
contributes no chunks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

_TEXTUTIL = shutil.which("textutil")

# ~200KB of decoded text per file is plenty for RAG over personal docs and
# keeps embedding time bounded.
_MAX_CHARS = 200_000

_TEXT_EXTS: frozenset[str] = frozenset(
    {
        "",
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".org",
        ".log",
        ".csv",
        ".tsv",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".env",
        ".py",
        ".pyi",
        ".rb",
        ".js",
        ".mjs",
        ".cjs",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".swift",
        ".c",
        ".h",
        ".hpp",
        ".cpp",
        ".cc",
        ".cs",
        ".php",
        ".scala",
        ".lua",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".sql",
        ".tex",
        ".bib",
        ".srt",
        ".vtt",
        ".tsv",
    }
)

_HTML_EXTS: frozenset[str] = frozenset({".html", ".htm", ".xml"})
_PDF_EXTS: frozenset[str] = frozenset({".pdf"})
_DOCX_EXTS: frozenset[str] = frozenset({".docx"})
_TEXTUTIL_EXTS: frozenset[str] = frozenset(
    {".rtf", ".rtfd", ".doc", ".pages", ".key", ".numbers", ".webarchive"}
)


def extract_text(path: Path) -> str:
    """Return best-effort plain text for this file (possibly empty)."""
    ext = path.suffix.lower()
    try:
        if ext in _TEXT_EXTS:
            return _read_text(path)
        if ext in _HTML_EXTS:
            return _strip_html(_read_text(path))
        if ext in _PDF_EXTS:
            return _read_pdf(path)
        if ext in _DOCX_EXTS:
            return _read_docx(path)
        if ext in _TEXTUTIL_EXTS:
            return _read_textutil(path)
    except Exception:
        return ""
    return ""


def _read_text(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(_MAX_CHARS)
    except OSError:
        return ""


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""
    out: list[str] = []
    total = 0
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text:
            continue
        out.append(text)
        total += len(text)
        if total >= _MAX_CHARS:
            break
    return "\n".join(out)[:_MAX_CHARS]


def _read_docx(path: Path) -> str:
    try:
        import docx  # type: ignore
    except ImportError:
        return ""
    try:
        document = docx.Document(str(path))
    except Exception:
        return ""
    parts: list[str] = []
    for para in document.paragraphs:
        if para.text:
            parts.append(para.text)
        if sum(len(p) for p in parts) > _MAX_CHARS:
            break
    return "\n".join(parts)[:_MAX_CHARS]


def _read_textutil(path: Path) -> str:
    if _TEXTUTIL is None:
        return ""
    try:
        proc = subprocess.run(
            [_TEXTUTIL, "-convert", "txt", "-stdout", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.decode("utf-8", errors="replace")[:_MAX_CHARS]


class _HtmlStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs) -> None:  # noqa: ARG002
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    parser = _HtmlStripper()
    try:
        parser.feed(raw)
    except Exception:
        return ""
    return parser.get_text()[:_MAX_CHARS]


def chunk_text(text: str, *, chunk_tokens: int = 400, overlap: int = 40) -> list[str]:
    """Chunk `text` into roughly `chunk_tokens`-token pieces with overlap.

    We approximate tokens with whitespace splits — close enough for MiniLM and
    dodges the tokenizer import cost at build time.
    """
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_tokens - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_tokens]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_tokens >= len(words):
            break
    return chunks


def guess_mime(path: Path) -> str | None:
    """Lightweight mime hint based on extension."""
    ext = path.suffix.lower()
    if ext in _PDF_EXTS:
        return "application/pdf"
    if ext in _DOCX_EXTS:
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext in _HTML_EXTS:
        return "text/html"
    if ext in _TEXT_EXTS:
        return "text/plain"
    if ext in _TEXTUTIL_EXTS:
        return "application/octet-stream"
    return None


def is_text_like(path: Path) -> bool:
    return path.suffix.lower() in (
        _TEXT_EXTS | _HTML_EXTS | _PDF_EXTS | _DOCX_EXTS | _TEXTUTIL_EXTS
    )
