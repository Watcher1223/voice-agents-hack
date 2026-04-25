"""Detect a PDF that's open in the foreground and extract its text.

Used when the user asks a question while a PDF viewer (Preview, Chrome,
Safari, Adobe Reader, …) is focused — we want to ground the answer in the
visible doc rather than the disk-wide RAG index.

Best-effort: if the title doesn't resolve to a real file, return None and
let the caller fall back to the normal answer path.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

# Apps whose front window is always a PDF.
_PDF_APPS: frozenset[str] = frozenset({
    "Preview",
    "Adobe Acrobat",
    "Adobe Acrobat Reader",
    "Adobe Acrobat Reader DC",
    "Adobe Reader",
    "Skim",
    "PDF Expert",
    "Highlights",
    "FoxitReader",
    "Foxit Reader",
})

# Browsers — only treat as a PDF if the title contains ".pdf".
_BROWSER_APPS: frozenset[str] = frozenset({
    "Google Chrome",
    "Safari",
    "Arc",
    "Firefox",
    "Brave Browser",
    "Microsoft Edge",
})

# Preview decorates titles with " — Edited" / " - Modified". Strip those.
_EDIT_SUFFIX_RE = re.compile(r"\s+[—\-]\s+(Edited|Modified)\s*$", re.IGNORECASE)
# Browsers append " — Google Chrome" / " - Safari" etc. to the tab title.
_BROWSER_SUFFIX_RE = re.compile(
    r"\s+[—\-]\s+(Google Chrome|Safari|Arc|Firefox|Brave Browser|Microsoft Edge)\s*$"
)


def _front_app_and_title() -> tuple[str, str]:
    """One osascript round-trip for app name + frontmost window title."""
    script = (
        'tell application "System Events"\n'
        '  set frontApp to name of first application process whose frontmost is true\n'
        '  try\n'
        '    set frontWin to name of front window of application process frontApp\n'
        '  on error\n'
        '    set frontWin to ""\n'
        '  end try\n'
        'end tell\n'
        'return frontApp & "||" & frontWin'
    )
    try:
        out = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=1.5,
        )
        text = (out.stdout or "").strip()
        if "||" in text:
            app, title = text.split("||", 1)
            return app.strip(), title.strip()
    except Exception:
        pass
    return "", ""


def _normalize_title(title: str) -> str:
    title = _EDIT_SUFFIX_RE.sub("", title).strip()
    title = _BROWSER_SUFFIX_RE.sub("", title).strip()
    return title


def _candidate_filename(app: str, title: str) -> str | None:
    """Pick a likely PDF filename from a window title, or None."""
    if not title:
        return None
    # Title already contains an explicit ".pdf" segment — pull it out.
    m = re.search(r"([^/\\:?*\"]+\.pdf)\b", title, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Native PDF apps usually show just the stem ("Lecture 4 notes").
    # Try appending ".pdf" so Spotlight can find it.
    if app in _PDF_APPS:
        return f"{title}.pdf"
    return None


def _resolve_filename(name: str) -> Path | None:
    """Use mdfind (Spotlight) to map a filename to an absolute path.

    Picks the most-recently-modified match if multiple files share the
    name — that's almost always the one the user is looking at.
    """
    name = name.replace('"', '')
    if not name:
        return None
    try:
        out = subprocess.run(
            ["mdfind", f'kMDItemFSName == "{name}"'],
            capture_output=True, text=True, timeout=2.0,
        )
    except Exception:
        return None
    candidates: list[Path] = []
    for line in (out.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if p.is_file():
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# Sticky cache: focus often moves to Cursor / Terminal between the user
# looking at the PDF and Ali processing the question. Stamp every successful
# detection so callers can fall back to "the PDF you were just on" within a
# bounded time window.
_last_detected: tuple[Path, float] | None = None


def detect_active_pdf() -> Path | None:
    """Return the absolute path of the PDF in the focused window, if any."""
    global _last_detected
    app, title = _front_app_and_title()
    if not app:
        return None
    title = _normalize_title(title)
    if app not in _PDF_APPS and app not in _BROWSER_APPS:
        return None
    name = _candidate_filename(app, title)
    if not name:
        return None
    path = _resolve_filename(name)
    if path is not None:
        _last_detected = (path, time.monotonic())
    return path


def note_focus(app: str, title: str) -> Path | None:
    """Resolve (app, title) to a PDF path and stamp the cache, if applicable.

    Lets the screen observer warm the cache passively as the user moves
    around — by the time they ask a question, the most recently focused
    PDF is already cached, even if focus has since moved to Ali / a code
    editor / a terminal.
    """
    global _last_detected
    if not app:
        return None
    title = _normalize_title(title or "")
    if app not in _PDF_APPS and app not in _BROWSER_APPS:
        return None
    name = _candidate_filename(app, title)
    if not name:
        return None
    path = _resolve_filename(name)
    if path is not None:
        _last_detected = (path, time.monotonic())
    return path


def recent_active_pdf(window_seconds: float = 120.0) -> Path | None:
    """Return the most recently focused PDF if it's still warm.

    Falls back to this when ``detect_active_pdf()`` returns None — typically
    because the user has switched to Cursor/Terminal/Ali while waiting for
    the answer. The PDF on disk is still the right doc to ground in.
    """
    if _last_detected is None:
        return None
    path, when = _last_detected
    if time.monotonic() - when > window_seconds:
        return None
    if not path.is_file():
        return None
    return path


def extract_active_pdf_text(
    max_chars: int = 200_000,
    *,
    fallback_window_seconds: float = 120.0,
) -> tuple[Path, str] | None:
    """If a PDF is in the focused window (or was recently), return (path, text).

    Uses the same `_read_pdf` helper that powers the disk index, so quirks
    in malformed PDFs are handled the same way. Falls back to the most
    recently focused PDF (within ``fallback_window_seconds``) when the
    current foreground window isn't a PDF — typical when Ali processes the
    question after focus has moved to its own UI.
    """
    path = detect_active_pdf() or recent_active_pdf(fallback_window_seconds)
    if path is None:
        return None
    try:
        from executors.local.disk_index.extract import extract_text
    except Exception:
        return None
    text = extract_text(path) or ""
    if not text.strip():
        return None
    return path, text[:max_chars]
