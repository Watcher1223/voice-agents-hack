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


def detect_active_pdf() -> Path | None:
    """Return the absolute path of the PDF in the focused window, if any."""
    app, title = _front_app_and_title()
    if not app:
        return None
    title = _normalize_title(title)
    if app not in _PDF_APPS and app not in _BROWSER_APPS:
        return None
    name = _candidate_filename(app, title)
    if not name:
        return None
    return _resolve_filename(name)


def extract_active_pdf_text(max_chars: int = 200_000) -> tuple[Path, str] | None:
    """If a PDF is in the focused window, return (path, extracted_text).

    Uses the same `_read_pdf` helper that powers the disk index, so quirks
    in malformed PDFs are handled the same way.
    """
    path = detect_active_pdf()
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
