#!/usr/bin/env python3
"""
Rebuild the contact-derived STT vocab cache.

Runs the AppleScript dump, applies the "unusual name" heuristics, and writes
~/.cache/ali/contact_vocab.json. Also prints a short summary so you can sanity
check which names were kept vs. dropped.

Usage:
    python scripts/refresh_contact_vocab.py

Exits 0 on success, 1 on failure (permission denial included). Safe to call
from cron or from build_index.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    from config.contact_vocab import refresh_cache

    try:
        payload = refresh_cache()
    except Exception as exc:
        print(f"[refresh-contact-vocab] failed: {exc}", file=sys.stderr)
        return 1

    names = payload.get("unusual_first_names") or []
    mis_splits = payload.get("mis_splits") or []
    preview = names[:10]
    print(f"[refresh-contact-vocab] kept {len(names)} name(s); first 10: {preview}")
    print(f"[refresh-contact-vocab] mis-split rules: {len(mis_splits)}")
    print(json.dumps({"names": len(names), "rules": len(mis_splits)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
