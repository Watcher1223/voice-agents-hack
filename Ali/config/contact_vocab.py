"""
Bootstrap STT bias from the user's macOS Contacts.

Most contacts already get recognised fine by Deepgram/Whisper. The biasing
budget (Deepgram caps keyterm at ~100 per request) is best spent on names
the general LM gets wrong: compound single-token first names like
"Alspencer", embedded-capital names like "DeAndre", non-English characters
like "Éloïse", and apostrophe names like "D'Angelo". Plain "Alex" / "Sam" /
"Chris" are excluded — biasing them would hijack unrelated audio.

The loader piggybacks on ContactsSource (same AppleScript that feeds the
disk index) so permission handling + name parsing are not duplicated.

Cache lives at ~/.cache/ali/contact_vocab.json (24h TTL). Refresh on demand
via scripts/refresh_contact_vocab.py, or let build_index.py warm it as
a side effect of a disk-index rebuild.

Set ALI_CONTACT_VOCAB_DISABLE=1 to opt out entirely (e.g. if Contacts
permission isn't granted and you don't want the startup probe).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

_CACHE_PATH = Path(os.path.expanduser("~/.cache/ali/contact_vocab.json"))
_CACHE_TTL_SECONDS = 24 * 60 * 60
_MAX_NAMES = 80

# Correction tuple shape matches config.vocab._Correction:
#   (wrong_set, right, hints) — empty hints == fire unconditionally.
_Rule = tuple[set[str], str, set[str]]

# Common US/UK first names — excluded from biasing because they're already
# well-handled by the base LM and biasing them tends to hijack homophones
# ("Alex" vs "LAX", "Sam" vs "some", "Mark" vs "marked"). List intentionally
# kept small; we only need to catch the high-frequency false positives.
_ENGLISH_COMMON_NAMES: frozenset[str] = frozenset(
    n.lower() for n in (
        # Male
        "James", "John", "Robert", "Michael", "William", "David", "Richard",
        "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
        "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
        "Kenneth", "Kevin", "Brian", "George", "Edward", "Ronald", "Timothy",
        "Jason", "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric",
        "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon", "Frank",
        "Benjamin", "Gregory", "Samuel", "Raymond", "Patrick", "Alexander",
        "Jack", "Dennis", "Jerry", "Tyler", "Aaron", "Henry", "Douglas",
        "Peter", "Jose", "Adam", "Zachary", "Nathan", "Walter", "Harold",
        "Kyle", "Carl", "Arthur", "Gerald", "Roger", "Keith", "Jeremy",
        "Terry", "Lawrence", "Sean", "Christian", "Albert", "Joe", "Ethan",
        "Austin", "Jesse", "Willie", "Billy", "Bryan", "Bruce", "Jordan",
        "Ralph", "Roy", "Noah", "Dylan", "Eugene", "Wayne", "Alan", "Juan",
        "Louis", "Russell", "Gabriel", "Randy", "Philip", "Harry", "Vincent",
        "Bobby", "Johnny", "Logan",
        # Female
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara",
        "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
        "Sandra", "Margaret", "Ashley", "Kimberly", "Emily", "Donna",
        "Michelle", "Carol", "Amanda", "Melissa", "Deborah", "Stephanie",
        "Dorothy", "Rebecca", "Sharon", "Laura", "Cynthia", "Amy", "Kathleen",
        "Angela", "Shirley", "Brenda", "Emma", "Anna", "Pamela", "Nicole",
        "Samantha", "Katherine", "Christine", "Helen", "Debra", "Rachel",
        "Carolyn", "Janet", "Maria", "Catherine", "Heather", "Diane", "Olivia",
        "Julie", "Joyce", "Victoria", "Ruth", "Virginia", "Lauren", "Kelly",
        "Christina", "Joan", "Evelyn", "Judith", "Andrea", "Hannah",
        "Megan", "Cheryl", "Jacqueline", "Martha", "Madison", "Teresa",
        "Gloria", "Sara", "Janice", "Ann", "Kathryn", "Abigail", "Sophia",
        "Frances", "Jean", "Alice", "Judy", "Isabella", "Julia", "Grace",
        "Amber", "Denise", "Danielle", "Marilyn", "Beverly", "Charlotte",
        "Natalie", "Theresa", "Diana", "Brittany", "Doris", "Kayla", "Alexis",
        "Lori", "Marie", "Chris", "Alex", "Sam", "Pat", "Max", "Kai", "Ava",
        "Mia", "Leo", "Noa", "Zoe", "Kim", "Lee",
    )
)


# ── Public API ────────────────────────────────────────────────────────────────


def get_unusual_first_names(refresh: bool = False) -> list[str]:
    """Return the cached list of unusual first names. Rebuild if stale or missing."""
    if _disabled():
        return []
    cache = _read_cache() if not refresh else None
    if cache is None:
        cache = refresh_cache()
    return list(cache.get("unusual_first_names") or [])[:_MAX_NAMES]


def get_mis_split_rules() -> list[_Rule]:
    """Return auto-derived (wrong_set, canonical, empty_hints) correction rules."""
    if _disabled():
        return []
    cache = _read_cache() or {}
    raw = cache.get("mis_splits") or []
    rules: list[_Rule] = []
    for entry in raw:
        try:
            wrongs = {str(w).lower() for w in entry.get("wrong", []) if w}
            right = str(entry.get("right") or "")
        except (AttributeError, TypeError):
            continue
        if not wrongs or not right:
            continue
        rules.append((wrongs, right, set()))
    return rules


def refresh_cache() -> dict:
    """Force a rebuild. Returns the freshly-written cache payload."""
    raw_names = _load_contact_first_names()
    unusual = filter_unusual(raw_names)[:_MAX_NAMES]
    mis_splits = [
        {"wrong": sorted(expand_mis_splits(name)), "right": name}
        for name in unusual
        if expand_mis_splits(name)
    ]
    payload = {
        "built_at": time.time(),
        "unusual_first_names": unusual,
        "mis_splits": mis_splits,
    }
    _write_cache(payload)
    return payload


# ── Heuristics ────────────────────────────────────────────────────────────────

_EMBEDDED_CAP = re.compile(r"[a-z][A-Z]")
_APOSTROPHE_CAP = re.compile(r"[A-Za-z][\u2019']['\u2019]?[A-Z]")


def filter_unusual(names: Iterable[str]) -> list[str]:
    """Keep only names that the STT model is likely to mis-hear.

    Order-preserving and de-duplicated — first occurrence wins so the most
    recently touched contact (if caller sorts that way) stays at the top.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in names:
        name = (raw or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        if not _is_unusual(name):
            continue
        out.append(name)
    return out


def _is_unusual(name: str) -> bool:
    if len(name) <= 2:
        return False
    if name.isupper():
        return False
    # Contacts with no real name fall back to their first email address in
    # the source's display_name; `_first_name_token` then hands us
    # "support@foo.com". Biasing on garbage tokens is useless — drop them.
    if "@" in name or name.count(".") >= 2:
        return False
    if name.lower() in _ENGLISH_COMMON_NAMES:
        return False
    if _EMBEDDED_CAP.search(name):
        return True
    if any(ord(ch) > 127 for ch in name):
        return True
    if _APOSTROPHE_CAP.search(name):
        return True
    # Long, unfamiliar single-token names — not in the common list and
    # at least 9 chars. Weakest signal, but catches things like
    # "Krzysztof" or "Siobhán" that don't hit the other rules.
    if len(name) >= 9 and " " not in name:
        return True
    return False


# ── Mis-split generation ──────────────────────────────────────────────────────

# Leading-syllable homophones. ASR commonly stretches a short leading vowel
# into its longer homophone, so "al spencer" also surfaces as "all spencer".
# Kept tiny and conservative — LHS must be a whole leading token, RHS is the
# most common confusion. Unidirectional (right never implies left).
_LEADING_HOMOPHONES: dict[str, tuple[str, ...]] = {
    "al":  ("all", "ale"),
    "an":  ("and", "ann"),
    "in":  ("inn",),
    "on":  ("own",),
    "or":  ("ore", "oar"),
    "er":  ("air",),
    "ad":  ("add",),
    "el":  ("ell",),
}

# Candidate leading prefixes for names with no explicit capital boundary. If
# the name starts with one of these (case-insensitive) and has enough letters
# after it, Deepgram is likely to split the name into "<prefix> <rest>".
# Keys must also appear as keys in _LEADING_HOMOPHONES when homophone swaps
# apply — otherwise only the raw split is emitted.
_LEADING_PREFIXES: tuple[str, ...] = (
    "al", "an", "in", "on", "or", "er", "ad", "el",
)
_PREFIX_MIN_REMAINDER = 3  # require at least 3 chars after the prefix


def expand_mis_splits(name: str) -> set[str]:
    """Phrases the ASR is likely to emit for `name` that should map back to it.

    Three strategies:

      1. Explicit capital boundary ("DeAndre") — split on [a-z][A-Z].
         Emits "de andre" + "deandre".

      2. Leading-prefix split ("Alspencer") — no capital boundary exists,
         but the name starts with a short English word like "Al". Emit
         the "al spencer" split and the "all spencer" homophone variant.

      3. Non-ASCII fold ("Éloïse") — emit the ASCII-folded lowercase form.

    Returns a set of lowercase phrases. Empty if nothing plausible applies.
    Case-insensitive replacement in `apply_corrections` means the lowercase
    joined form (e.g. "alspencer") also harmlessly rewrites to the canonical
    capitalisation.
    """
    out: set[str] = set()
    if not name:
        return out
    lname = name.lower()

    # Strategy 1: capital boundaries.
    tokens = _split_capital_run(name)
    if len(tokens) >= 2:
        out.add(" ".join(t.lower() for t in tokens))
        out.add(lname)
        first, rest_tokens = tokens[0].lower(), [t.lower() for t in tokens[1:]]
        for swap in _LEADING_HOMOPHONES.get(first, ()):
            out.add(" ".join([swap, *rest_tokens]))

    # Strategy 2: leading-prefix split for names with no capital boundary.
    # Only apply when the name is a single token (multi-word names are handled
    # by the caller's whitespace parsing) and the capital-boundary split
    # didn't already produce something.
    if len(tokens) == 1 and " " not in name:
        for prefix in _LEADING_PREFIXES:
            if len(lname) - len(prefix) < _PREFIX_MIN_REMAINDER:
                continue
            if not lname.startswith(prefix):
                continue
            rest = lname[len(prefix):]
            out.add(f"{prefix} {rest}")
            for swap in _LEADING_HOMOPHONES.get(prefix, ()):
                out.add(f"{swap} {rest}")
            out.add(lname)  # plain lowercased form of the compound name
            break  # first matching prefix wins — don't over-split

    # Strategy 3: non-ASCII fold.
    ascii_folded = _ascii_fold(name).lower()
    if ascii_folded and ascii_folded != lname:
        out.add(ascii_folded)

    return out


def _split_capital_run(name: str) -> list[str]:
    """Break a single token at [a-z][A-Z] boundaries. Leaves multi-word names alone."""
    if " " in name:
        return [name]
    parts = re.split(r"(?<=[a-z])(?=[A-Z])", name)
    return [p for p in parts if p]


def _ascii_fold(text: str) -> str:
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


# ── Contacts loader ───────────────────────────────────────────────────────────


def _load_contact_first_names() -> list[str]:
    """Pull first-name tokens from every contact. Empty list on permission denial."""
    try:
        from executors.local.disk_index.sources.contacts import ContactsSource
    except ImportError:
        return []
    src = ContactsSource()
    if not src.available():
        return []
    first_names: list[str] = []
    seen: set[str] = set()
    for doc in src.iter_docs():
        name = (doc.display_name or "").strip()
        token = _first_name_token(name)
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        first_names.append(token)
    return first_names


def _first_name_token(display_name: str) -> str:
    """Best-effort extraction of the first-name token from a Contacts display name.

    Handles "Last, First" (comma-swapped) and "First Last" (the common case).
    Falls back to the whole string if there's no whitespace.
    """
    if not display_name:
        return ""
    if "," in display_name:
        # "Tajima, Korin" -> "Korin"
        _, _, tail = display_name.partition(",")
        candidate = tail.strip().split()
        return candidate[0] if candidate else ""
    tokens = display_name.split()
    return tokens[0] if tokens else ""


# ── Cache I/O ─────────────────────────────────────────────────────────────────


def _disabled() -> bool:
    return os.environ.get("ALI_CONTACT_VOCAB_DISABLE", "").lower() in {"1", "true", "yes", "on"}


def _read_cache() -> dict | None:
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    built_at = payload.get("built_at")
    if not isinstance(built_at, (int, float)):
        return None
    if time.time() - float(built_at) > _CACHE_TTL_SECONDS:
        return None
    return payload


def _write_cache(payload: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CACHE_PATH.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"[contact-vocab] failed to write cache: {exc}")
