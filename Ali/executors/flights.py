"""
Flight search via Kiwi.com URL.

We don't call the Tequila API (now partner-gated) — we just build a search
URL and open it in the user's default browser. Kiwi renders the full
results page client-side so the demo "works" with zero auth.

URL shape:
  https://www.kiwi.com/en/search/results/{origin}/{destination}/{depart}/{return}

Dates are YYYY-MM-DD. Origin/destination accept city slugs ("tokyo",
"san-francisco") or IATA codes ("SFO", "NRT"). If a date is missing, Kiwi
shows "any date" on that leg, which is fine.
"""

from urllib.parse import quote

_BASE = "https://www.kiwi.com/en/search/results"


def _slug(city: str) -> str:
    """Normalise a city name into the dash-separated slug Kiwi expects."""
    return quote(city.strip().lower().replace(" ", "-"))


def build_kiwi_url(slots: dict) -> str:
    """Build a Kiwi search URL from flight slots.

    Slots shape (all optional except origin+destination):
        origin:      "San Francisco" | "SFO"
        destination: "Tokyo" | "NRT"
        depart_date: "2026-05-01"  (YYYY-MM-DD)
        return_date: "2026-05-08"  (YYYY-MM-DD, omit for one-way)
    """
    origin = _slug(str(slots.get("origin") or ""))
    destination = _slug(str(slots.get("destination") or ""))
    if not origin or not destination:
        raise ValueError("build_kiwi_url requires origin and destination")

    parts = [_BASE, origin, destination]
    depart = str(slots.get("depart_date") or "").strip()
    return_ = str(slots.get("return_date") or "").strip()
    if depart:
        parts.append(depart)
        if return_:
            parts.append(return_)
    return "/".join(parts)
