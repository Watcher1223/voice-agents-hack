"""
Flight search via Kiwi.com URL.

We don't call the Tequila API (now partner-gated) — we just build a search
URL and open it in the user's default browser. Kiwi renders the full
results page client-side so the demo "works" with zero auth.

URL shape:
  https://www.kiwi.com/en/search/tiles/{origin}/{destination}/{depart}/{return}

Dates are YYYY-MM-DD. Origin/destination accept city slugs ("tokyo",
"san-francisco") or IATA codes ("SFO", "NRT"). If a date is missing, Kiwi
shows "any date" on that leg, which is fine.
"""

from urllib.parse import quote

_BASE = "https://www.kiwi.com/en/search/tiles"

# Kiwi's URL expects IATA codes or fully-qualified geo slugs — bare city
# names ("tokyo") silently redirect to the homepage. Small lookup covers
# the demo set; anything else falls back to the raw slug (better than an
# error — Kiwi's search bar can sometimes resolve it client-side).
_CITY_TO_IATA = {
    # California (demo origin lives here)
    "sf": "SFO", "san francisco": "SFO", "sfo": "SFO",
    "sjc": "SJC", "san jose": "SJC",
    "oak": "OAK", "oakland": "OAK",
    "la": "LAX", "los angeles": "LAX", "lax": "LAX",
    "ontario": "ONT", "ont": "ONT",
    "san diego": "SAN", "san": "SAN",
    "sacramento": "SMF", "smf": "SMF",
    "burbank": "BUR", "bur": "BUR",
    "long beach": "LGB", "lgb": "LGB",
    "palm springs": "PSP", "psp": "PSP",
    "fresno": "FAT", "fat": "FAT",
    "santa ana": "SNA", "orange county": "SNA", "sna": "SNA",
    # Other US hubs
    "nyc": "JFK", "new york": "JFK", "jfk": "JFK",
    "lga": "LGA", "ewr": "EWR", "newark": "EWR",
    "boston": "BOS", "bos": "BOS",
    "seattle": "SEA", "sea": "SEA",
    "chicago": "ORD", "ord": "ORD",
    "austin": "AUS", "aus": "AUS",
    "denver": "DEN", "den": "DEN",
    "dallas": "DFW", "dfw": "DFW",
    "miami": "MIA", "mia": "MIA",
    "atlanta": "ATL", "atl": "ATL",
    # International
    "tokyo": "NRT", "nrt": "NRT", "hnd": "HND",
    "london": "LHR", "lhr": "LHR",
    "paris": "CDG", "cdg": "CDG",
    "lisbon": "LIS", "lis": "LIS",
    "berlin": "BER", "ber": "BER",
    "amsterdam": "AMS", "ams": "AMS",
    "dubai": "DXB", "dxb": "DXB",
    "singapore": "SIN", "sin": "SIN",
    "hong kong": "HKG", "hkg": "HKG",
    "toronto": "YYZ", "yyz": "YYZ",
    "mexico city": "MEX", "mex": "MEX",
}


def _slug(city: str) -> str:
    """IATA code if we know the city; otherwise the dash-separated slug.

    Kiwi accepts both `/sfo/nrt/...` and `/san-francisco-california-united-states/tokyo-japan/...`
    but NOT bare `/san-francisco/tokyo/...`. IATA is the safest demo choice.
    """
    key = city.strip().lower()
    if key in _CITY_TO_IATA:
        return _CITY_TO_IATA[key]
    return quote(key.replace(" ", "-"))


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
