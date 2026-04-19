"""
Flight search via Kiwi's public MCP server (https://mcp.kiwi.com/).

Why MCP over URL-building: Kiwi's search-URL format (`/search/tiles/...`)
silently redirects to the homepage unless you pass fully-qualified geo
slugs, and the city-slug format is undocumented. The MCP server returns
structured JSON (price, duration, deeplink) with zero auth — just an
MCP initialize handshake over HTTP.

We call it over plain `urllib` (no MCP client dependency) so the voice
agent stays lean. Responses come back as Server-Sent Events; we pluck
the first `data:` line.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

_MCP_URL = "https://mcp.kiwi.com/"
_PROTOCOL_VERSION = "2025-06-18"
_TIMEOUT_S = 20


class FlightSearchError(RuntimeError):
    pass


def _post(body: dict, session_id: str | None) -> tuple[dict, str | None]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": _PROTOCOL_VERSION,
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    req = urllib.request.Request(
        _MCP_URL,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        sid = resp.headers.get("mcp-session-id") or session_id
        raw = resp.read().decode()

    # SSE format — response may contain multiple `data: {...}` frames (e.g.,
    # progress notifications followed by the real result). Walk every frame;
    # prefer one with "result" or "error" whose id matches our request.
    wanted_id = body.get("id")
    fallback: dict = {}
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            frame = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if wanted_id is not None and frame.get("id") == wanted_id and ("result" in frame or "error" in frame):
            return frame, sid
        fallback = frame
    return fallback, sid


def _to_kiwi_date(iso_date: str) -> str:
    """Convert YYYY-MM-DD (our internal) to DD/MM/YYYY (Kiwi MCP requires)."""
    y, m, d = iso_date.split("-")
    return f"{d}/{m}/{y}"


def _search_sync(flights_from: str, flights_to: str, depart: str, return_date: str | None) -> list[dict]:
    # 1. initialize
    _, sid = _post(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "ali-voice-agent", "version": "0.1"},
            },
        },
        None,
    )
    if not sid:
        raise FlightSearchError("Kiwi MCP did not return a session id")

    # 2. notifications/initialized (required by MCP handshake)
    _post({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)

    # 3. search-flight
    args: dict[str, Any] = {
        "flyFrom": flights_from,
        "flyTo": flights_to,
        "departureDate": _to_kiwi_date(depart),
        "passengers": {"adults": 1},
        "curr": "USD",
        "locale": "en",
        "sort": "price",
    }
    if return_date:
        args["returnDate"] = _to_kiwi_date(return_date)

    resp, _ = _post(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "search-flight", "arguments": args},
        },
        sid,
    )
    if "error" in resp:
        raise FlightSearchError(resp["error"].get("message", "unknown MCP error"))

    content = resp.get("result", {}).get("content", [])
    for chunk in content:
        if chunk.get("type") == "text":
            try:
                parsed = json.loads(chunk["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                return parsed
    return []


async def search_flights(slots: dict) -> list[dict]:
    """Search Kiwi for flights matching the given slots.

    Returns a list of flight dicts sorted by price ascending. Raises
    FlightSearchError on network/MCP failure or invalid slots.
    """
    origin = str(slots.get("origin") or "").strip()
    destination = str(slots.get("destination") or "").strip()
    depart = str(slots.get("depart_date") or "").strip()
    return_date = str(slots.get("return_date") or "").strip() or None
    if not origin or not destination:
        raise FlightSearchError("Need both origin and destination")
    if not depart:
        raise FlightSearchError("Need a departure date")

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _search_sync, origin, destination, depart, return_date)


def format_flight_summary(flight: dict) -> str:
    """One-line human summary: `$348 • ONT→SFO • 1h 29m nonstop`."""
    price = flight.get("price")
    curr = flight.get("currency", "USD")
    fly_from = flight.get("flyFrom", "")
    fly_to = flight.get("flyTo", "")
    secs = int(flight.get("totalDurationInSeconds") or 0)
    hours, mins = divmod(secs // 60, 60)
    duration = f"{hours}h {mins}m" if hours else f"{mins}m"
    layovers = flight.get("layovers") or []
    via = f"{len(layovers)} stop{'s' if len(layovers) != 1 else ''}" if layovers else "nonstop"
    return f"${price} {curr} • {fly_from}→{fly_to} • {duration} • {via}"
