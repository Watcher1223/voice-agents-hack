"""
Calendar event creation via macOS Calendar (AppleScript).
Creates flight events in the user's first writable calendar.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime


def _applescript_for_event(
    title: str,
    iso_date: str,
    start_hour: int = 9,
    duration_hours: int = 2,
) -> str:
    dt = datetime.strptime(iso_date, "%Y-%m-%d")
    end_hour = start_hour + duration_hours

    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
    return f"""tell application "Calendar"
    set targetCal to first calendar whose writable is true
    tell targetCal
        set startDate to current date
        set year of startDate to {dt.year}
        set month of startDate to {dt.month}
        set day of startDate to {dt.day}
        set hours of startDate to {start_hour}
        set minutes of startDate to 0
        set seconds of startDate to 0
        set endDate to current date
        set year of endDate to {dt.year}
        set month of endDate to {dt.month}
        set day of endDate to {dt.day}
        set hours of endDate to {end_hour}
        set minutes of endDate to 0
        set seconds of endDate to 0
        make new event with properties {{summary:"{safe_title}", start date:startDate, end date:endDate}}
    end tell
end tell"""


def _event_title(origin: str, destination: str, price, iata_from: str, iata_to: str) -> str:
    """Build a human-readable event title using city names when available."""
    # Prefer full city names from slots; fall back to IATA codes from flight data
    src = origin.strip() or iata_from
    dst = destination.strip() or iata_to
    price_str = f" (${price})" if price else ""
    return f"Flight {src} \u2192 {dst}{price_str}"


def add_flight_events(slots: dict, flight: dict) -> int:
    """
    Add calendar event(s) for flight dates using macOS Calendar.

    Creates a departure event, and a return event if return_date is present.
    Returns the number of events successfully created.
    """
    if sys.platform != "darwin":
        return 0

    origin = slots.get("origin") or ""
    destination = slots.get("destination") or ""
    iata_from = flight.get("flyFrom") or origin
    iata_to = flight.get("flyTo") or destination
    price = flight.get("price")
    depart_date = slots.get("depart_date")
    return_date = slots.get("return_date") or None

    if not depart_date:
        return 0

    events = [
        (depart_date, _event_title(origin, destination, price, iata_from, iata_to)),
    ]
    if return_date:
        events.append((return_date, _event_title(destination, origin, None, iata_to, iata_from)))

    created = 0
    for iso_date, title in events:
        script = _applescript_for_event(title, iso_date)
        try:
            result = subprocess.run(
                ["osascript"],
                input=script,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                created += 1
        except Exception:
            pass

    return created
