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

    # Build date objects via property assignment — locale-independent.
    def date_block(var: str, hour: int) -> str:
        return f"""
        set {var} to current date
        set year of {var} to {dt.year}
        set month of {var} to {dt.month}
        set day of {var} to {dt.day}
        set hours of {var} to {hour}
        set minutes of {var} to 0
        set seconds of {var} to 0"""

    safe_title = title.replace('"', "'")
    return f"""
tell application "Calendar"
    set targetCal to first calendar whose writable is true
    tell targetCal
        {date_block("startDate", start_hour)}
        {date_block("endDate", end_hour)}
        make new event with properties {{summary:"{safe_title}", start date:startDate, end date:endDate}}
    end tell
end tell
"""


def add_flight_events(slots: dict, flight: dict) -> int:
    """
    Add calendar event(s) for flight dates using macOS Calendar.

    Creates a departure event, and a return event if return_date is present.
    Returns the number of events successfully created.
    """
    if sys.platform != "darwin":
        return 0

    fly_from = flight.get("flyFrom") or slots.get("origin", "")
    fly_to = flight.get("flyTo") or slots.get("destination", "")
    price = flight.get("price")
    depart_date = slots.get("depart_date")
    return_date = slots.get("return_date") or None

    if not depart_date:
        return 0

    price_str = f" (${price})" if price else ""
    created = 0

    for iso_date, title in [
        (depart_date, f"Flight {fly_from} to {fly_to}{price_str}"),
        *([(return_date, f"Return flight {fly_to} to {fly_from}")] if return_date else []),
    ]:
        script = _applescript_for_event(title, iso_date)
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                created += 1
        except Exception:
            pass

    return created
