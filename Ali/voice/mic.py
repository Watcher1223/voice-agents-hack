"""
Pinned microphone selection.

Ali is hardwired to record from the MacBook Air's built-in microphone so
headsets, AirPods, or virtual inputs (BlackHole, Loopback, etc.) can never
hijack the capture path. Every code path that opens a PyAudio input stream
should resolve the device index through :func:`get_pinned_input_device_index`.
"""
from __future__ import annotations

from typing import Optional

import pyaudio  # pyright: ignore[reportMissingModuleSource]

# Preferred device-name substrings, in priority order. PortAudio reports the
# built-in mic on Apple Silicon MacBook Airs as "MacBook Air Microphone";
# older/Intel machines sometimes report it as "Built-in Microphone".
_PREFERRED_NAME_SUBSTRINGS: tuple[str, ...] = (
    "macbook air microphone",
    "macbook air",
    "built-in microphone",
)

_cached_index: Optional[int] = None
_cached_name: Optional[str] = None


def get_pinned_input_device_index(audio: pyaudio.PyAudio) -> int:
    """
    Return the PyAudio device index for the MacBook Air microphone.

    Raises RuntimeError if no matching input device is present — we'd rather
    fail loud than silently record from whatever the OS has set as default
    (e.g. AirPods the user forgot were connected).
    """
    global _cached_index, _cached_name

    if _cached_index is not None:
        try:
            info = audio.get_device_info_by_index(_cached_index)
            if int(info.get("maxInputChannels", 0)) > 0:
                return _cached_index
        except Exception:
            pass
        _cached_index = None
        _cached_name = None

    candidates: list[tuple[int, dict]] = []
    for idx in range(audio.get_device_count()):
        try:
            info = audio.get_device_info_by_index(idx)
        except Exception:
            continue
        if int(info.get("maxInputChannels", 0)) <= 0:
            continue
        candidates.append((idx, info))

    for needle in _PREFERRED_NAME_SUBSTRINGS:
        for idx, info in candidates:
            name = str(info.get("name", "")).lower()
            if needle in name:
                _cached_index = idx
                _cached_name = str(info.get("name", ""))
                print(
                    f'[voice] Pinned mic: #{idx} "{_cached_name}" '
                    f'({int(info.get("defaultSampleRate", 0))} Hz)'
                )
                return idx

    available = ", ".join(
        f'#{i} "{info.get("name", "unknown")}"' for i, info in candidates
    ) or "(none)"
    raise RuntimeError(
        "MacBook Air microphone not found — Ali is hardwired to the built-in "
        f"mic. Available input devices: {available}"
    )


def get_pinned_input_device_name() -> Optional[str]:
    """Return the last-resolved pinned device name (for diagnostics)."""
    return _cached_name
