"""
Pinned microphone selection.

Ali prefers the MacBook's built-in microphone so headsets, AirPods, or
virtual inputs (BlackHole, Loopback, etc.) can't silently hijack the
capture path. But we also need to survive demo-day scenarios — different
MacBook models, a venue-supplied USB/lavalier mic, etc. — so the
selection order is:

  1. Explicit override via env var VOICE_AGENT_MIC_INDEX (numeric index)
     or VOICE_AGENT_MIC_NAME (case-insensitive substring match).
  2. Preferred built-in-mic names (Air, Pro, generic built-in).
  3. Fall back to the first available input device, with a warning so
     the user sees which mic was picked.

Every code path that opens a PyAudio input stream should resolve the
device index through :func:`get_pinned_input_device_index`.
"""
from __future__ import annotations

import os
from typing import Optional

import pyaudio  # pyright: ignore[reportMissingModuleSource]

# Preferred device-name substrings, in priority order. PortAudio reports the
# built-in mic as "MacBook Pro Microphone" on MBPs, "MacBook Air Microphone"
# on Airs, and sometimes "Built-in Microphone" on older/Intel machines.
_PREFERRED_NAME_SUBSTRINGS: tuple[str, ...] = (
    "macbook pro microphone",
    "macbook air microphone",
    "macbook pro",
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

    if not candidates:
        raise RuntimeError("no input devices available")

    # 1. Explicit env overrides (for demo-day venues with a lavalier/USB mic).
    override_idx = os.environ.get("VOICE_AGENT_MIC_INDEX", "").strip()
    if override_idx:
        try:
            want = int(override_idx)
            for idx, info in candidates:
                if idx == want:
                    return _pin(idx, info, reason="env-override (index)")
            print(
                f"[voice] VOICE_AGENT_MIC_INDEX={override_idx} not among "
                "input devices — falling back to preferred search."
            )
        except ValueError:
            print(f"[voice] VOICE_AGENT_MIC_INDEX={override_idx!r} not an int — ignoring.")

    override_name = os.environ.get("VOICE_AGENT_MIC_NAME", "").strip().lower()
    if override_name:
        for idx, info in candidates:
            if override_name in str(info.get("name", "")).lower():
                return _pin(idx, info, reason=f"env-override (name~={override_name!r})")
        print(
            f"[voice] VOICE_AGENT_MIC_NAME={override_name!r} did not match "
            "any input — falling back to preferred search."
        )

    # 2. Preferred built-in microphone names (priority order).
    for needle in _PREFERRED_NAME_SUBSTRINGS:
        for idx, info in candidates:
            name = str(info.get("name", "")).lower()
            if needle in name:
                return _pin(idx, info, reason=f"preferred ({needle!r})")

    # 3. Fail-graceful fallback: take the first available input. Print a
    # prominent warning so the user knows the "built-in only" guard is off.
    # Catches demo-day scenarios where the venue provides an unfamiliar mic.
    idx, info = candidates[0]
    available = ", ".join(
        f'#{i} "{info.get("name", "unknown")}"' for i, info in candidates
    )
    print(
        "[voice][WARN] No preferred built-in mic found. Falling back to "
        f'#{idx} "{info.get("name")}". Available: {available}. '
        "Set VOICE_AGENT_MIC_INDEX or VOICE_AGENT_MIC_NAME to pick a specific one."
    )
    return _pin(idx, info, reason="first-available fallback")


def _pin(idx: int, info: dict, *, reason: str) -> int:
    global _cached_index, _cached_name
    _cached_index = idx
    _cached_name = str(info.get("name", ""))
    print(
        f'[voice] Pinned mic: #{idx} "{_cached_name}" '
        f'({int(info.get("defaultSampleRate", 0))} Hz) [{reason}]'
    )
    return idx


def get_pinned_input_device_name() -> Optional[str]:
    """Return the last-resolved pinned device name (for diagnostics)."""
    return _cached_name
