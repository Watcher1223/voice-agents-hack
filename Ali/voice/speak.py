"""
TTS helper — non-blocking.

Priority:
  1. OpenAI `tts-1-hd` with a natural voice (nova by default) — used if
     OPENAI_API_KEY is set. The wake-camera greeting already uses this
     path; now every speak() call does too.
  2. macOS `say` fallback — uses Ava Enhanced if installed, else Samantha.
     Kept so Ali still talks on machines without an OpenAI key.

Callable from any thread. Returns immediately; speech plays in the
background. No change to the call site: `speak("text")`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading

# ── OpenAI TTS (primary) ──────────────────────────────────────────────────────

# nova — warm neutral female. Other options: alloy, echo, fable, onyx, shimmer.
OPENAI_TTS_VOICE = os.environ.get("ALI_TTS_VOICE", "nova")
OPENAI_TTS_MODEL = os.environ.get("ALI_TTS_MODEL", "tts-1-hd")
OPENAI_TTS_SPEED = float(os.environ.get("ALI_TTS_SPEED", "1.0"))

# ── macOS `say` fallback ──────────────────────────────────────────────────────

# "Ava (Enhanced)" is a neural voice; fallback chain ends at Samantha which
# is guaranteed present on macOS.
_PREFERRED_VOICES = ["Ava (Enhanced)", "Nicky (Enhanced)", "Zoe (Enhanced)", "Samantha"]
DEFAULT_VOICE = _PREFERRED_VOICES[0]
DEFAULT_RATE = "160"
tts_active = threading.Event()


def _best_available_voice() -> str:
    """Return the first installed enhanced voice, or Samantha."""
    try:
        result = subprocess.run(
            ["/usr/bin/say", "-v", "?"],
            capture_output=True, text=True, timeout=3,
        )
        installed = result.stdout.lower()
        for v in _PREFERRED_VOICES:
            if v.lower().split(" (")[0] in installed:
                return v
    except Exception:
        pass
    return "Samantha"


_VOICE_CACHE: str | None = None


def _voice() -> str:
    global _VOICE_CACHE
    if _VOICE_CACHE is None:
        _VOICE_CACHE = _best_available_voice()
    return _VOICE_CACHE


def track_tts_process(proc: subprocess.Popen[bytes]) -> None:
    """Mark TTS as active until this process exits."""
    tts_active.set()

    def _wait() -> None:
        try:
            proc.wait(timeout=60)
        except Exception:
            pass
        finally:
            tts_active.clear()

    threading.Thread(target=_wait, daemon=True).start()


# ── Backend detection + caching ──────────────────────────────────────────────

_OPENAI_CLIENT = None
_OPENAI_AVAILABLE: bool | None = None


def _openai_client():
    """Return a cached OpenAI client or None if unavailable."""
    global _OPENAI_CLIENT, _OPENAI_AVAILABLE
    if _OPENAI_AVAILABLE is False:
        return None
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        _OPENAI_AVAILABLE = False
        return None
    try:
        from openai import OpenAI  # type: ignore[reportMissingImports]
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
        _OPENAI_AVAILABLE = True
        print(f"[tts] backend=openai model={OPENAI_TTS_MODEL} voice={OPENAI_TTS_VOICE}")
        return _OPENAI_CLIENT
    except Exception as e:
        print(f"[tts] OpenAI unavailable ({e}) — falling back to macOS say")
        _OPENAI_AVAILABLE = False
        return None


def _synthesize_openai(text: str) -> str | None:
    """Generate speech via OpenAI TTS, return path to mp3 or None on error."""
    client = _openai_client()
    if client is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        with client.audio.speech.with_streaming_response.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            speed=OPENAI_TTS_SPEED,
        ) as resp:
            resp.stream_to_file(path)
        return path
    except Exception as e:
        print(f"[tts] OpenAI synth failed ({e}) — falling back to macOS say")
        return None


def _play_macos_say(text: str, voice: str | None, rate: str) -> None:
    """Fire-and-forget macOS `say`."""
    chosen_voice = voice or _voice()
    try:
        proc = subprocess.Popen(
            ["/usr/bin/say", "-v", chosen_voice, "-r", rate, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        track_tts_process(proc)
    except Exception:
        pass


def _play_mp3(path: str) -> None:
    """Play an mp3 via afplay; non-blocking, cleans up the file when done."""
    try:
        proc = subprocess.Popen(
            ["/usr/bin/afplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        track_tts_process(proc)

        def _cleanup() -> None:
            try:
                proc.wait(timeout=60)
            except Exception:
                pass
            try:
                os.unlink(path)
            except Exception:
                pass

        threading.Thread(target=_cleanup, daemon=True).start()
    except Exception:
        pass


def speak(text: str, voice: str | None = None, rate: str = DEFAULT_RATE) -> None:
    """Speak `text` asynchronously. Safe no-op on non-macOS.

    If OPENAI_API_KEY is set, synthesis runs on a background thread so the
    caller never blocks on the network round-trip; playback starts as soon
    as the mp3 is saved.
    """
    if not text or not text.strip():
        return
    if sys.platform != "darwin":
        return

    # Caller overrides the voice → stick with macOS `say` (OpenAI TTS uses
    # its own voice selection). Also used as the fallback path.
    if voice is not None:
        _play_macos_say(text, voice, rate)
        return

    def _worker() -> None:
        mp3 = _synthesize_openai(text)
        if mp3 is not None:
            _play_mp3(mp3)
        else:
            _play_macos_say(text, None, rate)

    threading.Thread(target=_worker, daemon=True).start()
