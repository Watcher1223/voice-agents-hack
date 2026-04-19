"""
TTS helper — OpenAI TTS (high quality) with macOS `say` fallback.

Callable from any thread. Returns immediately; speech plays in the background.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading

DEFAULT_RATE = "160"  # kept for macOS say fallback
tts_active = threading.Event()

# OpenAI TTS settings
_OAI_MODEL = "tts-1"   # swap to "tts-1-hd" for higher quality at cost of latency
_OAI_VOICE = "nova"    # alloy | echo | fable | onyx | nova | shimmer

_openai_client = None
_openai_lock = threading.Lock()


def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    with _openai_lock:
        if _openai_client is not None:
            return _openai_client
        try:
            from openai import OpenAI  # type: ignore
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return None
            _openai_client = OpenAI(api_key=api_key)
            return _openai_client
        except ImportError:
            return None


_FALLBACK_VOICES = ["Ava (Enhanced)", "Nicky (Enhanced)", "Zoe (Enhanced)", "Samantha"]
_VOICE_CACHE: str | None = None


def _voice() -> str:
    """Return best available macOS voice (used for fallback and overlay)."""
    global _VOICE_CACHE
    if _VOICE_CACHE is not None:
        return _VOICE_CACHE
    try:
        result = subprocess.run(
            ["/usr/bin/say", "-v", "?"],
            capture_output=True, text=True, timeout=3,
        )
        installed = result.stdout.lower()
        for v in _FALLBACK_VOICES:
            if v.lower().split(" (")[0] in installed:
                _VOICE_CACHE = v
                return v
    except Exception:
        pass
    _VOICE_CACHE = "Samantha"
    return _VOICE_CACHE


def track_tts_process(proc: subprocess.Popen[bytes]) -> None:
    """Mark TTS as active until this process exits."""
    tts_active.set()

    def _wait() -> None:
        try:
            proc.wait(timeout=12)
        except Exception:
            pass
        finally:
            tts_active.clear()

    threading.Thread(target=_wait, daemon=True).start()


def _speak_openai(text: str) -> bool:
    """Speak via OpenAI TTS API. Returns True if successfully dispatched."""
    client = _get_openai_client()
    if not client:
        return False

    tts_active.set()

    def _run() -> None:
        tmp_path: str | None = None
        try:
            response = client.audio.speech.create(
                model=_OAI_MODEL,
                voice=_OAI_VOICE,
                input=text,
                response_format="mp3",
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(response.content)
                tmp_path = f.name
            subprocess.run(
                ["afplay", tmp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            tts_active.clear()

    threading.Thread(target=_run, daemon=True).start()
    return True


async def wait_for_tts() -> None:
    """Async-friendly wait until the current TTS playback finishes."""
    import asyncio
    await asyncio.sleep(0.15)   # brief pause to let TTS start
    while tts_active.is_set():
        await asyncio.sleep(0.1)


def speak(text: str, voice: str | None = None, rate: str = DEFAULT_RATE) -> None:
    """Speak `text` asynchronously. Uses OpenAI TTS; falls back to macOS say."""
    if not text or not text.strip():
        return
    if sys.platform != "darwin":
        return

    # Use OpenAI TTS unless a specific local voice is explicitly requested
    if voice is None and _speak_openai(text):
        return

    # Fallback: macOS say
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
