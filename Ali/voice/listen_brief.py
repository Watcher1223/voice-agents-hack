"""
Short one-shot listener used by the end-of-meeting confirmation dialog.

Streams mic audio via Deepgram for up to `timeout` seconds, returns the
first committed utterance (or "" on timeout). Designed for yes/no
answers — not a general-purpose transcription API.
"""
from __future__ import annotations

import asyncio
import threading


async def listen_brief(timeout: float = 6.0) -> str:
    """
    Open the mic, listen for up to `timeout` seconds, return the first
    committed utterance (lower-cased, trimmed). Returns "" if nothing was
    said or the stream errored.
    """
    loop = asyncio.get_event_loop()
    fut: asyncio.Future[str] = loop.create_future()
    stop_event = threading.Event()
    got_final = threading.Event()
    result_holder: dict[str, str] = {"text": ""}

    def _on_interim(_text: str) -> None:
        pass

    def _on_final(text: str) -> None:
        if got_final.is_set():
            return
        got_final.set()
        result_holder["text"] = text.strip()
        stop_event.set()
        if not fut.done():
            loop.call_soon_threadsafe(fut.set_result, result_holder["text"])

    from voice.deepgram_stream import stream_transcription_sync

    def _runner() -> None:
        try:
            stream_transcription_sync(stop_event, _on_interim, _on_final)
        except Exception as e:
            print(f"[listen_brief] stream error: {e}")
            if not fut.done():
                loop.call_soon_threadsafe(fut.set_result, "")

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    try:
        # Cap the wait at `timeout` even if no final arrives.
        return await asyncio.wait_for(fut, timeout=timeout)
    except asyncio.TimeoutError:
        return ""
    finally:
        stop_event.set()
        # Give the Deepgram stream a moment to close cleanly.
        thread.join(timeout=1.5)
