"""System-audio streaming transcription.

Parallel to `deepgram_stream.stream_transcription_sync`, but the audio
source is ScreenCaptureKit (via the bundled Swift helper at
`tools/bin/SysAudio.app`) instead of the microphone. This is what lets
Ali transcribe FaceTime / Zoom / Meet participants whose voices only
exist in the Mac's audio *output*, not its mic input.

The Swift helper writes raw 16-bit 16kHz mono PCM to stdout. We pipe
those bytes into a second Deepgram WebSocket. Final transcripts are
forwarded to the same `on_final` callback as the mic stream, but
prefixed with `[Remote]` so the ambient prompt (and the overlay) can
attribute them to the other side of the call.

First run will trigger macOS's Screen Recording permission prompt. Once
granted the helper runs silently.
"""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Callable

from config.settings import DEEPGRAM_API_KEY


SAMPLE_RATE = 16000
CHUNK_BYTES = 4096     # ~128 ms at 16-bit 16 kHz mono

_REPO = Path(__file__).resolve().parent.parent
_SYSAUDIO_APP_BIN = _REPO / "tools" / "bin" / "SysAudio.app" / "Contents" / "MacOS" / "sysaudio"
_SYSAUDIO_BARE_BIN = _REPO / "tools" / "bin" / "sysaudio"


def _pick_binary() -> Path | None:
    """Prefer the bundled .app path (so macOS TCC associates permission
    with the bundle identifier `com.ali.sysaudio`). Fall back to the
    bare binary if the bundle hasn't been built yet."""
    if _SYSAUDIO_APP_BIN.exists() and os.access(_SYSAUDIO_APP_BIN, os.X_OK):
        return _SYSAUDIO_APP_BIN
    if _SYSAUDIO_BARE_BIN.exists() and os.access(_SYSAUDIO_BARE_BIN, os.X_OK):
        return _SYSAUDIO_BARE_BIN
    return None


def stream_sysaudio_transcription_sync(
    stop_event: threading.Event,
    on_interim: Callable[[str], None],
    on_final: Callable[[str], None],
) -> None:
    """Blocking — runs in a background thread. Spawns the Swift
    SysAudio helper, pipes its stdout into a Deepgram WebSocket,
    invokes `on_final` for each committed transcript (prefixed
    `[Remote] ...`). Returns cleanly when `stop_event` is set or the
    helper exits (the caller can decide whether to restart)."""
    binary = _pick_binary()
    if binary is None:
        print("[sysaudio] no SysAudio binary found; run: cd Ali && "
              "swiftc -O -framework ScreenCaptureKit -framework AVFoundation "
              "-framework CoreMedia -framework CoreGraphics "
              "-o tools/bin/sysaudio tools/sysaudio.swift")
        return

    try:
        from deepgram import DeepgramClient  # type: ignore[reportMissingImports]
        from deepgram.listen.v1.socket_client import (  # type: ignore[reportMissingImports]
            ListenV1Results,
        )
        from deepgram.core.events import EventType  # type: ignore[reportMissingImports]
    except ImportError:
        print("[sysaudio] deepgram-sdk not installed — skipping system-audio stream")
        return

    dg = DeepgramClient(api_key=DEEPGRAM_API_KEY)

    proc = subprocess.Popen(
        [str(binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,  # unbuffered — we want bytes as fast as SCStream delivers them
    )

    # Helper: drain stderr in a thread so Swift log lines appear in our
    # console instead of blocking the subprocess pipe. Any `sysaudio:`
    # message (including the TCC permission nudge) shows up here.
    def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode(errors="ignore").rstrip()
            if line:
                print(f"[sysaudio] {line}")

    threading.Thread(target=_drain_stderr, daemon=True).start()

    print("[sysaudio] Streaming started (system audio → Deepgram)")
    try:
        with dg.listen.v1.connect(
            model="nova-2",
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
        ) as connection:

            from collections import Counter
            from config.vocab import apply_corrections

            def _dominant_speaker(words) -> int | None:
                try:
                    counts = Counter(
                        getattr(w, "speaker", None)
                        for w in (words or [])
                        if getattr(w, "speaker", None) is not None
                    )
                    if counts:
                        return counts.most_common(1)[0][0]
                except Exception:
                    pass
                return None

            _msg_count = [0]
            def _on_message(msg) -> None:
                _msg_count[0] += 1
                if _msg_count[0] in (1, 10, 100):
                    print(f"[sysaudio] Deepgram message #{_msg_count[0]} type={type(msg).__name__}")
                if not isinstance(msg, ListenV1Results):
                    return
                try:
                    alt = msg.channel.alternatives[0]
                    text = (alt.transcript or "").strip()
                    if not text:
                        return
                    # System audio contains the remote side of the call
                    # (plus any other app audio). Tag with [Remote] so the
                    # ambient prompt can distinguish commitments made by
                    # the other party from the user's own turns.
                    spk = _dominant_speaker(getattr(alt, "words", None))
                    label = f"[Remote-S{spk}]" if spk is not None else "[Remote]"
                    text = f"{label} {text}"
                    if msg.is_final:
                        on_final(apply_corrections(text))
                    else:
                        on_interim(text)
                except Exception:
                    pass

            connection.on(EventType.MESSAGE, _on_message)
            listener = threading.Thread(target=connection.start_listening, daemon=True)
            listener.start()

            # Pump PCM bytes from the Swift helper → Deepgram.
            assert proc.stdout is not None
            total_bytes = 0
            last_log = 0
            while not stop_event.is_set():
                chunk = proc.stdout.read(CHUNK_BYTES)
                if not chunk:
                    print("[sysaudio] helper exited (stdout EOF)")
                    break
                connection.send_media(chunk)
                total_bytes += len(chunk)
                # Log byte rate every ~1 MB so we can tell whether the
                # helper is producing audio and whether Deepgram is
                # receiving it. 16kHz mono PCM16 = 32 kB/s → 1 MB ≈ 32s.
                if total_bytes - last_log >= 1_000_000:
                    print(f"[sysaudio] sent {total_bytes // 1024} KB to Deepgram")
                    last_log = total_bytes

            try:
                connection.send_close_stream()
            except Exception:
                pass
            listener.join(timeout=2.0)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("[sysaudio] Streaming stopped")
