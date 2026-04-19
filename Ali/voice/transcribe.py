"""
Layer 1 — Speech-to-Text

Primary path: Deepgram Nova-3 (REST `prerecorded` endpoint) with keyterm
biasing drawn from config/vocab.py. Strong on proper nouns and acronyms.

Fallback path: faster-whisper `base.en` on-device — used only when
Deepgram is unreachable or the API returns an error. Keeps Ali working
offline at reduced accuracy.

Tertiary fallback: Cactus CLI (`cactus transcribe`, Parakeet). Kept for
environments without the faster-whisper wheel.

The audio never leaves the device in the Whisper/Cactus branches.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request

try:
    from faster_whisper import WhisperModel  # type: ignore
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

CACTUS_CLI = shutil.which("cactus")
CACTUS_AVAILABLE = CACTUS_CLI is not None

from config.settings import DEEPGRAM_API_KEY, WHISPER_MODEL_SIZE

# Default to nova-2 — works on every Deepgram account. Opt into nova-3 via env
# only if your account has it enabled (otherwise you get a silent HTTP 400 and
# fall back to Whisper).
DEEPGRAM_MODEL = os.environ.get("DEEPGRAM_PTT_MODEL", "nova-2")

_whisper_model = None


def warmup():
    """
    Pre-load the Whisper fallback model at startup so that if Deepgram
    ever errors mid-demo, the failover call is instant. Safe no-op if
    Whisper isn't installed.
    """
    if WHISPER_AVAILABLE:
        print("[stt] Warming up Whisper fallback...")
        _get_whisper()
    if DEEPGRAM_API_KEY:
        print(f"[stt] Primary: Deepgram {DEEPGRAM_MODEL}. Fallback: Whisper {WHISPER_MODEL_SIZE}.")
    else:
        print(f"[stt] Primary: Whisper {WHISPER_MODEL_SIZE} (no DEEPGRAM_API_KEY).")


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


async def transcribe(audio_bytes: bytes) -> str:
    """
    Transcribe WAV audio bytes → text. Tries Deepgram first, falls back
    to Whisper on any error. Applies vocab-based post-correction as a
    final safety net on whatever branch succeeds.
    """
    from config.vocab import apply_corrections

    loop = asyncio.get_event_loop()

    if DEEPGRAM_API_KEY:
        try:
            result = await loop.run_in_executor(
                None, _transcribe_deepgram, audio_bytes
            )
            if result is not None:
                corrected = apply_corrections(result)
                print(f"[stt] backend=deepgram:{DEEPGRAM_MODEL}  → {corrected!r}")
                return corrected
        except Exception as e:
            print(f"[stt] Deepgram failed ({e}); falling back to Whisper")

    if WHISPER_AVAILABLE:
        result = await loop.run_in_executor(None, _transcribe_whisper, audio_bytes)
        print(f"[stt] backend=whisper:{WHISPER_MODEL_SIZE}  → {result!r}")
        return result

    if CACTUS_AVAILABLE:
        result = await _transcribe_cactus_cli(audio_bytes)
        print(f"[stt] backend=cactus  → {result!r}")
        return result

    raise RuntimeError(
        "No STT backend available. Set DEEPGRAM_API_KEY or run: pip install faster-whisper"
    )


def _transcribe_deepgram(audio_bytes: bytes) -> str | None:
    """
    POST audio to Deepgram's prerecorded `/v1/listen` endpoint with keyterm
    biasing. Returns the transcript, or None if the request failed (caller
    falls back to Whisper).

    Uses urllib to avoid a hard dep on `requests` — the body is raw audio,
    params go in the query string. Keyterm params are repeated per term.
    """
    from config.vocab import keyterms

    terms = keyterms()
    params: list[tuple[str, str]] = [
        ("model", DEEPGRAM_MODEL),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("language", "en"),
    ]
    # Bias proper nouns. Nova-3 takes `keyterm=`; Nova-2 and older take
    # `keywords=term:intensifier`. We send only the one matching the
    # active model — mixing them causes 400s on some accounts.
    is_nova3 = DEEPGRAM_MODEL.startswith("nova-3")
    for term in terms:
        if is_nova3:
            params.append(("keyterm", term))
        else:
            params.append(("keywords", f"{term}:3.0"))

    qs = urllib.parse.urlencode(params)
    url = f"https://api.deepgram.com/v1/listen?{qs}"
    req = urllib.request.Request(
        url,
        data=audio_bytes,
        headers={
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Print the response body so the operator can see Deepgram's
        # actual error (unsupported model, bad audio, param conflict…).
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:400]
        except Exception:
            pass
        print(f"[stt] Deepgram HTTP {e.code}: {body or e.reason}")
        return None
    except Exception as e:
        print(f"[stt] Deepgram REST error: {e}")
        return None

    try:
        txt = payload["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (KeyError, IndexError, TypeError):
        print(f"[stt] Deepgram unexpected response shape: {list(payload.keys())}")
        return None

    return (txt or "").strip()


def _transcribe_whisper(audio_bytes: bytes) -> str:
    from config.vocab import whisper_initial_prompt, apply_corrections

    model = _get_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        segments, _ = model.transcribe(
            tmp_path,
            beam_size=5,
            initial_prompt=whisper_initial_prompt(),
        )
        raw = " ".join(seg.text for seg in segments).strip()
        return apply_corrections(raw)
    finally:
        os.unlink(tmp_path)


async def _transcribe_cactus_cli(audio_bytes: bytes) -> str:
    from config.vocab import apply_corrections

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        proc = await asyncio.create_subprocess_exec(
            CACTUS_CLI, "transcribe", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode().strip())
        return apply_corrections(stdout.decode().strip())
    finally:
        os.unlink(tmp_path)
