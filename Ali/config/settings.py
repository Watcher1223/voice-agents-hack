"""
Global configuration.
API keys are loaded from .env in the project root (never commit that file --- Alspencer --- I know claude--- don't leak your source code next time).
"""

import os
from pathlib import Path

# Load .env from project root (silently ignored if file doesn't exist)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed — fall back to environment variables only

# ── Gemini API (for fast text intent parsing) ─────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── Deepgram API (real-time streaming STT for meeting capture) ────────────────
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")

# ── Cactus Cloud API ──────────────────────────────────────────────────────────
CACTUS_API_KEY = os.environ.get("CACTUS_API_KEY", "")

# ── Cactus / Gemma 4 ─────────────────────────────────────────────────────────
# General-purpose on-device model used by RAG, file resolution, and the
# visual planner. Powerful but slow — ~30s/call on M2 without dedicated ANE.
CACTUS_GEMMA4_MODEL = os.getenv("CACTUS_GEMMA4_MODEL", "google/gemma-4-E2B-it")

# Intent classification is a tight JSON-shaped task that doesn't need the
# 2B model. functiongemma-270m-it is purpose-built for function-calling /
# structured output and runs ~20x faster (~1.5s/call) on ANE.
CACTUS_INTENT_MODEL = os.getenv("CACTUS_INTENT_MODEL", "google/functiongemma-270m-it")

# Route specific ambient-pipeline decisions through the local Gemma-4 sidecar
# instead of the default (keyword heuristic for mode, always-call-Gemini for
# silence). Both flags are opt-in — if the sidecar is down or a call times out
# the caller transparently falls back to the original behaviour, so turning
# these on can only *improve* the pipeline, never break it.
#
#   VOICE_AGENT_GEMMA_SILENCE=1  — Local pre-gate: if Gemma says "silent" we
#                                  skip the Gemini ambient-analysis call.
#                                  Eval: skips 53% of Gemini calls with 5% FN.
GEMMA_SILENCE_ENABLED = os.environ.get("VOICE_AGENT_GEMMA_SILENCE", "0").lower() in {"1", "true", "yes"}

# System-audio capture via ScreenCaptureKit helper at tools/bin/SysAudio.app.
# When enabled, a second Deepgram stream runs alongside the mic stream with
# audio piped from the Mac's output — this lets Ali transcribe the OTHER
# side of a FaceTime/Zoom/Meet call (voices that only exist in the speaker
# output, not the mic). First run will trigger the macOS Screen Recording
# permission prompt; after granting, relaunch. Falls through silently if
# the helper binary is missing or permission is denied.
SYS_AUDIO_ENABLED = os.environ.get("VOICE_AGENT_SYS_AUDIO", "0").lower() in {"1", "true", "yes"}

# ── Cactus VL (browser sub-agent) ────────────────────────────────────────────
# The browser sub-agent runs inside a Chrome extension whose LLM is configured
# via chrome.storage.local. scripts/cactus_server.py is the HTTP sidecar the
# extension talks to when provider='cactus'; the AI Studio path (default) is
# used when provider='google'.
CACTUS_VL_MODEL    = os.getenv("CACTUS_VL_MODEL",    "google/gemma-4-E2B-it")
CACTUS_SIDECAR_URL = os.getenv("CACTUS_SIDECAR_URL", "http://127.0.0.1:8765")
AGENT_NODE_BIN     = os.getenv("AGENT_NODE_BIN",     "node")

# ── OpenCLI routing (deterministic, pre-built adapters) ──────────────────────
# When ROUTE_OPENCLI_ENABLED is on, voice transcripts that match an entry in
# config/opencli_intents.json are dispatched to the opencli CLI (no LLM in the
# loop). If no intent matches or the flag is off, we fall back to
# ROUTE_BROWSER_TASK_ENABLED — the LLM-driven browser sub-agent. Both flags can
# be toggled at the command line via env to A/B the two paths live.
ROUTE_OPENCLI_ENABLED     = os.environ.get("VOICE_AGENT_ROUTE_OPENCLI", "1").lower() in {"1", "true", "yes"}
ROUTE_BROWSER_TASK_ENABLED = os.environ.get("VOICE_AGENT_ROUTE_BROWSER_TASK", "1").lower() in {"1", "true", "yes"}
# Ambient mode: run Deepgram from boot + glass-style intent detection that
# surfaces suggestions every 5 final transcripts (not 12s wall-clock). On
# by default because the task checklist (loose utterances → tickable rows)
# depends on it; set VOICE_AGENT_AMBIENT=0 to use PTT-only.
AMBIENT_ENABLED = os.environ.get("VOICE_AGENT_AMBIENT", "1").lower() in {"1", "true", "yes"}
# Event-driven screen context: snap when focus changes or screen is stale.
# Passed as an image + app/title to the ambient analyser so tier 1-3 can
# reference what's on screen. On by default when ambient is on.
AMBIENT_SCREEN_ENABLED = os.environ.get(
    "VOICE_AGENT_AMBIENT_SCREEN", "1" if AMBIENT_ENABLED else "0"
).lower() in {"1", "true", "yes"}
# Voice readback (TTS via `say`) is OFF by default — overlay text is the
# primary response channel and voice gets in the way during meetings.
# Set to 1 to re-enable speaking tier 1/2 answers.
AMBIENT_SPEAK_ENABLED = os.environ.get(
    "VOICE_AGENT_AMBIENT_SPEAK", "0"
).lower() in {"1", "true", "yes"}
# How many Deepgram finals before one Gemini ambient analysis. Lower =
# faster checklist updates (more API calls). Default 5.
try:
    AMBIENT_TRIGGER_EVERY_FINALS = max(
        1, int(os.environ.get("VOICE_AGENT_AMBIENT_TRIGGER_FINALS", "5"))
    )
except ValueError:
    AMBIENT_TRIGGER_EVERY_FINALS = 5
# Transient Gemini errors (503, overload) — retry before giving up.
try:
    AMBIENT_ANALYSE_RETRIES = max(
        1, int(os.environ.get("VOICE_AGENT_AMBIENT_ANALYSE_RETRIES", "4"))
    )
except ValueError:
    AMBIENT_ANALYSE_RETRIES = 4
# Idle-flush window — after the last final, if no new final arrives
# within this many seconds and >=1 final is buffered, fire an analysis
# anyway. Prevents a single complete sentence ("email Hanzi about
# Hawaii") from sitting in the buffer forever when the user stops
# talking before the 5-final trigger. Set to 0 to disable.
try:
    AMBIENT_IDLE_FLUSH_S = max(
        0.0, float(os.environ.get("VOICE_AGENT_AMBIENT_IDLE_FLUSH_S", "3.5"))
    )
except ValueError:
    AMBIENT_IDLE_FLUSH_S = 3.5
# Ambient Cactus fallback: when Gemini returns 429 / RESOURCE_EXHAUSTED or
# the retry budget is blown, run the analysis locally via the Cactus
# CLI. OFF by default because the small on-device models we can afford
# to call per utterance (gemma-3-1b-it, functiongemma-270m) aren't
# reliable enough on this schema — they hallucinate names and drop
# slots, which makes bad checklist rows that are worse than silent
# failures. gemma-4-E2B-it is reliable but ~30-60s cold / ~3-5s warm,
# which only makes sense behind a persistent sidecar server
# (scripts/cactus_server.py). Flip VOICE_AGENT_AMBIENT_CACTUS_FALLBACK=1
# to opt in and pick a model via VOICE_AGENT_AMBIENT_CACTUS_MODEL.
AMBIENT_CACTUS_FALLBACK = os.environ.get(
    "VOICE_AGENT_AMBIENT_CACTUS_FALLBACK", "0"
).lower() in {"1", "true", "yes"}
try:
    AMBIENT_CACTUS_TIMEOUT_S = max(
        5.0, float(os.environ.get("VOICE_AGENT_AMBIENT_CACTUS_TIMEOUT_S", "25"))
    )
except ValueError:
    AMBIENT_CACTUS_TIMEOUT_S = 25.0
AMBIENT_CACTUS_MODEL = os.environ.get(
    "VOICE_AGENT_AMBIENT_CACTUS_MODEL", "google/gemma-3-1b-it"
)
# OpenCLI requires Node >= 22.19. Bypass the shebang and invoke node+entry
# directly so a stale `env node` in PATH can't resolve to an older version.
# Defaults are derived from `which node` so this works on any dev's machine
# as long as they've run `nvm use 22` (or have Node 22+ as their default).
# Override with OPENCLI_NODE_BIN / OPENCLI_ENTRY if the setup is non-standard.
import shutil as _shutil

_NODE_BIN_ON_PATH = _shutil.which("node") or ""
_OPENCLI_ENTRY_FROM_NODE = ""
if _NODE_BIN_ON_PATH:
    _candidate = (
        Path(_NODE_BIN_ON_PATH).resolve().parent.parent
        / "lib" / "node_modules" / "@jackwener" / "opencli"
        / "dist" / "src" / "main.js"
    )
    if _candidate.exists():
        _OPENCLI_ENTRY_FROM_NODE = str(_candidate)

OPENCLI_NODE_BIN = os.environ.get("OPENCLI_NODE_BIN", _NODE_BIN_ON_PATH)
OPENCLI_ENTRY = os.environ.get("OPENCLI_ENTRY", _OPENCLI_ENTRY_FROM_NODE)

# ── Whisper fallback ──────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base.en"   # tiny.en | base.en | small.en

# ── Chrome persistent context ─────────────────────────────────────────────────
# macOS default Chrome profile. Change "Default" if you use a different profile.
CHROME_PROFILE_PATH = os.path.expanduser(
    "~/Library/Application Support/Google/Chrome/Default"
)

# ── Push-to-talk hotkey ───────────────────────────────────────────────────────
# Configured in voice/capture.py — keyboard.Key.alt = Option key

# ── Demo safety ────────────────────────────────────────────────────────────────
# When set (1/true/yes), irreversible actions are skipped.
DRY_RUN = os.environ.get("VOICE_AGENT_DRY_RUN", "").lower() in {"1", "true", "yes"}

# ── Vision-first orchestration ─────────────────────────────────────────────────
VISION_FIRST_ENABLED = os.environ.get("VOICE_AGENT_VISION_FIRST", "1").lower() in {"1", "true", "yes"}
VISION_MAX_ACTION_STEPS = int(os.environ.get("VOICE_AGENT_VISION_MAX_ACTION_STEPS", "8"))
VISION_ARTIFACT_DIR = os.path.expanduser(
    os.environ.get("VOICE_AGENT_VISION_ARTIFACT_DIR", "~/tmp/yc_voice_agent_observations")
)

# ── macOS app names ───────────────────────────────────────────────────────────
MESSAGES_APP = "Messages"
MAIL_APP = "Mail"
CALENDAR_APP = "Calendar"
CONTACTS_APP = "Contacts"

# ── File resolver ─────────────────────────────────────────────────────────────
# Local-only resolver that turns natural-language transcripts into concrete
# file paths using Spotlight (mdfind) + Cactus-proposed predicates.

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0")
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _parse_search_roots(raw: str) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            resolved = Path(item).expanduser().resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen:
            continue
        if not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(key)
        roots.append(resolved)
    return roots


FILE_RESOLVER_ENABLED = _env_bool("VOICE_AGENT_FILE_RESOLVER", True)
FILE_RESOLVER_ALIAS_FIRST = _env_bool("VOICE_AGENT_FILE_RESOLVER_ALIAS_FIRST", True)
FILE_RESOLVER_USE_SPOTLIGHT = _env_bool("VOICE_AGENT_USE_SPOTLIGHT", True)

FILE_SEARCH_ROOTS: list[Path] = _parse_search_roots(
    os.environ.get(
        "VOICE_AGENT_FILE_SEARCH_ROOTS",
        "~/Desktop,~/Documents,~/Downloads",
    )
)

FILE_PREDICATE_MAX_ROUNDS = _env_int("VOICE_AGENT_FILE_PREDICATE_MAX_ROUNDS", 2)
FILE_MDFIND_MAX_RESULTS = _env_int("VOICE_AGENT_FILE_MDFIND_MAX_RESULTS", 40)
FILE_INDEX_MAX_CHARS = _env_int("VOICE_AGENT_FILE_INDEX_MAX_CHARS", 2000)
FILE_WALK_MAX_FILES = _env_int("VOICE_AGENT_FILE_WALK_MAX_FILES", 500)
FILE_WALK_MAX_DEPTH = _env_int("VOICE_AGENT_FILE_WALK_MAX_DEPTH", 4)

FILE_RESOLVE_DEBUG = _env_bool("VOICE_AGENT_FILE_RESOLVE_DEBUG", False)

# ── Disk index (laptop-wide content + embeddings) ────────────────────────────
# Local-first retrieval-augmented Q&A over the user's files + macOS data.
# Default is a focused scope: the three docs folders, /Applications, plus
# Contacts / Calendar / Messages. Set ALI_INDEX_FULL_DISK=1 (or pass
# `--full-disk` on the CLI) for a full home-directory scan.

INDEX_DIR: Path = Path(os.path.expanduser(
    os.environ.get("ALI_INDEX_DIR", "~/.cache/ali/index")
))

INDEX_FULL_DISK = _env_bool("ALI_INDEX_FULL_DISK", False)

_DEFAULT_SCAN_ROOTS = "~/Documents,~/Downloads,~/Desktop,/Applications"
_FULL_DISK_ROOTS = "~,/Applications"

INDEX_SCAN_ROOTS: list[Path] = _parse_search_roots(
    os.environ.get(
        "ALI_INDEX_SCAN_ROOTS",
        _FULL_DISK_ROOTS if INDEX_FULL_DISK else _DEFAULT_SCAN_ROOTS,
    )
)

INDEX_MAX_FILE_BYTES = _env_int("ALI_INDEX_MAX_FILE_BYTES", 5_000_000)
INDEX_EMBED_MODEL = os.environ.get(
    "ALI_INDEX_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
INDEX_CHUNK_TOKENS = _env_int("ALI_INDEX_CHUNK_TOKENS", 400)
INDEX_ENABLE_EMBEDDINGS = _env_bool("ALI_INDEX_EMBEDDINGS", True)

# Comma-separated list of non-filesystem data sources to index alongside the
# filesystem walk. Each entry names a module under
# `executors/local/disk_index/sources/`. Set to "" to disable all of them.
INDEX_SOURCES: list[str] = [
    name.strip().lower()
    for name in os.environ.get(
        "ALI_INDEX_SOURCES", "contacts,calendar,messages"
    ).split(",")
    if name.strip()
]

# How far back to go when indexing time-ordered sources (Messages, Calendar).
INDEX_SOURCE_HISTORY_DAYS = _env_int("ALI_INDEX_SOURCE_HISTORY_DAYS", 365)

# Local-first RAG. Default: 100% on-device via Cactus/Gemma 4. Set to 1 to
# allow falling back to Gemini if Cactus is unavailable for answer generation.
ALI_ALLOW_CLOUD_FALLBACK = _env_bool("ALI_ALLOW_CLOUD_FALLBACK", False)
