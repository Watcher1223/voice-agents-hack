"""Gemma-4 (Cactus) silence-gate evaluation harness.

One task: does Gemma correctly decide whether a short transcript window
contains anything worth surfacing to the user? We compare it against the
"always surface" baseline (what the pipeline does when the gate is off).

Each call is repeated --reps times so we can report *reliability*, i.e.
the fraction of calls that returned a parseable, in-schema answer. For a
production gate that value matters more than raw accuracy — a 95%
accurate classifier that occasionally emits prose is useless in the hot
path.

Usage:
    python scripts/gemma_eval.py --dataset datasets/gemma_silence.jsonl --reps 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


SIDECAR_URL = "http://127.0.0.1:8765/v1/complete"
VALID_SILENCE = {"surface", "silent"}


# ── Cactus client ─────────────────────────────────────────────────────────────

def cactus_complete(system: str, user: str, max_tokens: int = 64, timeout_s: float = 30) -> tuple[str, float, bool]:
    """POST to the sidecar. Returns (response_text, latency_ms, ok)."""
    body = json.dumps({
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(SIDECAR_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return (f"__error__:{e}", (time.time() - t0) * 1000, False)
    latency_ms = (time.time() - t0) * 1000
    return (data.get("response", "").strip(), latency_ms, bool(data.get("success")))


# ── Output parsing helpers ────────────────────────────────────────────────────

def parse_label(raw: str, valid: set[str]) -> str | None:
    """Extract the first recognised label from a Gemma response. Accepts JSON,
    bare words, markdown, quoted strings — returns None if nothing valid is
    found so the reliability metric reflects actual parse success."""
    if not raw:
        return None
    # Strip code fences and whitespace.
    cleaned = raw.strip().strip("`").strip()
    # Try JSON first.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            for k in ("label", "mode", "answer", "result", "decision"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip().lower() in valid:
                    return v.strip().lower()
        if isinstance(obj, str) and obj.strip().lower() in valid:
            return obj.strip().lower()
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to word scan.
    low = cleaned.lower()
    for v in valid:
        if v in low:
            return v
    return None


# ── Prompt templates ──────────────────────────────────────────────────────────

_SILENCE_SYSTEM = """You are a "should I interrupt?" gate for an ambient AI assistant.
The user is in a conversation and you see a short rolling transcript.
Decide whether there's anything worth surfacing to them right now.

Output EXACTLY one word:
  surface  - there is a factual question to answer, a jargon term to define, or a concrete action to take (send email, add calendar event, lookup)
  silent   - small talk, filler, greetings, repetition, or nothing actionable

Output one of: surface, silent"""


# ── Baselines ─────────────────────────────────────────────────────────────────

def baseline_silence(window: list[str]) -> str:
    # Current pipeline: always calls Gemini. So the "always surface" baseline is
    # what we compare Gemma against — the point is to show Gemma can save calls.
    return "surface"


# ── Runner ────────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    id: str
    label: str
    base_pred: str
    gemma_preds: list[str] = field(default_factory=list)   # one per rep
    gemma_latencies_ms: list[float] = field(default_factory=list)
    gemma_valid: list[bool] = field(default_factory=list)   # did parse_label succeed?


def run_case(case: dict, reps: int) -> CaseResult:
    window = case["window"]
    label = case["label"]
    user = "\n".join(f"- {line}" for line in window)

    base = baseline_silence(window)
    system = _SILENCE_SYSTEM
    valid = VALID_SILENCE
    max_tokens = 8

    res = CaseResult(id=case["id"], label=label, base_pred=base)
    for _ in range(reps):
        raw, latency_ms, ok = cactus_complete(system, user, max_tokens=max_tokens)
        parsed = parse_label(raw, valid) if ok else None
        res.gemma_preds.append(parsed or "__invalid__")
        res.gemma_latencies_ms.append(latency_ms)
        res.gemma_valid.append(parsed is not None)
    return res


# ── Metrics ───────────────────────────────────────────────────────────────────

def summarise(results: list[CaseResult]) -> dict[str, Any]:
    n = len(results)
    base_correct = sum(1 for r in results if r.base_pred == r.label)

    # Gemma: per-rep accuracy + majority-vote accuracy.
    total_reps = sum(len(r.gemma_preds) for r in results)
    per_rep_correct = sum(
        1
        for r in results
        for p in r.gemma_preds
        if p == r.label
    )
    maj_correct = 0
    for r in results:
        # Majority vote across reps, ignoring invalid reps.
        valid_preds = [p for p in r.gemma_preds if p != "__invalid__"]
        if not valid_preds:
            continue
        vote = max(set(valid_preds), key=valid_preds.count)
        if vote == r.label:
            maj_correct += 1

    all_latencies = [l for r in results for l in r.gemma_latencies_ms]
    p50 = statistics.median(all_latencies) if all_latencies else 0.0
    p95 = statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) >= 20 else (max(all_latencies) if all_latencies else 0.0)
    mean = statistics.mean(all_latencies) if all_latencies else 0.0
    valid_rate = sum(sum(r.gemma_valid) for r in results) / total_reps if total_reps else 0.0

    # Per-label breakdown
    labels = sorted(set(r.label for r in results))
    per_label: dict[str, dict[str, float]] = {}
    for lab in labels:
        subset = [r for r in results if r.label == lab]
        base_c = sum(1 for r in subset if r.base_pred == r.label)
        # Majority-vote accuracy per label
        maj_c = 0
        for r in subset:
            valid_preds = [p for p in r.gemma_preds if p != "__invalid__"]
            if not valid_preds:
                continue
            vote = max(set(valid_preds), key=valid_preds.count)
            if vote == r.label:
                maj_c += 1
        per_label[lab] = {
            "n": float(len(subset)),
            "acc_base": base_c / len(subset),
            "acc_gemma_maj": maj_c / len(subset),
        }

    return {
        "n_cases": n,
        "n_reps_total": total_reps,
        "acc_base": base_correct / n if n else 0.0,
        "acc_gemma_per_rep": per_rep_correct / total_reps if total_reps else 0.0,
        "acc_gemma_majority": maj_correct / n if n else 0.0,
        "valid_rate": valid_rate,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_mean_ms": mean,
        "per_label": per_label,
    }


def print_report(results: list[CaseResult], summary: dict[str, Any]) -> None:
    print(f"\n══════════ GEMMA SILENCE-GATE EVAL ══════════")
    print(f"  cases={summary['n_cases']}  total_reps={summary['n_reps_total']}")
    print(f"  accuracy:")
    print(f"    baseline (current system):      {summary['acc_base']*100:6.2f}%")
    print(f"    Gemma per-rep:                  {summary['acc_gemma_per_rep']*100:6.2f}%")
    print(f"    Gemma majority-vote over reps:  {summary['acc_gemma_majority']*100:6.2f}%")
    print(f"  reliability (valid parseable output): {summary['valid_rate']*100:6.2f}%")
    print(f"  latency: p50={summary['latency_p50_ms']:.0f}ms  p95={summary['latency_p95_ms']:.0f}ms  mean={summary['latency_mean_ms']:.0f}ms")
    print(f"  per-label:")
    for lab, stats in summary["per_label"].items():
        print(f"    {lab:12s} n={int(stats['n']):2d}  base={stats['acc_base']*100:6.2f}%  gemma_maj={stats['acc_gemma_maj']*100:6.2f}%")

    # Mismatches (first 10)
    mism = [r for r in results
            if max(set(p for p in r.gemma_preds if p != '__invalid__') or {'__none__'},
                   key=(lambda p, r=r: [q for q in r.gemma_preds if q != '__invalid__'].count(p))) != r.label]
    if mism:
        print(f"\n  mismatches (first 10 of {len(mism)}):")
        for r in mism[:10]:
            valid_preds = [p for p in r.gemma_preds if p != "__invalid__"]
            vote = max(set(valid_preds), key=valid_preds.count) if valid_preds else "__all_invalid__"
            print(f"    {r.id:22s} label={r.label:10s} base={r.base_pred:10s} gemma={vote:10s}  ({','.join(r.gemma_preds)})")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="datasets/gemma_silence.jsonl")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="only run first N cases (debug)")
    p.add_argument("--out", default="", help="write per-case JSONL to this path")
    args = p.parse_args()

    # Sidecar reachable?
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/healthz", timeout=3) as resp:
            hz = json.loads(resp.read())
        print(f"[sidecar] healthz={hz}")
    except Exception as e:
        print(f"[sidecar] UNREACHABLE: {e}")
        print("Start first: PYTHONPATH=/opt/homebrew/Cellar/cactus/1.14_1/libexec/python/src "
              ".venv/bin/python scripts/cactus_server.py")
        sys.exit(1)

    cases = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    if args.limit:
        cases = cases[: args.limit]
    print(f"[eval] loaded {len(cases)} cases from {args.dataset}")
    print(f"[eval] reps per case: {args.reps}  total calls: {len(cases) * args.reps}")

    results: list[CaseResult] = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        r = run_case(case, args.reps)
        results.append(r)
        if i % 5 == 0 or i == len(cases):
            print(f"  [{i:3d}/{len(cases):3d}] id={case['id']:22s} label={case['label']:10s} "
                  f"base={r.base_pred:10s} gemma={','.join(r.gemma_preds)}")

    summary = summarise(results)
    wall = time.time() - t0
    print(f"\n[eval] wall time {wall:.1f}s  ({wall/len(cases):.2f}s/case × {args.reps} reps)")
    print_report(results, summary)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text("\n".join(json.dumps({
            "id": r.id, "label": r.label, "base_pred": r.base_pred,
            "gemma_preds": r.gemma_preds, "gemma_latencies_ms": r.gemma_latencies_ms,
            "gemma_valid": r.gemma_valid,
        }) for r in results) + "\n")
        print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
