"""Answer a question grounded in the contents of a single open PDF.

Two-stage routing:
- Gemini 2.5 Pro answers every PDF question. It has the strongest native PDF
  understanding plus the lowest cost-per-call.
- For proof-shaped questions (regex-detected), Claude Sonnet 4.6 fires in
  parallel as a verifier. When both succeed, Claude's answer wins because it
  follows step-by-step proof structure more reliably. Otherwise we fall back
  to whichever model produced a non-empty result.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from config.settings import (
    ALI_ALLOW_CLOUD_FALLBACK,
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
)


@dataclass(frozen=True)
class PdfAnswer:
    text: str
    pdf_path: str
    backend: str  # "gemini" | "claude" | "stub"


_SYSTEM = (
    "You are Ali, helping the user understand a PDF that's open on their screen.\n"
    "You will be given the PDF's full text and the user's question.\n"
    "\n"
    "Rules:\n"
    "• Answer the question directly and in detail. The user wants the actual "
    "  answer, not a summary pointer or 'see page 5'.\n"
    "• The user may reference a question by label (\"#49\", \"problem 50\"), "
    "  by position (\"the first one\", \"second question\"), or by topic "
    "  (\"the idempotent one\", \"the subfield test\"). Pick whichever question "
    "  they most plausibly mean from the PDF and solve it. If the reference is "
    "  genuinely ambiguous (e.g. \"the third one\" with two equally good "
    "  matches), answer with one short clarifying question instead of guessing.\n"
    "• Use LaTeX for math. Inline math goes in $...$, display math in $$...$$. "
    "  Don't fall back to ascii like 'x^2' or 'sqrt(x)'.\n"
    "• Quote or paraphrase the relevant passages so the user can verify.\n"
    "• Plain prose. No markdown headers, no bullet vomit. A few short "
    "  paragraphs is fine when the answer needs steps or sub-points.\n"
    "• If the PDF doesn't contain enough to answer, say so plainly and "
    "  point to what is in the PDF that's closest.\n"
    "• No 'Based on the PDF…' preamble. Just answer."
)


# Gemini 2.5 Pro takes ~1M tokens of input, so we don't have to be precious
# about truncation — but extremely long PDFs still slow the call and cost
# more, so cap at a generous-but-bounded size.
_MAX_PDF_CHARS = 180_000

# Proof-shaped questions where Claude's step-by-step structure tends to beat
# Gemini's. Match conservatively — false positives just mean an extra parallel
# API call, not a wrong answer.
_PROOF_HINTS_RE = re.compile(
    r"\b(prove|show that|show R|verify|derive|if and only if|iff|"
    r"by induction|deduce|hence|theorem|lemma|corollary)\b",
    re.IGNORECASE,
)


def _looks_proof_heavy(question: str, pdf_text: str) -> bool:
    """Cheap heuristic: does the user's ask look like a proof / formal derivation?"""
    if _PROOF_HINTS_RE.search(question):
        return True
    # The question itself is short ("answer #49") but the *PDF* references a
    # proof — common case for math homework. Peek at the PDF too.
    if _PROOF_HINTS_RE.search(pdf_text[:8000]):
        return True
    return False


async def answer_from_pdf(
    question: str,
    pdf_path: Path,
    pdf_text: str,
) -> PdfAnswer:
    question = (question or "").strip()
    if not question:
        return PdfAnswer(text="", pdf_path=str(pdf_path), backend="stub")
    if not pdf_text.strip():
        return PdfAnswer(
            text=(
                "I can see the PDF but couldn't read any text from it — it "
                "might be a scan. Try copying the question into the chat."
            ),
            pdf_path=str(pdf_path),
            backend="stub",
        )

    if not (ALI_ALLOW_CLOUD_FALLBACK and GEMINI_API_KEY):
        return PdfAnswer(
            text=(
                "I'm in local-only mode and can't reason through the PDF right "
                "now. Enable cloud fallback to get a detailed answer."
            ),
            pdf_path=str(pdf_path),
            backend="stub",
        )

    truncated = pdf_text[:_MAX_PDF_CHARS]
    use_claude = bool(ANTHROPIC_API_KEY) and _looks_proof_heavy(question, truncated)

    gemini_task = asyncio.create_task(_call_gemini(question, pdf_path, truncated))
    claude_task: asyncio.Task[str] | None = None
    if use_claude:
        claude_task = asyncio.create_task(_call_claude(question, pdf_path, truncated))

    gemini_reply = await gemini_task
    claude_reply = await claude_task if claude_task is not None else ""

    # Prefer Claude when both succeed on proof-heavy questions.
    if claude_reply:
        print(f"[pdf_answer] using claude verifier ({len(claude_reply)} chars)")
        return PdfAnswer(text=claude_reply, pdf_path=str(pdf_path), backend="claude")
    if gemini_reply:
        return PdfAnswer(text=gemini_reply, pdf_path=str(pdf_path), backend="gemini")

    return PdfAnswer(
        text=(
            "I couldn't reach the model right now. Try again in a moment, or "
            "paste the question into the chat."
        ),
        pdf_path=str(pdf_path),
        backend="stub",
    )


async def _call_gemini(question: str, pdf_path: Path, pdf_text: str) -> str:
    try:
        from google import genai as _genai  # type: ignore
    except ImportError:
        return ""

    prompt = (
        f"{_SYSTEM}\n\n"
        f"PDF FILENAME: {pdf_path.name}\n\n"
        f"PDF CONTENT (full text, may be truncated):\n{pdf_text}\n\n"
        f"User's question: {question}\n\n"
        f"Detailed answer:"
    )

    loop = asyncio.get_event_loop()

    def _sync() -> str:
        client = _genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=_genai.types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1200,
            ),
        )
        return (response.text or "").strip()

    try:
        return await loop.run_in_executor(None, _sync)
    except Exception as exc:
        print(f"[pdf_answer][warn] gemini call failed: {exc}")
        return ""


async def _call_claude(question: str, pdf_path: Path, pdf_text: str) -> str:
    try:
        import anthropic  # type: ignore
    except ImportError:
        return ""

    user_msg = (
        f"PDF FILENAME: {pdf_path.name}\n\n"
        f"PDF CONTENT (full text, may be truncated):\n{pdf_text}\n\n"
        f"User's question: {question}\n\n"
        f"Detailed answer:"
    )

    loop = asyncio.get_event_loop()

    def _sync() -> str:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        parts = []
        for block in msg.content:
            text = getattr(block, "text", "") or ""
            if text:
                parts.append(text)
        return "".join(parts).strip()

    try:
        return await loop.run_in_executor(None, _sync)
    except Exception as exc:
        print(f"[pdf_answer][warn] claude call failed: {exc}")
        return ""
