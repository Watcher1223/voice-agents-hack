"""Answer a question grounded in the contents of a single open PDF.

Unlike the spoken-style RAG answer (which is capped at one or two
sentences), this path is meant to give the user a real, detailed answer —
the use case is "I'm looking at this PDF, what's the answer to question
3?". Gemini is the primary backend; if it's not available we return a
short stub so the caller can decide to fall back.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from config.settings import ALI_ALLOW_CLOUD_FALLBACK, GEMINI_API_KEY


@dataclass(frozen=True)
class PdfAnswer:
    text: str
    pdf_path: str
    backend: str  # "gemini" | "stub"


_SYSTEM = (
    "You are Ali, helping the user understand a PDF that's open on their screen.\n"
    "You will be given the PDF's full text and the user's question.\n"
    "\n"
    "Rules:\n"
    "• Answer the question directly and in detail. The user wants the actual "
    "  answer, not a summary pointer or 'see page 5'.\n"
    "• If the question references a numbered question (e.g. \"question 3\"), "
    "  find that question in the PDF and answer it as if you were solving it.\n"
    "• Quote or paraphrase the relevant passages so the user can verify.\n"
    "• Plain prose. No markdown headers, no bullet vomit. A few short "
    "  paragraphs is fine when the answer needs steps or sub-points.\n"
    "• If the PDF doesn't contain enough to answer, say so plainly and "
    "  point to what is in the PDF that's closest.\n"
    "• No 'Based on the PDF…' preamble. Just answer."
)


# Gemini 2.5 Flash takes ~1M tokens of input, so we don't have to be
# precious about truncation — but extremely long PDFs still slow the call
# and cost more, so cap at a generous-but-bounded size.
_MAX_PDF_CHARS = 180_000


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

    if ALI_ALLOW_CLOUD_FALLBACK and GEMINI_API_KEY:
        reply = await _call_gemini(question, pdf_path, pdf_text[:_MAX_PDF_CHARS])
        if reply:
            return PdfAnswer(text=reply, pdf_path=str(pdf_path), backend="gemini")

    return PdfAnswer(
        text=(
            "I'm in local-only mode and can't reason through the PDF right "
            "now. Enable cloud fallback to get a detailed answer."
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
            model="gemini-2.5-flash",
            contents=prompt,
            config=_genai.types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=900,
            ),
        )
        return (response.text or "").strip()

    try:
        return await loop.run_in_executor(None, _sync)
    except Exception as exc:
        print(f"[pdf_answer][warn] gemini call failed: {exc}")
        return ""
