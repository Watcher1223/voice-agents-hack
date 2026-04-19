"""
Lazy sentence-transformers wrapper.

We load the MiniLM weights once per process and cache the model globally.
The encoder is imported at call time so plain metadata workflows (file
lookup, user profile) work without the ~150MB torch/sentence-transformers
import cost.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    from sentence_transformers import SentenceTransformer  # type: ignore


_model_lock = threading.Lock()
_model: "SentenceTransformer | None" = None
_model_name: str | None = None

EMBED_DIM = 384  # MiniLM-L6-v2


def _load(model_name: str) -> "SentenceTransformer":
    global _model, _model_name
    with _model_lock:
        if _model is not None and _model_name == model_name:
            return _model
        from sentence_transformers import SentenceTransformer  # type: ignore

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        return _model


def warmup(model_name: str) -> None:
    """Pre-load the encoder in a background thread; safe to call more than once."""
    try:
        _load(model_name)
    except Exception as exc:
        print(f"[disk-index] embedder warmup failed: {exc}")


def embed_texts(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int = 64,
    show_progress: bool = False,
):
    """Return an (N, EMBED_DIM) numpy array with normalised embeddings."""
    import numpy as np

    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")
    model = _load(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return vectors.astype("float32", copy=False)


def embed_query(query: str, *, model_name: str):
    """Embed a single query string; shape (EMBED_DIM,)."""
    import numpy as np

    vecs = embed_texts([query], model_name=model_name, batch_size=1)
    if vecs.shape[0] == 0:
        return np.zeros(EMBED_DIM, dtype="float32")
    return vecs[0]
