from __future__ import annotations

import os
import time
from typing import Sequence

import numpy as np
from google import genai
from google.genai import types


def prepare_embedding_text(text: str, *, task: str = "clustering") -> str:
    return f"task: {task} | query: {text}"


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32, copy=False)


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = "gemini-embedding-2",
    output_dimensionality: int = 768,
    batch_size: int = 100,
    max_retries: int = 8,
    normalize: bool = True,
    task: str = "clustering",
) -> np.ndarray:
    """Embed texts with Gemini and optionally L2-normalize rows."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env or export it.")

    client = genai.Client(api_key=api_key)
    all_vectors: list[np.ndarray] = []
    prepared = [prepare_embedding_text(text, task=task) for text in texts]

    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
                )
                embeddings = getattr(response, "embeddings", None)
                if not embeddings or len(embeddings) != len(batch):
                    raise RuntimeError(f"Expected {len(batch)} embeddings, got {0 if embeddings is None else len(embeddings)}")
                vectors = np.vstack([np.asarray(item.values, dtype=np.float32) for item in embeddings])
                all_vectors.append(l2_normalize(vectors) if normalize else vectors)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == max_retries:
                    raise RuntimeError(f"Embedding batch failed after {max_retries} attempts: {last_error}") from exc
                sleep_s = 60 + min(60, 10 * attempt) if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc) else min(30, 2 ** (attempt - 1))
                time.sleep(sleep_s)

    return np.vstack(all_vectors) if all_vectors else np.empty((0, output_dimensionality), dtype=np.float32)
