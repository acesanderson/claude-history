from __future__ import annotations

import logging
import os
from collections.abc import Sequence

log = logging.getLogger(__name__)

_MODEL = "text-embedding-3-small"
_DIMS = 1536
_BATCH = 100


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = list(texts[i : i + _BATCH])
        resp = client.embeddings.create(model=_MODEL, input=batch)
        results.extend(item.embedding for item in resp.data)
        log.debug("embedded %d/%d texts", min(i + _BATCH, len(texts)), len(texts))
    return results
