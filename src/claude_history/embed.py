from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_history.models import Turn

log = logging.getLogger(__name__)

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = int(os.environ.get("CH_EMBED_BATCH_SIZE", "64"))


def embed_turns(turns: list[Turn]) -> list[list[float] | None]:
    """
    Embed turns in chunks of EMBED_BATCH_SIZE.

    Skips turns with empty content_text (returns None for those positions).
    Validates model once before the loop.
    Returns embeddings in input order; None for skipped turns.
    """
    from conduit.embeddings.generate_embeddings import generate_embeddings
    from conduit.embeddings.generate_embeddings import validate_model

    validate_model(_MODEL)

    results: list[list[float] | None] = [None] * len(turns)
    active = [(i, t) for i, t in enumerate(turns) if t.content_text]

    for batch_start in range(0, len(active), EMBED_BATCH_SIZE):
        batch = active[batch_start : batch_start + EMBED_BATCH_SIZE]
        ids = [f"{t.session_id}:{t.seq}" for _, t in batch]
        docs = [t.content_text for _, t in batch]
        vecs = generate_embeddings(ids, docs, model=_MODEL)
        for (orig_idx, _), vec in zip(batch, vecs):
            results[orig_idx] = vec
        log.debug(
            "embedded batch %d-%d / %d",
            batch_start,
            batch_start + len(batch),
            len(active),
        )

    return results
