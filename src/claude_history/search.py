from __future__ import annotations

import logging

import psycopg2

from claude_history import db, embed

log = logging.getLogger(__name__)

_RRF_K = 60


def fts(
    conn: psycopg2.extensions.connection,
    query: str,
    limit: int = 20,
) -> list[dict]:
    return db.search_fts(conn, query, limit=limit)


def semantic(
    conn: psycopg2.extensions.connection,
    query: str,
    limit: int = 10,
) -> list[dict]:
    vec = embed.embed_texts([query])[0]
    return db.search_vector(conn, vec, limit=limit)


def hybrid(
    conn: psycopg2.extensions.connection,
    query: str,
    limit: int = 10,
) -> list[dict]:
    # Run both queries at 2x limit for fusion candidates.
    fts_results = db.search_fts(conn, query, limit=limit * 2)
    vec = embed.embed_texts([query])[0]
    vec_results = db.search_vector(conn, vec, limit=limit * 2)

    # RRF: score = sum of 1/(k + rank) across result sets.
    scores: dict[str, float] = {}
    by_key: dict[str, dict] = {}

    def _key(row: dict) -> str:
        return f"{row['project_name']}:{row['content'][:80]}"

    for rank, row in enumerate(fts_results):
        k = _key(row)
        scores[k] = scores.get(k, 0.0) + 1.0 / (_RRF_K + rank + 1)
        by_key[k] = row

    for rank, row in enumerate(vec_results):
        k = _key(row)
        scores[k] = scores.get(k, 0.0) + 1.0 / (_RRF_K + rank + 1)
        by_key[k] = row

    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [by_key[k] for k in ranked[:limit]]
