from __future__ import annotations

import logging
import os

from claude_history import db

log = logging.getLogger(__name__)

_RRF_K = 60


def _db_name() -> str:
    return os.environ.get("CH_DB", "claude_history")


def _open_conn(db_name: str):
    from dbclients.clients.postgres import get_postgres_client

    return get_postgres_client(client_type="context_db", dbname=db_name)()


def fts(
    query: str,
    include_subagents: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    with _open_conn(_db_name()) as conn:
        results = db.search_fts(
            conn, query, include_subagents=include_subagents, limit=limit, offset=offset
        )
    for r in results:
        r.setdefault("score", None)
    return results


def semantic(
    query: str,
    include_subagents: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    from conduit.embeddings.generate_embeddings import quick_embedding

    vec = quick_embedding(query, model="sentence-transformers/all-MiniLM-L6-v2")
    with _open_conn(_db_name()) as conn:
        return db.search_vector(
            conn, vec, include_subagents=include_subagents, limit=limit, offset=offset
        )


def hybrid(
    query: str,
    include_subagents: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    candidates = min((offset + limit) * 10, 200)

    with _open_conn(_db_name()) as conn:
        fts_rows = db.search_fts(
            conn, query, include_subagents=include_subagents, limit=candidates, offset=0
        )
        has_embeddings = db.has_any_embeddings(conn)

    if not has_embeddings:
        log.debug("no embeddings yet; falling back to FTS-only")
        for r in fts_rows:
            r.setdefault("score", None)
        return fts_rows[offset : offset + limit]

    from conduit.embeddings.generate_embeddings import quick_embedding

    vec = quick_embedding(query, model="sentence-transformers/all-MiniLM-L6-v2")

    with _open_conn(_db_name()) as conn:
        vec_rows = db.search_vector(
            conn, vec, include_subagents=include_subagents, limit=candidates, offset=0
        )

    # RRF fusion keyed on turn id
    scores: dict[int, float] = {}
    by_id: dict[int, dict] = {}

    for rank, row in enumerate(fts_rows):
        rid = row["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        by_id[rid] = row

    for rank, row in enumerate(vec_rows):
        rid = row["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        by_id[rid] = row

    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
    page = ranked[offset : offset + limit]
    results = []
    for rid in page:
        row = dict(by_id[rid])
        row["score"] = scores[rid]
        results.append(row)
    return results
