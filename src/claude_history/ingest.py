from __future__ import annotations

import logging
from pathlib import Path

import psycopg2

from claude_history import db, embed, parser

log = logging.getLogger(__name__)


def run(
    conn: psycopg2.extensions.connection,
    root: Path | None = None,
    with_embeddings: bool = False,
    include_subagents: bool = True,
) -> None:
    db.ensure_schema(conn)

    new_sessions = 0
    new_turns = 0

    for session, turns in parser.iter_sessions(root=root, include_subagents=include_subagents):
        if db.session_exists(conn, str(session.session_id)):
            continue
        db.upsert_session(conn, session)
        db.insert_turns(conn, turns)
        new_sessions += 1
        new_turns += len(turns)
        log.info("ingested %s (%d turns)", session.project_name, len(turns))

    log.info("done: %d new sessions, %d new turns", new_sessions, new_turns)

    if with_embeddings:
        _backfill_embeddings(conn)


def _backfill_embeddings(conn: psycopg2.extensions.connection) -> None:
    total = 0
    while True:
        rows = db.turns_missing_embeddings(conn)
        if not rows:
            break
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        vectors = embed.embed_texts(texts)
        db.update_embeddings(conn, list(zip(vectors, ids)))
        total += len(ids)
        log.info("embedded %d turns (total so far: %d)", len(ids), total)
