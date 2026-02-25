from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from claude_history import db
from claude_history import embed as embed_mod
from claude_history import parser
from claude_history import titler

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_DEFAULT_TRANSCRIPTS_DIR = Path.home() / ".claude" / "projects"


def _db_name() -> str:
    return os.environ.get("CH_DB", "claude_history")


def _open_conn(db_name: str):
    from dbclients.clients.postgres import get_postgres_client

    return get_postgres_client(client_type="context_db", dbname=db_name)()


def ingest_session(path: Path) -> None:
    """Parse a single JSONL transcript and persist to Postgres. Entry point for the hook."""
    result = parser.parse_session_file(path)
    if result is None:
        return

    session, turns = result
    db_name = _db_name()

    # Parse → upsert; also fetch existing title to avoid redundant LLM calls on re-ingest
    existing_title: str | None = None
    try:
        with _open_conn(db_name) as conn:
            db.create_schema(conn)
            db.upsert_session(conn, session)
            db.upsert_turns(conn, turns)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT title FROM cc_sessions WHERE session_id = %s",
                    (session.session_id,),
                )
                row = cur.fetchone()
                existing_title = row[0] if row else None
    except Exception as exc:
        print(f"claude-history: DB unavailable: {exc}", file=sys.stderr)
        return

    # Embed
    nonempty = [t for t in turns if t.content_text]
    vecs: list[list[float] | None] = []
    if nonempty:
        try:
            vecs = embed_mod.embed_turns(nonempty)
        except Exception as exc:
            print(f"claude-history: embed skipped: {exc}", file=sys.stderr)

    # Title — skip if already set
    title: str | None = None
    if not existing_title:
        try:
            title = titler.generate_title(turns)
        except Exception as exc:
            print(f"claude-history: title skipped: {exc}", file=sys.stderr)

    # Write back embeddings + title
    if vecs or title:
        try:
            with _open_conn(db_name) as conn:
                for turn, vec in zip(nonempty, vecs):
                    if vec is not None:
                        db.update_embedding(conn, session.session_id, turn.seq, vec)
                if title:
                    db.update_title(conn, session.session_id, title)
        except Exception as exc:
            print(f"claude-history: DB write-back failed: {exc}", file=sys.stderr)


def ingest_all(
    transcripts_dir: Path | None = None,
    backfill_embeddings: bool = False,
) -> None:
    """Scan all JSONL transcripts and upsert any not yet in the database."""
    transcripts_dir = transcripts_dir or Path(
        os.environ.get("TRANSCRIPTS_DIR", str(_DEFAULT_TRANSCRIPTS_DIR))
    )
    db_name = _db_name()

    try:
        with _open_conn(db_name) as conn:
            db.create_schema(conn)
    except Exception as exc:
        print(f"claude-history: DB unavailable: {exc}", file=sys.stderr)
        return

    paths = sorted(transcripts_dir.glob("**/*.jsonl"), key=lambda p: p.stat().st_mtime)
    log.info("scanning %d transcript files", len(paths))

    for path in paths:
        try:
            ingest_session(path)
        except Exception:
            log.debug("failed ingesting %s", path, exc_info=True)

    if backfill_embeddings:
        _backfill_embeddings(db_name)


def _backfill_embeddings(db_name: str) -> None:
    """Server-side cursor backfill: embed all turns where embedding IS NULL."""
    from conduit.embeddings.generate_embeddings import generate_embeddings
    from conduit.embeddings.generate_embeddings import validate_model

    _MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size = embed_mod.EMBED_BATCH_SIZE

    try:
        validate_model(_MODEL)
    except Exception as exc:
        print(f"claude-history: embed server unavailable: {exc}", file=sys.stderr)
        return

    total = 0
    try:
        with _open_conn(db_name) as conn:
            with conn.cursor("embed_backfill") as cur:  # named → server-side
                cur.itersize = batch_size
                cur.execute(
                    "SELECT session_id, seq, content_text FROM cc_turns"
                    " WHERE embedding IS NULL AND content_text <> '' ORDER BY id"
                )
                batch = cur.fetchmany(batch_size)
                while batch:
                    ids = [f"{row[0]}:{row[1]}" for row in batch]
                    docs = [row[2] for row in batch]
                    vecs = generate_embeddings(ids, docs, model=_MODEL)
                    # Write back via a separate (non-server-side) cursor
                    with conn.cursor() as wcur:
                        for (session_id, seq, _), vec in zip(batch, vecs):
                            vec_str = "[" + ",".join(str(x) for x in vec) + "]"
                            wcur.execute(
                                "UPDATE cc_turns SET embedding = %s::vector"
                                " WHERE session_id = %s AND seq = %s",
                                (vec_str, session_id, seq),
                            )
                    conn.commit()
                    total += len(batch)
                    log.info("backfilled %d embeddings (total: %d)", len(batch), total)
                    batch = cur.fetchmany(batch_size)
    except Exception as exc:
        print(f"claude-history: embed backfill failed: {exc}", file=sys.stderr)

    log.info("embedding backfill complete: %d turns", total)
