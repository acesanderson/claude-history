from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import psycopg2
import psycopg2.extras

from claude_history.models import Session, Turn

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cc_sessions (
    session_id   UUID PRIMARY KEY,
    project_path TEXT NOT NULL,
    project_name TEXT NOT NULL,
    git_branch   TEXT,
    cc_version   TEXT,
    started_at   TIMESTAMPTZ,
    ended_at     TIMESTAMPTZ,
    turn_count   INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cc_turns (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES cc_sessions(session_id),
    seq         INT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    content_tsv TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('english', content)
                ) STORED,
    embedding   vector(1536),
    ts          TIMESTAMPTZ,
    UNIQUE (session_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_cc_turns_tsv
    ON cc_turns USING GIN(content_tsv);
CREATE INDEX IF NOT EXISTS idx_cc_turns_session
    ON cc_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_cc_sessions_proj
    ON cc_sessions(project_name);
"""


def ensure_schema(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def session_exists(conn: psycopg2.extensions.connection, session_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM cc_sessions WHERE session_id = %s", (session_id,))
        return cur.fetchone() is not None


def upsert_session(conn: psycopg2.extensions.connection, s: Session) -> None:
    sql = """
    INSERT INTO cc_sessions
        (session_id, project_path, project_name, git_branch, cc_version,
         started_at, ended_at, turn_count)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (session_id) DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            str(s.session_id),
            s.project_path,
            s.project_name,
            s.git_branch,
            s.cc_version,
            s.started_at,
            s.ended_at,
            s.turn_count,
        ))
    conn.commit()


def insert_turns(conn: psycopg2.extensions.connection, turns: Sequence[Turn]) -> None:
    sql = """
    INSERT INTO cc_turns (session_id, seq, role, content, ts)
    VALUES %s
    ON CONFLICT (session_id, seq) DO NOTHING
    """
    rows = [
        (str(t.session_id), t.seq, t.role, t.content, t.ts)
        for t in turns
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
    conn.commit()


def turns_missing_embeddings(
    conn: psycopg2.extensions.connection,
    batch_size: int = 100,
) -> list[tuple[int, str]]:
    sql = """
    SELECT id, content FROM cc_turns
    WHERE embedding IS NULL
    ORDER BY id
    LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (batch_size,))
        return cur.fetchall()


def update_embeddings(
    conn: psycopg2.extensions.connection,
    rows: list[tuple[list[float], int]],
) -> None:
    sql = "UPDATE cc_turns SET embedding = %s WHERE id = %s"
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=100)
    conn.commit()


def search_fts(
    conn: psycopg2.extensions.connection,
    query: str,
    limit: int = 20,
) -> list[dict]:
    sql = """
    SELECT s.project_name, t.role, t.content, t.ts,
           ts_rank(t.content_tsv, plainto_tsquery('english', %s)) AS rank
    FROM cc_turns t
    JOIN cc_sessions s USING (session_id)
    WHERE t.content_tsv @@ plainto_tsquery('english', %s)
    ORDER BY rank DESC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (query, query, limit))
        return [dict(r) for r in cur.fetchall()]


def search_vector(
    conn: psycopg2.extensions.connection,
    embedding: list[float],
    limit: int = 10,
) -> list[dict]:
    sql = """
    SELECT s.project_name, t.role, t.content, t.ts,
           1 - (t.embedding <=> %s::vector) AS score
    FROM cc_turns t
    JOIN cc_sessions s USING (session_id)
    WHERE t.embedding IS NOT NULL
    ORDER BY t.embedding <=> %s::vector
    LIMIT %s
    """
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (vec_str, vec_str, limit))
        return [dict(r) for r in cur.fetchall()]
