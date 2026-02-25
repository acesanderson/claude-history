from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import psycopg2
import psycopg2.extras

from claude_history.models import Session
from claude_history.models import Turn

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
    is_subagent  BOOLEAN NOT NULL DEFAULT FALSE,
    title        TEXT
);

CREATE TABLE IF NOT EXISTS cc_turns (
    id           BIGSERIAL PRIMARY KEY,
    session_id   UUID NOT NULL REFERENCES cc_sessions(session_id),
    seq          INT NOT NULL,
    role         TEXT NOT NULL,
    content_raw  JSONB NOT NULL,
    content_text TEXT NOT NULL,
    content_tsv  TSVECTOR GENERATED ALWAYS AS (
                     to_tsvector('english', content_text)
                 ) STORED,
    embedding    vector(384),
    ts           TIMESTAMPTZ,
    UNIQUE (session_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_cc_turns_tsv
    ON cc_turns USING GIN(content_tsv);
CREATE INDEX IF NOT EXISTS idx_cc_turns_session
    ON cc_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_cc_sessions_proj
    ON cc_sessions(project_name);
"""


def create_schema(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def upsert_session(conn: psycopg2.extensions.connection, s: Session) -> None:
    sql = """
    INSERT INTO cc_sessions
        (session_id, project_path, project_name, git_branch, cc_version,
         started_at, ended_at, is_subagent)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (session_id) DO UPDATE
        SET ended_at   = EXCLUDED.ended_at,
            git_branch = EXCLUDED.git_branch
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                s.session_id,
                s.project_path,
                s.project_name,
                s.git_branch,
                s.cc_version,
                s.started_at,
                s.ended_at,
                s.is_subagent,
            ),
        )
    conn.commit()


def upsert_turns(conn: psycopg2.extensions.connection, turns: Sequence[Turn]) -> None:
    if not turns:
        return
    sql = """
    INSERT INTO cc_turns (session_id, seq, role, content_raw, content_text, ts)
    VALUES %s
    ON CONFLICT (session_id, seq) DO NOTHING
    """
    rows = [
        (
            t.session_id,
            t.seq,
            t.role,
            psycopg2.extras.Json(t.content_raw),
            t.content_text,
            t.ts,
        )
        for t in turns
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
    conn.commit()


def update_embedding(
    conn: psycopg2.extensions.connection,
    session_id: str,
    seq: int,
    vec: list[float],
) -> None:
    vec_str = "[" + ",".join(str(x) for x in vec) + "]"
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE cc_turns SET embedding = %s::vector WHERE session_id = %s AND seq = %s",
            (vec_str, session_id, seq),
        )
    conn.commit()


def update_title(
    conn: psycopg2.extensions.connection,
    session_id: str,
    title: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE cc_sessions SET title = %s WHERE session_id = %s AND title IS NULL",
            (title, session_id),
        )
    conn.commit()


def sessions_without_titles(
    conn: psycopg2.extensions.connection,
) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT session_id FROM cc_sessions WHERE title IS NULL")
        return [row[0] for row in cur.fetchall()]


def get_turns_for_session(
    conn: psycopg2.extensions.connection,
    session_id: str,
    offset: int = 0,
    limit: int = 50,
) -> list[dict]:
    sql = """
    SELECT id, seq, role, content_text, content_raw, ts
    FROM cc_turns
    WHERE session_id = %s
    ORDER BY seq
    LIMIT %s OFFSET %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (session_id, limit, offset))
        return [dict(r) for r in cur.fetchall()]


def search_fts(
    conn: psycopg2.extensions.connection,
    query: str,
    include_subagents: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    sql = """
    SELECT t.id, s.session_id, s.project_name, s.title,
           t.role, t.content_text, t.content_raw, t.ts
    FROM cc_turns t
    JOIN cc_sessions s USING (session_id)
    WHERE t.content_tsv @@ plainto_tsquery('english', %s)
      AND (%s OR NOT s.is_subagent)
    ORDER BY t.ts DESC
    LIMIT %s OFFSET %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (query, include_subagents, limit, offset))
        return [dict(r) for r in cur.fetchall()]


def search_vector(
    conn: psycopg2.extensions.connection,
    vec: list[float],
    include_subagents: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    vec_str = "[" + ",".join(str(x) for x in vec) + "]"
    sql = """
    SELECT t.id, s.session_id, s.project_name, s.title,
           t.role, t.content_text, t.content_raw,
           1 - (t.embedding <=> %s::vector) AS score, t.ts
    FROM cc_turns t
    JOIN cc_sessions s USING (session_id)
    WHERE t.embedding IS NOT NULL
      AND (%s OR NOT s.is_subagent)
    ORDER BY t.embedding <=> %s::vector
    LIMIT %s OFFSET %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (vec_str, include_subagents, vec_str, limit, offset))
        return [dict(r) for r in cur.fetchall()]


def has_any_embeddings(conn: psycopg2.extensions.connection) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM cc_turns WHERE embedding IS NOT NULL LIMIT 1")
        return cur.fetchone() is not None


def list_sessions(
    conn: psycopg2.extensions.connection,
    project: str | None = None,
    since=None,
    until=None,
    include_subagents: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    sql = """
    SELECT s.session_id, s.project_name, s.title,
           s.started_at, s.ended_at, s.is_subagent,
           COUNT(t.id) AS turn_count,
           snippet.content_text AS first_message_snippet
    FROM cc_sessions s
    LEFT JOIN cc_turns t USING (session_id)
    LEFT JOIN LATERAL (
        SELECT LEFT(content_text, 300) AS content_text
        FROM cc_turns
        WHERE session_id = s.session_id
          AND role = 'user'
          AND content_text <> ''
        ORDER BY seq
        LIMIT 1
    ) snippet ON true
    WHERE (%s IS NULL OR s.project_name = %s)
      AND (%s IS NULL OR s.started_at >= %s)
      AND (%s IS NULL OR s.started_at <= %s)
      AND (%s OR NOT s.is_subagent)
    GROUP BY s.session_id, snippet.content_text
    ORDER BY s.started_at DESC
    LIMIT %s OFFSET %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            sql,
            (
                project, project,
                since, since,
                until, until,
                include_subagents,
                limit, offset,
            ),
        )
        return [dict(r) for r in cur.fetchall()]
