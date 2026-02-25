"""Pure library interface for the Claude Code skill.

No argparse, no __main__, no side effects on import.
All functions are synchronous.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Literal

if TYPE_CHECKING:
    pass


def _db_name() -> str:
    return os.environ.get("CH_DB", "claude_history")


def _open_conn(db_name: str):
    from dbclients.clients.postgres import get_postgres_client

    return get_postgres_client(client_type="context_db", dbname=db_name)()


def search(
    query: str,
    mode: Literal["fts", "semantic", "hybrid"] = "hybrid",
    limit: int = 20,
    offset: int = 0,
    include_subagents: bool = False,
) -> list[dict]:
    """
    Returns: [{session_id, project_name, title, role, content_text, content_raw, ts, score}]
    """
    from claude_history import search as search_mod

    if mode == "fts":
        return search_mod.fts(
            query,
            include_subagents=include_subagents,
            limit=limit,
            offset=offset,
        )
    if mode == "semantic":
        return search_mod.semantic(
            query,
            include_subagents=include_subagents,
            limit=limit,
            offset=offset,
        )
    return search_mod.hybrid(
        query,
        include_subagents=include_subagents,
        limit=limit,
        offset=offset,
    )


def list_sessions(
    project: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    include_subagents: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """
    Returns: [{session_id, project_name, title, started_at, ended_at,
               turn_count, is_subagent, first_message_snippet}]
    """
    from claude_history import db

    with _open_conn(_db_name()) as conn:
        return db.list_sessions(
            conn,
            project=project,
            since=since,
            until=until,
            include_subagents=include_subagents,
            limit=limit,
            offset=offset,
        )


def get_session_turns(
    session_id: str,
    offset: int = 0,
    limit: int = 50,
) -> list[dict]:
    """
    Returns: [{id, seq, role, content_text, content_raw, ts}]
    Paginate with offset to handle large sessions.
    """
    from claude_history import db

    with _open_conn(_db_name()) as conn:
        return db.get_turns_for_session(conn, session_id, offset=offset, limit=limit)
