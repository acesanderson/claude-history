from __future__ import annotations

from datetime import datetime
from datetime import timezone

from claude_history import db
from tests.conftest import make_session
from tests.conftest import make_turn


def _seed_session(conn, session_id: str, text: str, is_subagent: bool = False):
    s = make_session(session_id=session_id, is_subagent=is_subagent)
    t = make_turn(seq=0, session_id=session_id, text=text)
    db.upsert_session(conn, s)
    db.upsert_turns(conn, [t])
    return s, t


def test_skill_list_sessions(patched_db):
    from claude_history.skill import list_sessions
    _seed_session(patched_db, "eeeeeeee-0000-0000-0000-000000000001", "hello")
    _seed_session(patched_db, "eeeeeeee-0000-0000-0000-000000000002", "world")

    rows = list_sessions()
    assert len(rows) == 2
    # Returns expected fields
    assert "session_id" in rows[0]
    assert "project_name" in rows[0]
    assert "turn_count" in rows[0]
    assert "first_message_snippet" in rows[0]


def test_skill_list_sessions_excludes_subagents(patched_db):
    from claude_history.skill import list_sessions
    _seed_session(patched_db, "eeeeeeee-0000-0000-0000-000000000001", "main session")
    _seed_session(patched_db, "eeeeeeee-0000-0000-0000-000000000002", "sub session", is_subagent=True)

    rows = list_sessions()
    assert len(rows) == 1
    rows_all = list_sessions(include_subagents=True)
    assert len(rows_all) == 2


def test_skill_search_fts(patched_db):
    from claude_history.skill import search
    _seed_session(patched_db, "eeeeeeee-0000-0000-0000-000000000001", "implementing vector embeddings")

    results = search("vector embeddings", mode="fts")
    assert len(results) >= 1
    assert "session_id" in results[0]
    assert "content_text" in results[0]
    assert "role" in results[0]


def test_skill_search_hybrid_fallback(patched_db):
    from claude_history.skill import search
    _seed_session(patched_db, "eeeeeeee-0000-0000-0000-000000000001", "database connection pooling")

    # No embeddings → falls back to FTS
    results = search("database connection", mode="hybrid")
    assert len(results) >= 1


def test_skill_get_session_turns(patched_db):
    from claude_history.skill import get_session_turns
    sid = "eeeeeeee-0000-0000-0000-000000000001"
    s = make_session(session_id=sid)
    turns = [make_turn(seq=i, session_id=sid, text=f"turn {i}") for i in range(5)]
    db.upsert_session(patched_db, s)
    db.upsert_turns(patched_db, turns)

    rows = get_session_turns(sid)
    assert len(rows) == 5
    assert rows[0]["seq"] == 0
    assert "content_text" in rows[0]
    assert "content_raw" in rows[0]


def test_skill_get_session_turns_pagination(patched_db):
    from claude_history.skill import get_session_turns
    sid = "eeeeeeee-0000-0000-0000-000000000001"
    s = make_session(session_id=sid)
    db.upsert_session(patched_db, s)
    db.upsert_turns(patched_db, [make_turn(seq=i, session_id=sid, text=f"t{i}") for i in range(10)])

    page1 = get_session_turns(sid, limit=4, offset=0)
    page2 = get_session_turns(sid, limit=4, offset=4)
    assert len(page1) == 4
    assert len(page2) == 4
    assert page1[0]["seq"] == 0
    assert page2[0]["seq"] == 4
