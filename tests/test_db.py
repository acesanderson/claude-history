from __future__ import annotations

from datetime import datetime
from datetime import timezone

import pytest

from claude_history import db
from tests.conftest import make_session
from tests.conftest import make_turn

_DT = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


# ─── schema idempotency ───────────────────────────────────────────────────────

def test_create_schema_idempotent(conn):
    db.create_schema(conn)  # second call; tables already exist
    db.create_schema(conn)  # third call; still fine


# ─── session upserts ─────────────────────────────────────────────────────────

def test_upsert_session_insert(conn):
    s = make_session()
    db.upsert_session(conn, s)
    with conn.cursor() as cur:
        cur.execute("SELECT session_id, project_name, is_subagent FROM cc_sessions")
        row = cur.fetchone()
    assert str(row[0]) == s.session_id
    assert row[1] == "myproject"
    assert row[2] is False


def test_upsert_session_conflict_updates_ended_at_and_git_branch(conn):
    s = make_session(ended_at=_DT, git_branch="main")
    db.upsert_session(conn, s)

    new_end = datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
    s2 = make_session(ended_at=new_end, git_branch="feature-x")
    db.upsert_session(conn, s2)

    with conn.cursor() as cur:
        cur.execute("SELECT ended_at, git_branch FROM cc_sessions WHERE session_id = %s", (s.session_id,))
        ended_at, branch = cur.fetchone()
    assert ended_at.date() == new_end.date()
    assert branch == "feature-x"


def test_upsert_session_conflict_preserves_title(conn):
    s = make_session()
    db.upsert_session(conn, s)
    db.update_title(conn, s.session_id, "My Title")

    # Re-upsert (simulating re-ingest)
    db.upsert_session(conn, s)

    with conn.cursor() as cur:
        cur.execute("SELECT title FROM cc_sessions WHERE session_id = %s", (s.session_id,))
        row = cur.fetchone()
    assert row[0] == "My Title"  # title not cleared


# ─── turn upserts ─────────────────────────────────────────────────────────────

def test_upsert_turns_insert(conn):
    s = make_session()
    db.upsert_session(conn, s)
    turns = [make_turn(seq=i, text=f"msg {i}") for i in range(3)]
    db.upsert_turns(conn, turns)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM cc_turns WHERE session_id = %s", (s.session_id,))
        assert cur.fetchone()[0] == 3


def test_upsert_turns_conflict_do_nothing(conn):
    s = make_session()
    db.upsert_session(conn, s)
    turns = [make_turn(seq=0, text="original")]
    db.upsert_turns(conn, turns)
    # Insert again — should silently ignore
    db.upsert_turns(conn, [make_turn(seq=0, text="overwrite attempt")])

    with conn.cursor() as cur:
        cur.execute("SELECT content_text FROM cc_turns WHERE session_id = %s AND seq = 0", (s.session_id,))
        assert cur.fetchone()[0] == "original"


def test_upsert_turns_empty_content_text_stored(conn):
    s = make_session()
    db.upsert_session(conn, s)
    t = make_turn(seq=0, role="assistant", text="")
    t.content_raw = [{"type": "tool_use", "id": "x", "name": "Bash", "input": {}}]
    db.upsert_turns(conn, [t])

    with conn.cursor() as cur:
        cur.execute("SELECT content_text FROM cc_turns WHERE session_id = %s", (s.session_id,))
        assert cur.fetchone()[0] == ""


# ─── embeddings ───────────────────────────────────────────────────────────────

def test_update_embedding(conn):
    s = make_session()
    db.upsert_session(conn, s)
    db.upsert_turns(conn, [make_turn(seq=0)])

    vec = [0.1] * 384
    db.update_embedding(conn, s.session_id, 0, vec)

    with conn.cursor() as cur:
        cur.execute("SELECT embedding IS NOT NULL FROM cc_turns WHERE session_id = %s AND seq = 0", (s.session_id,))
        assert cur.fetchone()[0] is True


def test_has_any_embeddings_false_when_none(conn):
    s = make_session()
    db.upsert_session(conn, s)
    db.upsert_turns(conn, [make_turn(seq=0)])
    assert db.has_any_embeddings(conn) is False


def test_has_any_embeddings_true_after_set(conn):
    s = make_session()
    db.upsert_session(conn, s)
    db.upsert_turns(conn, [make_turn(seq=0)])
    db.update_embedding(conn, s.session_id, 0, [0.0] * 384)
    assert db.has_any_embeddings(conn) is True


# ─── title ────────────────────────────────────────────────────────────────────

def test_update_title_sets_when_null(conn):
    s = make_session()
    db.upsert_session(conn, s)
    db.update_title(conn, s.session_id, "Great Title")

    with conn.cursor() as cur:
        cur.execute("SELECT title FROM cc_sessions WHERE session_id = %s", (s.session_id,))
        assert cur.fetchone()[0] == "Great Title"


def test_update_title_does_not_overwrite(conn):
    s = make_session()
    db.upsert_session(conn, s)
    db.update_title(conn, s.session_id, "First")
    db.update_title(conn, s.session_id, "Second")  # should be no-op

    with conn.cursor() as cur:
        cur.execute("SELECT title FROM cc_sessions WHERE session_id = %s", (s.session_id,))
        assert cur.fetchone()[0] == "First"


def test_sessions_without_titles(conn):
    s1 = make_session(session_id="aaaaaaaa-0000-0000-0000-000000000001")
    s2 = make_session(session_id="aaaaaaaa-0000-0000-0000-000000000002")
    db.upsert_session(conn, s1)
    db.upsert_session(conn, s2)
    db.update_title(conn, s1.session_id, "Titled")

    untitled = db.sessions_without_titles(conn)
    assert len(untitled) == 1
    assert str(untitled[0]) == s2.session_id


# ─── list_sessions ────────────────────────────────────────────────────────────

def test_list_sessions_basic(conn):
    s = make_session()
    db.upsert_session(conn, s)
    db.upsert_turns(conn, [make_turn(seq=0, role="user", text="first question")])

    rows = db.list_sessions(conn)
    assert len(rows) == 1
    assert str(rows[0]["session_id"]) == s.session_id
    assert rows[0]["turn_count"] == 1
    assert "first question" in (rows[0]["first_message_snippet"] or "")


def test_list_sessions_filter_by_project(conn):
    s1 = make_session(session_id="aaaaaaaa-0000-0000-0000-000000000001", project_name="proj-a")
    s2 = make_session(session_id="aaaaaaaa-0000-0000-0000-000000000002", project_name="proj-b")
    db.upsert_session(conn, s1)
    db.upsert_session(conn, s2)

    rows = db.list_sessions(conn, project="proj-a")
    assert len(rows) == 1
    assert rows[0]["project_name"] == "proj-a"


def test_list_sessions_excludes_subagents_by_default(conn):
    s1 = make_session(session_id="aaaaaaaa-0000-0000-0000-000000000001", is_subagent=False)
    s2 = make_session(session_id="aaaaaaaa-0000-0000-0000-000000000002", is_subagent=True)
    db.upsert_session(conn, s1)
    db.upsert_session(conn, s2)

    rows = db.list_sessions(conn, include_subagents=False)
    assert len(rows) == 1
    rows_all = db.list_sessions(conn, include_subagents=True)
    assert len(rows_all) == 2


def test_list_sessions_since_until(conn):
    early = datetime(2026, 1, 1, tzinfo=timezone.utc)
    late = datetime(2026, 6, 1, tzinfo=timezone.utc)
    s1 = make_session(session_id="aaaaaaaa-0000-0000-0000-000000000001", started_at=early)
    s2 = make_session(session_id="aaaaaaaa-0000-0000-0000-000000000002", started_at=late)
    db.upsert_session(conn, s1)
    db.upsert_session(conn, s2)

    rows = db.list_sessions(conn, since=datetime(2026, 3, 1, tzinfo=timezone.utc))
    assert len(rows) == 1
    assert str(rows[0]["session_id"]) == s2.session_id


# ─── FTS search ───────────────────────────────────────────────────────────────

def test_search_fts_finds_match(conn):
    s = make_session()
    db.upsert_session(conn, s)
    db.upsert_turns(conn, [make_turn(seq=0, text="implementing a database migration")])

    rows = db.search_fts(conn, "database migration")
    assert len(rows) == 1
    assert "database" in rows[0]["content_text"]


def test_search_fts_no_match(conn):
    s = make_session()
    db.upsert_session(conn, s)
    db.upsert_turns(conn, [make_turn(seq=0, text="hello world")])

    rows = db.search_fts(conn, "zxqfoo12345")
    assert rows == []


def test_search_fts_excludes_subagents_by_default(conn):
    s_main = make_session(session_id="aaaaaaaa-0000-0000-0000-000000000001", is_subagent=False)
    s_sub = make_session(session_id="aaaaaaaa-0000-0000-0000-000000000002", is_subagent=True)
    db.upsert_session(conn, s_main)
    db.upsert_session(conn, s_sub)
    db.upsert_turns(conn, [make_turn(seq=0, session_id=s_main.session_id, text="the quick brown fox")])
    db.upsert_turns(conn, [make_turn(seq=0, session_id=s_sub.session_id, text="the quick brown fox")])

    rows = db.search_fts(conn, "quick", include_subagents=False)
    assert len(rows) == 1
    rows_all = db.search_fts(conn, "quick", include_subagents=True)
    assert len(rows_all) == 2


def test_search_fts_skips_empty_content_text(conn):
    s = make_session()
    db.upsert_session(conn, s)
    t = make_turn(seq=0, text="")
    t.content_raw = [{"type": "tool_use"}]
    db.upsert_turns(conn, [t])

    rows = db.search_fts(conn, "tool")
    assert rows == []


# ─── vector search ────────────────────────────────────────────────────────────

def test_search_vector_finds_by_embedding(conn):
    s = make_session()
    db.upsert_session(conn, s)
    db.upsert_turns(conn, [make_turn(seq=0, text="hello")])
    vec = [0.5] * 384
    db.update_embedding(conn, s.session_id, 0, vec)

    rows = db.search_vector(conn, vec)
    assert len(rows) == 1
    assert rows[0]["score"] > 0.99  # identical vector → cosine ~1.0


def test_search_vector_empty_when_no_embeddings(conn):
    s = make_session()
    db.upsert_session(conn, s)
    db.upsert_turns(conn, [make_turn(seq=0, text="hello")])
    rows = db.search_vector(conn, [0.1] * 384)
    assert rows == []


# ─── get_turns_for_session ───────────────────────────────────────────────────

def test_get_turns_for_session_ordered(conn):
    s = make_session()
    db.upsert_session(conn, s)
    turns = [make_turn(seq=i, text=f"msg {i}") for i in range(5)]
    db.upsert_turns(conn, turns)

    rows = db.get_turns_for_session(conn, s.session_id)
    assert [r["seq"] for r in rows] == [0, 1, 2, 3, 4]


def test_get_turns_for_session_pagination(conn):
    s = make_session()
    db.upsert_session(conn, s)
    db.upsert_turns(conn, [make_turn(seq=i, text=f"t{i}") for i in range(10)])

    page1 = db.get_turns_for_session(conn, s.session_id, limit=3, offset=0)
    page2 = db.get_turns_for_session(conn, s.session_id, limit=3, offset=3)
    assert [r["seq"] for r in page1] == [0, 1, 2]
    assert [r["seq"] for r in page2] == [3, 4, 5]
