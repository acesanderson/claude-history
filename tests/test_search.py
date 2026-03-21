from __future__ import annotations

import pytest

from claude_history import db
from tests.conftest import make_session
from tests.conftest import make_turn


def _seed(conn, text: str = "implementing a database migration", session_id: str = "dddddddd-0000-0000-0000-000000000001"):
    """Insert one session + one turn with embedding."""
    s = make_session(session_id=session_id)
    t = make_turn(seq=0, session_id=session_id, text=text)
    db.upsert_session(conn, s)
    db.upsert_turns(conn, [t])
    return s, t


# ─── FTS ─────────────────────────────────────────────────────────────────────

def test_fts_finds_keyword(patched_db):
    from claude_history.search import fts
    _seed(patched_db, "implementing a database migration")
    results = fts("database migration")
    assert len(results) >= 1
    assert any("database" in r["content_text"] for r in results)


def test_fts_no_results_for_garbage(patched_db):
    from claude_history.search import fts
    _seed(patched_db)
    results = fts("xyzqwerty99999")
    assert results == []


def test_fts_score_is_none(patched_db):
    from claude_history.search import fts
    _seed(patched_db, "database schema design")
    results = fts("database")
    assert results[0].get("score") is None


def test_fts_excludes_subagents_by_default(patched_db):
    from claude_history.search import fts
    s_main = make_session(session_id="dddddddd-0000-0000-0000-000000000001", is_subagent=False)
    s_sub = make_session(session_id="dddddddd-0000-0000-0000-000000000002", is_subagent=True)
    db.upsert_session(patched_db, s_main)
    db.upsert_session(patched_db, s_sub)
    db.upsert_turns(patched_db, [make_turn(seq=0, session_id=s_main.session_id, text="the quick brown fox")])
    db.upsert_turns(patched_db, [make_turn(seq=0, session_id=s_sub.session_id, text="the quick brown fox")])

    assert len(fts("quick")) == 1
    assert len(fts("quick", include_subagents=True)) == 2


# ─── Semantic ────────────────────────────────────────────────────────────────

def test_semantic_returns_results_with_scores(patched_db):
    from claude_history.search import semantic
    from claude_history.embed import embed_turns

    s, t = _seed(patched_db, "Python list comprehensions are faster than map")
    vecs = embed_turns([t])
    db.update_embedding(patched_db, s.session_id, t.seq, vecs[0])

    results = semantic("list comprehensions in Python")
    assert len(results) >= 1
    assert results[0]["score"] is not None
    assert 0.0 <= results[0]["score"] <= 1.0


def test_semantic_empty_when_no_embeddings(patched_db):
    from claude_history.search import semantic
    _seed(patched_db, "some content without embedding")
    results = semantic("some content")
    assert results == []


# ─── Hybrid ──────────────────────────────────────────────────────────────────

def test_hybrid_returns_results(patched_db):
    from claude_history.search import hybrid
    from claude_history.embed import embed_turns

    s, t = _seed(patched_db, "refactoring legacy code in Python")
    vecs = embed_turns([t])
    db.update_embedding(patched_db, s.session_id, t.seq, vecs[0])

    results = hybrid("refactoring Python code")
    assert len(results) >= 1


def test_hybrid_falls_back_to_fts_when_no_embeddings(patched_db):
    from claude_history.search import hybrid
    _seed(patched_db, "database migration strategy")
    results = hybrid("database migration")
    assert len(results) >= 1
    # FTS fallback: score is None
    assert results[0].get("score") is None


def test_hybrid_rrf_scores_present_when_embeddings_exist(patched_db):
    from claude_history.search import hybrid
    from claude_history.embed import embed_turns

    s, t = _seed(patched_db, "optimizing SQL queries for large tables")
    vecs = embed_turns([t])
    db.update_embedding(patched_db, s.session_id, t.seq, vecs[0])

    results = hybrid("SQL optimization")
    assert len(results) >= 1
    assert results[0]["score"] is not None
    assert results[0]["score"] > 0


def test_hybrid_pagination(patched_db):
    from claude_history.search import hybrid
    from claude_history.embed import embed_turns

    # Insert 5 sessions each with one turn matching "python"
    for i in range(5):
        sid = f"dddddddd-0000-0000-0000-00000000000{i+1}"
        s = make_session(session_id=sid)
        t = make_turn(seq=0, session_id=sid, text=f"python programming example {i}")
        db.upsert_session(patched_db, s)
        db.upsert_turns(patched_db, [t])
        vecs = embed_turns([t])
        db.update_embedding(patched_db, sid, 0, vecs[0])

    page1 = hybrid("python", limit=3, offset=0)
    page2 = hybrid("python", limit=3, offset=3)
    assert len(page1) == 3
    # page2 should have the remaining 2
    all_ids = {r["session_id"] for r in page1} | {r["session_id"] for r in page2}
    assert len(all_ids) == 5
