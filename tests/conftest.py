from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import pytest

# Point all code at claude_history DB; tables live in cc_test schema
os.environ["CH_DB"] = "claude_history"

_SCHEMA = "cc_test"

# ─── low-level connection helper ─────────────────────────────────────────────

@contextmanager
def open_test_conn():
    from dbclients.clients.postgres import get_postgres_client
    from claude_history import db as db_mod

    with get_postgres_client(client_type="context_db", dbname="claude_history")() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {_SCHEMA}, public")
        yield conn


# ─── session-scoped schema setup ─────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def schema():
    """Drop + recreate cc_test schema at the start of every test session."""
    from claude_history import db as db_mod
    from dbclients.clients.postgres import get_postgres_client

    # Bootstrap: need to exist in public scope to run CREATE SCHEMA
    with get_postgres_client(client_type="context_db", dbname="claude_history")() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
            cur.execute(f"CREATE SCHEMA {_SCHEMA}")
            cur.execute(f"SET search_path TO {_SCHEMA}, public")
        conn.commit()
        db_mod.create_schema(conn)

    yield

    with get_postgres_client(client_type="context_db", dbname="claude_history")() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
        conn.commit()


# ─── per-test fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def conn(schema):
    """Fresh connection scoped to cc_test schema; tables truncated before each test."""
    with open_test_conn() as c:
        with c.cursor() as cur:
            cur.execute(f"TRUNCATE {_SCHEMA}.cc_turns, {_SCHEMA}.cc_sessions CASCADE")
        c.commit()
        yield c


@pytest.fixture
def patched_db(conn, monkeypatch):
    """
    Monkeypatches _open_conn in ingest / search / skill modules so integration
    tests hit the cc_test schema instead of the real DB.
    """
    @contextmanager
    def mock_open(_db_name=None):
        yield conn

    monkeypatch.setattr("claude_history.ingest._open_conn", mock_open)
    monkeypatch.setattr("claude_history.search._open_conn", mock_open)
    monkeypatch.setattr("claude_history.skill._open_conn", mock_open)
    yield conn


# ─── JSONL helpers ────────────────────────────────────────────────────────────

def make_record(**kwargs) -> str:
    base = {
        "cwd": "/Users/test/myproject",
        "sessionId": "aaaaaaaa-0000-0000-0000-000000000001",
        "version": "2.1.0",
        "gitBranch": "main",
        "timestamp": "2026-01-01T10:00:00.000Z",
    }
    base.update(kwargs)
    return json.dumps(base)


def write_jsonl(path: Path, records: list[str]) -> None:
    path.write_text("\n".join(records) + "\n")


@pytest.fixture
def real_jsonl() -> Path:
    """First real JSONL transcript found in ~/.claude/projects/."""
    root = Path.home() / ".claude" / "projects"
    candidates = sorted(root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        pytest.skip("no real JSONL transcripts found")
    return candidates[0]


# ─── model factories ─────────────────────────────────────────────────────────

def make_session(**kwargs):
    from claude_history.models import Session
    defaults = dict(
        session_id="aaaaaaaa-0000-0000-0000-000000000001",
        project_path="/Users/test/myproject",
        project_name="myproject",
        git_branch="main",
        cc_version="2.1.0",
        started_at=datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
        is_subagent=False,
    )
    defaults.update(kwargs)
    return Session(**defaults)


def make_turn(seq: int = 0, role: str = "user", text: str = "hello", **kwargs):
    from claude_history.models import Turn
    defaults = dict(
        session_id="aaaaaaaa-0000-0000-0000-000000000001",
        seq=seq,
        role=role,
        content_raw=text if role != "system" else {"subtype": "local_command", "content": None},
        content_text=text if role != "system" else "",
        ts=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=seq),
    )
    defaults.update(kwargs)
    return Turn(**defaults)
