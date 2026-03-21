from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from claude_history import db
from tests.conftest import make_record
from tests.conftest import write_jsonl

_UUID = "cccccccc-0000-0000-0000-000000000001"


def _write_session(tmp_path: Path, stem: str = _UUID, extra_records: list[str] | None = None) -> Path:
    records = [
        make_record(
            type="user",
            cwd="/Users/test/myproject",
            gitBranch="main",
            version="2.1.0",
            message={"role": "user", "content": "explain recursion"},
        ),
        make_record(
            type="assistant",
            message={"role": "assistant", "content": [{"type": "text", "text": "recursion is self-referential"}]},
        ),
    ]
    if extra_records:
        records += extra_records
    path = tmp_path / f"{stem}.jsonl"
    write_jsonl(path, records)
    return path


# ─── ingest_session ───────────────────────────────────────────────────────────

def test_ingest_session_creates_session_and_turns(tmp_path, patched_db):
    from claude_history.ingest import ingest_session
    path = _write_session(tmp_path)
    ingest_session(path)

    rows = db.list_sessions(patched_db)
    assert len(rows) == 1
    assert str(rows[0]["session_id"]) == _UUID

    turns = db.get_turns_for_session(patched_db, _UUID)
    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert turns[0]["content_text"] == "explain recursion"


def test_ingest_session_idempotent(tmp_path, patched_db):
    from claude_history.ingest import ingest_session
    path = _write_session(tmp_path)
    ingest_session(path)
    ingest_session(path)  # second run

    with patched_db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM cc_sessions")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM cc_turns")
        assert cur.fetchone()[0] == 2


def test_ingest_session_existing_title_not_regenerated(tmp_path, patched_db, monkeypatch):
    from claude_history.ingest import ingest_session
    path = _write_session(tmp_path)
    ingest_session(path)

    db.update_title(patched_db, _UUID, "Existing Title")

    call_count = 0

    def mock_generate_title(turns):
        nonlocal call_count
        call_count += 1
        return "New Title"

    monkeypatch.setattr("claude_history.ingest.titler.generate_title", mock_generate_title)
    ingest_session(path)  # re-ingest with existing title

    assert call_count == 0


def test_ingest_session_db_down_warns_stderr(tmp_path, monkeypatch, capsys):
    from claude_history.ingest import ingest_session

    def mock_open_conn(_db_name=None):
        raise OSError("connection refused")

    monkeypatch.setattr("claude_history.ingest._open_conn", mock_open_conn)

    path = _write_session(tmp_path)
    ingest_session(path)  # must not raise

    err = capsys.readouterr().err
    assert "DB unavailable" in err


def test_ingest_session_empty_file_is_noop(tmp_path, patched_db):
    from claude_history.ingest import ingest_session
    path = tmp_path / f"{_UUID}.jsonl"
    path.write_text("")
    ingest_session(path)

    rows = db.list_sessions(patched_db)
    assert rows == []


def test_ingest_session_embeds_nonempty_turns(tmp_path, patched_db):
    from claude_history.ingest import ingest_session
    path = _write_session(tmp_path)
    ingest_session(path)

    with patched_db.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM cc_turns WHERE embedding IS NOT NULL AND content_text <> ''"
        )
        count = cur.fetchone()[0]
    # Both turns have non-empty content_text → both should be embedded
    assert count == 2


def test_ingest_session_real_file(real_jsonl, patched_db):
    from claude_history.ingest import ingest_session
    ingest_session(real_jsonl)

    rows = db.list_sessions(patched_db)
    assert len(rows) >= 1
    turns = db.get_turns_for_session(patched_db, str(rows[0]["session_id"]))
    assert len(turns) > 0
