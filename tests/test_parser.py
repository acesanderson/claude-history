from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_history.parser import parse_session_file
from tests.conftest import make_record
from tests.conftest import write_jsonl


# ─── helpers ─────────────────────────────────────────────────────────────────

def _session_file(tmp_path: Path, stem: str, records: list[str]) -> Path:
    path = tmp_path / f"{stem}.jsonl"
    write_jsonl(path, records)
    return path


SESSION_UUID = "aaaaaaaa-bbbb-cccc-dddd-000000000001"


# ─── basic parsing ────────────────────────────────────────────────────────────

def test_parse_real_file(real_jsonl):
    result = parse_session_file(real_jsonl)
    assert result is not None
    session, turns = result
    assert session.session_id == real_jsonl.stem
    assert session.project_path  # non-empty (from cwd field)
    assert session.project_name  # last path component
    assert len(turns) > 0
    assert turns[0].seq == 0


def test_session_id_from_filename(tmp_path):
    records = [
        make_record(type="user", message={"role": "user", "content": "hi"}),
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    session, _ = parse_session_file(path)
    assert session.session_id == SESSION_UUID


def test_metadata_from_first_record(tmp_path):
    records = [
        make_record(
            type="user",
            cwd="/Users/test/myproject",
            gitBranch="feature-x",
            version="2.5.0",
            message={"role": "user", "content": "hi"},
        )
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    session, _ = parse_session_file(path)
    assert session.project_path == "/Users/test/myproject"
    assert session.project_name == "myproject"
    assert session.git_branch == "feature-x"
    assert session.cc_version == "2.5.0"


# ─── record filtering ─────────────────────────────────────────────────────────

def test_progress_records_skipped(tmp_path):
    records = [
        make_record(type="progress", data={}),
        make_record(type="progress", data={}),
        make_record(type="user", message={"role": "user", "content": "hello"}),
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    _, turns = parse_session_file(path)
    assert len(turns) == 1
    assert turns[0].seq == 0


def test_file_history_snapshot_skipped(tmp_path):
    records = [
        make_record(type="file-history-snapshot"),
        make_record(type="user", message={"role": "user", "content": "yo"}),
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    _, turns = parse_session_file(path)
    assert len(turns) == 1


def test_system_turn_duration_skipped(tmp_path):
    records = [
        make_record(type="user", message={"role": "user", "content": "first"}),
        make_record(type="system", subtype="turn_duration", durationMs=1000),
        make_record(type="user", message={"role": "user", "content": "second"}),
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    _, turns = parse_session_file(path)
    assert len(turns) == 2
    assert turns[0].content_text == "first"
    assert turns[1].content_text == "second"
    assert turns[1].seq == 1


def test_system_compact_boundary_skipped(tmp_path):
    records = [
        make_record(type="user", message={"role": "user", "content": "question"}),
        make_record(type="system", subtype="compact_boundary"),
        make_record(type="assistant", message={"role": "assistant", "content": [{"type": "text", "text": "answer"}]}),
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    _, turns = parse_session_file(path)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"


def test_system_local_command_stored(tmp_path):
    records = [
        make_record(type="user", message={"role": "user", "content": "start"}),
        make_record(type="system", subtype="local_command", content="/copy"),
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    _, turns = parse_session_file(path)
    assert len(turns) == 2
    lc = turns[1]
    assert lc.role == "system"
    assert lc.content_text == ""
    assert lc.content_raw == {"subtype": "local_command", "content": "/copy"}
    assert lc.seq == 1


# ─── seq numbering ────────────────────────────────────────────────────────────

def test_seq_zero_indexed_skips_not_counted(tmp_path):
    records = [
        make_record(type="progress", data={}),           # skip
        make_record(type="user", message={"role": "user", "content": "a"}),   # seq=0
        make_record(type="system", subtype="turn_duration"),                   # skip
        make_record(type="assistant", message={"role": "assistant", "content": [{"type": "text", "text": "b"}]}),  # seq=1
        make_record(type="system", subtype="local_command", content="/x"),    # seq=2
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    _, turns = parse_session_file(path)
    assert [t.seq for t in turns] == [0, 1, 2]


# ─── content extraction ──────────────────────────────────────────────────────

def test_string_content_stored_as_is(tmp_path):
    records = [
        make_record(type="user", message={"role": "user", "content": "plain string"}),
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    _, turns = parse_session_file(path)
    assert turns[0].content_text == "plain string"
    assert turns[0].content_raw == "plain string"


def test_text_blocks_joined_with_double_newline(tmp_path):
    records = [
        make_record(type="assistant", message={"role": "assistant", "content": [
            {"type": "text", "text": "first"},
            {"type": "tool_use", "id": "x", "name": "Bash", "input": {}},
            {"type": "text", "text": "second"},
        ]}),
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    # Need a user turn first so metadata is extracted
    records2 = [
        make_record(type="user", message={"role": "user", "content": "go"}),
    ] + records
    path = _session_file(tmp_path, SESSION_UUID, records2)
    _, turns = parse_session_file(path)
    assistant_turn = next(t for t in turns if t.role == "assistant")
    assert assistant_turn.content_text == "first\n\nsecond"


def test_tool_only_turn_has_empty_content_text(tmp_path):
    records = [
        make_record(type="user", message={"role": "user", "content": "go"}),
        make_record(type="assistant", message={"role": "assistant", "content": [
            {"type": "tool_use", "id": "x", "name": "Bash", "input": {"command": "ls"}},
        ]}),
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    _, turns = parse_session_file(path)
    tool_turn = next(t for t in turns if t.role == "assistant")
    assert tool_turn.content_text == ""
    assert isinstance(tool_turn.content_raw, list)


def test_content_raw_preserved_full_list(tmp_path):
    content = [
        {"type": "thinking", "thinking": "hmm"},
        {"type": "text", "text": "hello"},
    ]
    records = [
        make_record(type="assistant", message={"role": "assistant", "content": content}),
    ]
    records = [make_record(type="user", message={"role": "user", "content": "x"})] + records
    path = _session_file(tmp_path, SESSION_UUID, records)
    _, turns = parse_session_file(path)
    assistant = next(t for t in turns if t.role == "assistant")
    assert assistant.content_raw == content


# ─── session metadata ─────────────────────────────────────────────────────────

def test_started_at_from_first_stored_turn(tmp_path):
    records = [
        make_record(type="progress", data={}, timestamp="2026-01-01T09:00:00.000Z"),  # skipped
        make_record(type="user", message={"role": "user", "content": "hi"}, timestamp="2026-01-01T10:00:00.000Z"),
        make_record(type="assistant", message={"role": "assistant", "content": "ok"}, timestamp="2026-01-01T11:00:00.000Z"),
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    session, _ = parse_session_file(path)
    assert session.started_at.hour == 10
    assert session.ended_at.hour == 11


def test_empty_file_returns_none(tmp_path):
    path = tmp_path / f"{SESSION_UUID}.jsonl"
    path.write_text("")
    assert parse_session_file(path) is None


def test_all_progress_returns_none(tmp_path):
    records = [
        make_record(type="progress", data={}),
        make_record(type="progress", data={}),
    ]
    path = _session_file(tmp_path, SESSION_UUID, records)
    assert parse_session_file(path) is None


# ─── subagent detection ───────────────────────────────────────────────────────

def test_is_subagent_false_for_normal_path(tmp_path):
    records = [make_record(type="user", message={"role": "user", "content": "hi"})]
    path = _session_file(tmp_path, SESSION_UUID, records)
    session, _ = parse_session_file(path)
    assert session.is_subagent is False


def test_is_subagent_true_for_subagent_path(tmp_path):
    subdir = tmp_path / "subagents"
    subdir.mkdir()
    records = [make_record(type="user", message={"role": "user", "content": "hi"})]
    path = subdir / "agent-abc123.jsonl"
    write_jsonl(path, records)
    session, _ = parse_session_file(path)
    assert session.is_subagent is True


def test_non_uuid_stem_hashed_to_uuid(tmp_path):
    records = [make_record(type="user", message={"role": "user", "content": "hi"})]
    path = _session_file(tmp_path, "agent-abc123", records)
    session, _ = parse_session_file(path)
    import uuid
    uuid.UUID(session.session_id)  # must not raise
    assert session.session_id != "agent-abc123"


def test_uuid_stem_used_verbatim(tmp_path):
    records = [make_record(type="user", message={"role": "user", "content": "hi"})]
    stem = "12345678-1234-5678-1234-567812345678"
    path = _session_file(tmp_path, stem, records)
    session, _ = parse_session_file(path)
    assert session.session_id == stem
