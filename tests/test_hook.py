from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import make_record
from tests.conftest import write_jsonl

_UUID = "ffffffff-0000-0000-0000-000000000001"


def _run_hook(stdin_payload: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "claude_history.hook"],
        input=json.dumps(stdin_payload),
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )


# ─── stdin parsing ────────────────────────────────────────────────────────────

def test_hook_missing_transcript_path_exits_0():
    result = _run_hook({"session_id": _UUID})
    assert result.returncode == 0


def test_hook_invalid_json_exits_0():
    proc = subprocess.run(
        [sys.executable, "-m", "claude_history.hook"],
        input="not valid json",
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )
    assert proc.returncode == 0
    assert "failed to read stdin" in proc.stderr


def test_hook_nonexistent_path_exits_0():
    result = _run_hook({"transcript_path": "/nonexistent/path/abc.jsonl"})
    assert result.returncode == 0
    assert "not found" in result.stderr


def test_hook_valid_transcript_exits_0(tmp_path):
    """Hook runs to completion and exits 0 even if DB/embed/title fail."""
    records = [
        make_record(type="user", message={"role": "user", "content": "hello hook"}),
    ]
    path = tmp_path / f"{_UUID}.jsonl"
    write_jsonl(path, [records[0]])

    result = _run_hook({"transcript_path": str(path)})
    # Must exit 0 regardless of whether DB is reachable
    assert result.returncode == 0
