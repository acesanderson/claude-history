from __future__ import annotations

import pytest

from claude_history.titler import _build_title_transcript
from claude_history.titler import generate_title
from tests.conftest import make_turn


# ─── _build_title_transcript ─────────────────────────────────────────────────

def test_build_empty_turns_returns_empty():
    assert _build_title_transcript([]) == ""


def test_build_all_empty_content_text():
    turns = [make_turn(seq=i, text="") for i in range(5)]
    assert _build_title_transcript(turns) == ""


def test_build_includes_role_prefix():
    turns = [make_turn(seq=0, role="user", text="hello")]
    result = _build_title_transcript(turns)
    assert result.startswith("USER:")


def test_build_small_session_all_turns_included():
    # < 30 turns: all should appear
    turns = [make_turn(seq=i, text=f"turn {i}") for i in range(10)]
    result = _build_title_transcript(turns)
    for i in range(10):
        assert f"turn {i}" in result


def test_build_large_session_first_30_last_10():
    # 50 turns: expect first 30 and last 10, middle 10 absent
    turns = [make_turn(seq=i, text=f"msg {i:03d}") for i in range(50)]
    result = _build_title_transcript(turns)
    assert "msg 000" in result    # first
    assert "msg 029" in result    # 30th
    assert "msg 049" in result    # last
    # Middle turns (30-39) should NOT be in the transcript
    assert "msg 035" not in result


def test_build_deduplicates_head_tail_overlap():
    # 35 turns: head=30, tail=last 10 → last 5 of head = first 5 of tail, no dupes
    turns = [make_turn(seq=i, text=f"msg {i}") for i in range(35)]
    result = _build_title_transcript(turns)
    # msg 30 appears in both tail and NOT head (head is 0-29)
    count = result.count("msg 30")
    assert count == 1


def test_build_respects_token_budget():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    # 5 very long turns
    long_text = "word " * 10_000
    turns = [make_turn(seq=i, text=long_text) for i in range(5)]
    result = _build_title_transcript(turns, budget=500)
    token_count = len(enc.encode(result))
    # Should be roughly budget (within 20% due to per-turn truncation approximation)
    assert token_count <= 600


# ─── generate_title ───────────────────────────────────────────────────────────

def test_generate_title_returns_string():
    turns = [
        make_turn(seq=0, role="user", text="How do I implement a binary search tree in Python?"),
        make_turn(seq=1, role="assistant", text="I'll show you how to implement a BST with insert, search, and delete operations."),
    ]
    title = generate_title(turns)
    assert title is not None
    assert isinstance(title, str)
    assert len(title) > 0


def test_generate_title_no_empty_turns_returns_none():
    turns = [make_turn(seq=0, text="")]
    title = generate_title(turns)
    assert title is None


def test_generate_title_reasonable_length():
    turns = [
        make_turn(seq=0, role="user", text="Write me a Flask REST API with authentication and rate limiting"),
        make_turn(seq=1, role="assistant", text="Here is a complete Flask API implementation with JWT auth and rate limiting using Flask-Limiter."),
    ]
    title = generate_title(turns)
    assert title is not None
    word_count = len(title.split())
    # Spec says 5-10 words; allow some slack for the LLM
    assert 3 <= word_count <= 15
