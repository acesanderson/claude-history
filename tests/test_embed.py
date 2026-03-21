from __future__ import annotations

from tests.conftest import make_turn


def test_embed_turns_correct_dims():
    from claude_history.embed import embed_turns
    turns = [make_turn(seq=0, text="hello world")]
    vecs = embed_turns(turns)
    assert len(vecs) == 1
    assert vecs[0] is not None
    assert len(vecs[0]) == 384


def test_embed_turns_empty_content_text_returns_none():
    from claude_history.embed import embed_turns
    t = make_turn(seq=0, text="")
    vecs = embed_turns([t])
    assert len(vecs) == 1
    assert vecs[0] is None


def test_embed_turns_mixed_empty_and_nonempty():
    from claude_history.embed import embed_turns
    turns = [
        make_turn(seq=0, text="real content"),
        make_turn(seq=1, text=""),
        make_turn(seq=2, text="more content"),
    ]
    vecs = embed_turns(turns)
    assert len(vecs) == 3
    assert vecs[0] is not None
    assert vecs[1] is None
    assert vecs[2] is not None
    assert len(vecs[0]) == 384
    assert len(vecs[2]) == 384


def test_embed_turns_order_preserved():
    from claude_history.embed import embed_turns
    texts = ["apple", "banana", "cherry"]
    turns = [make_turn(seq=i, text=t) for i, t in enumerate(texts)]
    vecs = embed_turns(turns)
    assert len(vecs) == 3
    # All non-None and all 384-dim
    assert all(v is not None and len(v) == 384 for v in vecs)
    # Embeddings for different texts should differ
    assert vecs[0] != vecs[1]


def test_embed_turns_batch_larger_than_one():
    from claude_history.embed import embed_turns, EMBED_BATCH_SIZE
    # Generate more turns than EMBED_BATCH_SIZE to exercise chunking
    n = EMBED_BATCH_SIZE + 2
    turns = [make_turn(seq=i, text=f"sentence number {i}") for i in range(n)]
    vecs = embed_turns(turns)
    assert len(vecs) == n
    assert all(v is not None and len(v) == 384 for v in vecs)
