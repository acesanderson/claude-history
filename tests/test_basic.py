def test_import():
    import claude_history
    assert claude_history is not None


def test_parser_import():
    from claude_history import parser
    assert hasattr(parser, "iter_sessions")
    assert hasattr(parser, "parse_session_file")
