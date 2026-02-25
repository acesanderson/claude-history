"""SessionEnd hook entry point.

Invoked by Claude Code on session exit:
  uv run --directory ~/vibe/claude-history-project python -m claude_history.hook

Stdin: JSON with at minimum {"transcript_path": "..."}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        print(f"claude-history hook: failed to read stdin: {exc}", file=sys.stderr)
        sys.exit(0)

    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        sys.exit(0)

    path = Path(transcript_path)
    if not path.exists():
        print(f"claude-history hook: transcript not found: {path}", file=sys.stderr)
        sys.exit(0)

    try:
        from claude_history.ingest import ingest_session

        ingest_session(path)
    except Exception as exc:
        print(f"claude-history hook: {exc}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
