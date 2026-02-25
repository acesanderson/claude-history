from __future__ import annotations

import json
import logging
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from uuid import UUID

from claude_history.models import Session, Turn

log = logging.getLogger(__name__)

_TRANSCRIPTS_ROOT = Path.home() / ".claude" / "projects"

_SKIP_TYPES = frozenset({"progress", "system", "file-history-snapshot"})


def _extract_text(content: str | list) -> str | None:
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = " ".join(parts).strip()
        return text or None
    return None


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def iter_sessions(
    root: Path | None = None,
    include_subagents: bool = True,
) -> Generator[tuple[Session, list[Turn]], None, None]:
    root = root or _TRANSCRIPTS_ROOT
    pattern = "**/*.jsonl" if include_subagents else "*/*.jsonl"

    for path in sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime):
        try:
            result = parse_session_file(path)
        except Exception:
            log.debug("skipping %s", path, exc_info=True)
            continue
        if result is not None:
            yield result


def parse_session_file(path: Path) -> tuple[Session, list[Turn]] | None:
    turns: list[Turn] = []
    session_id_str: str | None = None
    project_path_str: str | None = None
    cc_version: str | None = None
    started_at: datetime | None = None

    # Derive project_name and project_path from the file path.
    # Structure: ~/.claude/projects/<encoded-path>/<session-id>.jsonl
    # or:        ~/.claude/projects/<encoded-path>/<session-id>/subagents/agent-<id>.jsonl
    parts = path.relative_to(_TRANSCRIPTS_ROOT).parts
    encoded = parts[0]
    project_path_str = encoded.replace("-", "/").lstrip("/")
    project_name = project_path_str.rsplit("/", 1)[-1]

    seq = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = record.get("type")
            if rtype in _SKIP_TYPES:
                continue

            if rtype == "system":
                session_id_str = record.get("sessionId") or session_id_str
                cc_version = record.get("cliVersion") or cc_version
                continue

            if rtype not in ("user", "assistant"):
                continue

            msg = record.get("message", {})
            content = msg.get("content", "")
            text = _extract_text(content)
            if not text:
                continue

            ts = _parse_ts(record.get("timestamp"))
            if started_at is None:
                started_at = ts

            turns.append(
                Turn(
                    session_id=UUID(int=0),  # patched below
                    seq=seq,
                    role=rtype,
                    content=text,
                    ts=ts,
                )
            )
            seq += 1

    if not turns:
        return None

    # Fall back to filename stem as session ID if not found in records.
    raw_id = session_id_str or path.stem
    try:
        sid = UUID(raw_id)
    except ValueError:
        import hashlib
        sid = UUID(hashlib.md5(raw_id.encode()).hexdigest())  # noqa: S324

    for turn in turns:
        turn.session_id = sid

    session = Session(
        session_id=sid,
        project_path=project_path_str,
        project_name=project_name,
        git_branch=None,
        cc_version=cc_version,
        started_at=started_at,
        ended_at=turns[-1].ts if turns else None,
        turn_count=len(turns),
    )
    return session, turns
