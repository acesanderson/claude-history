from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from uuid import UUID

from claude_history.models import Session
from claude_history.models import Turn

log = logging.getLogger(__name__)

_TRANSCRIPTS_ROOT = Path.home() / ".claude" / "projects"


def _extract_text(content: str | list) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n\n".join(parts).strip()
    return ""


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _session_id_from_stem(stem: str) -> str:
    """Return a canonical UUID string from a file stem. Non-UUID stems are hashed."""
    try:
        UUID(stem)
        return stem
    except ValueError:
        return str(UUID(hashlib.md5(stem.encode()).hexdigest()))  # noqa: S324


def iter_sessions(
    root: Path | None = None,
    include_subagents: bool = True,
) -> Generator[tuple[Session, list[Turn]], None, None]:
    root = root or _TRANSCRIPTS_ROOT
    for path in sorted(root.glob("**/*.jsonl"), key=lambda p: p.stat().st_mtime):
        if not include_subagents and "/subagents/" in str(path):
            continue
        try:
            result = parse_session_file(path)
        except Exception:
            log.debug("skipping %s", path, exc_info=True)
            continue
        if result is not None:
            yield result


def parse_session_file(path: Path) -> tuple[Session, list[Turn]] | None:
    """Parse a JSONL transcript. session_id is always taken from the filename stem."""
    is_subagent = "/subagents/" in str(path)
    session_id = _session_id_from_stem(path.stem)

    # Metadata from first record regardless of type
    project_path: str | None = None
    git_branch: str | None = None
    cc_version: str | None = None

    turns: list[Turn] = []
    seq = 0

    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract metadata from the very first parseable record
            if project_path is None:
                project_path = record.get("cwd", "")
                git_branch = record.get("gitBranch")
                cc_version = record.get("version")

            rtype = record.get("type")

            if rtype in ("progress", "file-history-snapshot"):
                continue

            if rtype == "system":
                if record.get("subtype") != "local_command":
                    continue
                ts = _parse_ts(record.get("timestamp"))
                turns.append(
                    Turn(
                        session_id=session_id,
                        seq=seq,
                        role="system",
                        content_raw={
                            "subtype": "local_command",
                            "content": record.get("content"),
                        },
                        content_text="",
                        ts=ts,
                    )
                )
                seq += 1
                continue

            if rtype not in ("user", "assistant"):
                continue

            msg = record.get("message", {})
            raw_content = msg.get("content", "")
            content_text = _extract_text(raw_content)
            ts = _parse_ts(record.get("timestamp"))

            turns.append(
                Turn(
                    session_id=session_id,
                    seq=seq,
                    role=rtype,
                    content_raw=raw_content,
                    content_text=content_text,
                    ts=ts,
                )
            )
            seq += 1

    if not turns:
        return None

    project_path = project_path or ""
    project_name = Path(project_path).name if project_path else ""

    session = Session(
        session_id=session_id,
        project_path=project_path,
        project_name=project_name,
        git_branch=git_branch,
        cc_version=cc_version,
        started_at=turns[0].ts,
        ended_at=turns[-1].ts,
        is_subagent=is_subagent,
    )
    return session, turns
