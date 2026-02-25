from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class Session:
    session_id: str
    project_path: str
    project_name: str
    git_branch: str | None
    cc_version: str | None
    started_at: datetime | None
    ended_at: datetime | None
    is_subagent: bool
    title: str | None = None


@dataclass
class Turn:
    session_id: str
    seq: int
    role: str  # 'user' | 'assistant' | 'system'
    content_raw: list | dict
    content_text: str  # extracted text blocks; '' for tool-only / local_command turns
    ts: datetime | None
    # Not set by parser.py; populated after DB upsert / embedding
    embedding: list[float] | None = None
