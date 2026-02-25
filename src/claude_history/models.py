from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    pass


@dataclass
class Session:
    session_id: UUID
    project_path: str
    project_name: str
    git_branch: str | None
    cc_version: str | None
    started_at: datetime | None
    ended_at: datetime | None
    turn_count: int = 0


@dataclass
class Turn:
    session_id: UUID
    seq: int
    role: str  # 'user' | 'assistant'
    content: str
    ts: datetime | None
    embedding: list[float] | None = None
