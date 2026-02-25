# claude-history-project

Local store for Claude Code session transcripts with full-text and semantic search.

## Problem

Raw JSONL session files at `~/.claude/projects/` are:
- ~93% `progress` noise records
- Bloated with inline tool results (can be 10s of KB per record)
- Not searchable across sessions
- Tied to file modification time for ordering

## Solution

Ingest clean conversation turns into Postgres with:
- FTS via generated `tsvector` column + GIN index
- Vector search via pgvector (`sentence-transformers/all-MiniLM-L6-v2`, 384 dims)
- Hybrid search via RRF (reciprocal rank fusion)
- LLM-generated session titles via ConduitRemote (`gpt-oss:latest`)

## Runtime Environment

This is a personal single-user tool. Two external services are involved, both on the local network:

**Postgres** — reachable only over WireGuard VPN. `dbclients` handles host detection automatically based on network context; no manual host config is needed. If the VPN is down, Postgres is unreachable — this is expected and handled with a one-line warning everywhere.

**Headwater ML server** — Brian's personal ML server running sentence-transformers and `gpt-oss:latest` (via Ollama or equivalent). Accessed through the `conduit` and `headwater_client` libraries. If the server is down, embeddings and title generation are skipped gracefully. Neither service being down should ever crash an entry point.

## Dependencies & Local Installs

`dbclients` and `conduit` are **not on PyPI** — they are local editable installs from `~/Brian_Code/`. Both are declared in `pyproject.toml` via `[tool.uv.sources]`:

```toml
[tool.uv.sources]
dbclients = { path = "../../Brian_Code/dbclients-project", editable = true }
conduit   = { path = "../../Brian_Code/conduit-project",   editable = true }
```

Full dependency list:

```toml
dependencies = [
    "psycopg2-binary",
    "pgvector",
    "tiktoken",
    "jinja2",
    "conduit",
    "dbclients",
    "rich",
    "click",
]
```

`ch-resolve-titles` entry point also needs adding to `[project.scripts]`:

```toml
[project.scripts]
ch-ingest          = "claude_history.cli:ingest"
ch-search          = "claude_history.cli:search"
ch-resolve-titles  = "claude_history.cli:resolve_titles"
```

### Key import paths

```python
# Postgres — returns a context manager; use as:
#   with get_postgres_client(client_type="context_db", dbname="claude_history")() as conn:
from dbclients.clients.postgres import get_postgres_client

# Embeddings — validate_model() makes a server round-trip; call once, not per batch
from conduit.embeddings.generate_embeddings import generate_embeddings, quick_embedding

# Title LLM — result.content is the response string
from conduit.remote import RemoteModelSync
```

## Design Rationale

Brief notes on non-obvious decisions made during spec design:

**No `DatabaseManager` / connection pool**: every entry point (hook, CLI, skill) is a short-lived subprocess. There are no concurrent coroutines racing to initialize a shared pool within a single process. The `context_db` context manager is the right primitive here.

**Two content columns (`content_raw JSONB` + `content_text TEXT`)**: all turns are stored at full fidelity (including tool calls and tool results) in `content_raw`. `content_text` extracts only the human-readable text blocks for FTS and embeddings — keeping search quality high without polluting the index with JSON field names or tool arguments.

**System records are operational events, not prompt injections**: CLAUDE.md and skill content is passed to the API at inference time and is never written to the JSONL transcript. System records only have three subtypes: `turn_duration` (timing metadata), `compact_boundary` (session compaction marker), and `local_command` (slash commands like `/copy`, `/skills`). Only `local_command` is worth storing — the others are pure metadata. No deduplication logic needed.

**`session_id` from filename, not record fields**: JSONL records use camelCase `sessionId` internally. The `SessionEnd` hook provides `transcript_path` whose stem is the canonical UUID. Filename wins to avoid any divergence between record fields and the hook payload.

**Proportional trim + first/last split for titles**: the beginning of a session establishes the problem; the end shows resolution. Middle turns are typically iterative tool-call/response cycles that add noise to a title. Budget is `cl100k_base` tokens as a conservative proxy across models — `gpt-oss:latest` may have different tokenisation, so the 25k ceiling is intentionally conservative.

## Schema

Run in psql against the `claude_history` database. Handled automatically by `db.py:create_schema()`.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE cc_sessions (
    session_id   UUID PRIMARY KEY,
    project_path TEXT NOT NULL,
    project_name TEXT NOT NULL,   -- last component of project_path
    git_branch   TEXT,
    cc_version   TEXT,
    started_at   TIMESTAMPTZ,
    ended_at     TIMESTAMPTZ,
    is_subagent  BOOLEAN NOT NULL DEFAULT FALSE,
    title        TEXT              -- LLM-generated; NULL until resolved
);

CREATE TABLE cc_turns (
    id           BIGSERIAL PRIMARY KEY,
    session_id   UUID NOT NULL REFERENCES cc_sessions(session_id),
    seq          INT NOT NULL,
    role         TEXT NOT NULL,       -- 'user' | 'assistant' | 'system' (system = local_command only)
    content_raw  JSONB NOT NULL,      -- full structured content blocks
    content_text TEXT NOT NULL,       -- extracted text blocks only; may be '' for tool-only turns
    content_tsv  TSVECTOR GENERATED ALWAYS AS (
                     to_tsvector('english', content_text)
                 ) STORED,
    embedding    vector(384),         -- all-MiniLM-L6-v2; NULL if content_text is empty
    ts           TIMESTAMPTZ,
    UNIQUE (session_id, seq)
);

CREATE INDEX idx_cc_turns_tsv     ON cc_turns USING GIN(content_tsv);
CREATE INDEX idx_cc_turns_session ON cc_turns(session_id);
CREATE INDEX idx_cc_sessions_proj ON cc_sessions(project_name);

-- Add after initial bulk load when >10k turns:
-- CREATE INDEX idx_cc_turns_hnsw ON cc_turns
--     USING hnsw(embedding vector_cosine_ops)
--     WITH (m=16, ef_construction=64);
```

`turn_count` is not stored — computed dynamically via `COUNT(*)` join in `list_sessions`. No stale state, negligible performance impact at this scale.

## Ingestion Trigger

### Primary: Claude Code `SessionEnd` Hook

Claude Code fires `SessionEnd` on session exit, providing on stdin:

```json
{
  "session_id": "...",
  "transcript_path": "/Users/.../.claude/projects/.../abc123.jsonl",
  "cwd": "/Users/...",
  "hook_event_name": "SessionEnd"
}
```

`session_id` is always taken from the **filename** of `transcript_path`, which is the canonical identifier.

The hook runs: parse → upsert → embed → title. Both embed and title are attempted. If Postgres is unavailable, a single one-line warning is printed to stderr and the hook exits cleanly (exit 0). If the ML server is unavailable, embed and title are each skipped with a one-line warning; the session is stored without embeddings or title for later resolution.

Hook configuration (add to `~/.claude/settings.json`):

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "uv run --directory ~/vibe/claude-history-project python -m claude_history.hook",
            "timeout": 60,
            "async": true
          }
        ]
      }
    ]
  }
}
```

`async: true` ensures session exit is never delayed by ingest. Errors go to stderr silently in the background.

### Secondary: Backfill CLI

`ch-ingest` scans all JSONL files in `~/.claude/projects/` (including `subagents/` subdirs) and upserts any sessions not yet in the database. Idempotent. Used for initial historical backfill and re-syncing after downtime.

```bash
ch-ingest            # parse + upsert all sessions
ch-ingest --embed    # also backfill all missing embeddings (server-side cursor)
```

## Parsing Rules

Stream line-by-line. Never `json.load()` a full file. `session_id` is always the JSONL filename stem (UUID), not a field inside records.

Every record carries top-level fields `timestamp` (ISO 8601, e.g. `"2026-02-20T22:08:38.029Z"`), `cwd`, `gitBranch`, and `version`. Session metadata is extracted from the **first record** in the file regardless of type.

| Record type  | Subtype            | Action |
|--------------|--------------------|--------|
| `progress`   | any                | Skip entirely |
| `file-history-snapshot` | any   | Skip entirely |
| `system`     | `turn_duration`    | Skip entirely |
| `system`     | `compact_boundary` | Skip entirely |
| `system`     | `local_command`    | Store as `role='system'`; `content_raw` = `{"subtype": "local_command", "content": record["content"]}`; `content_text` = `''` |
| `user`       | —                  | Store; `content_raw` = `message["content"]` as JSONB; `content_text` = all `{type: "text"}` blocks joined with `\n\n` |
| `assistant`  | —                  | Store; `content_raw` = `message["content"]` as JSONB; `content_text` = all `{type: "text"}` blocks joined with `\n\n` |

**`seq`** is the zero-indexed count of stored turns within the session, in parse order. Skipped records (`progress`, `file-history-snapshot`, skipped `system` subtypes) do not increment `seq`. The first stored turn is `seq=0`. This value is assigned by the parser and must be stable across re-parses of the same file — it is the key used for idempotent upserts.

**`ts`** is parsed from the record's `timestamp` field (ISO 8601 string) via `datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))`.

**`content_text` may be an empty string** for tool-only turns and `local_command` turns. These are stored but never embedded (embedding stays NULL).

### Session Metadata

Extracted from top-level fields of the **first record** in the file:
- `project_path` ← `record["cwd"]`
- `git_branch` ← `record["gitBranch"]`
- `cc_version` ← `record["version"]`
- `started_at` ← `ts` of the first stored turn
- `ended_at` ← `ts` of the last stored turn

### Subagent Detection

Subagent transcripts live at `<session-id>/subagents/<subagent-id>.jsonl`. Detected by presence of `/subagents/` in the path. Stored with `is_subagent = TRUE`. All queries accept an `include_subagents` flag (default `False`).

## Title Generation

Each session gets a single LLM-generated title stored in `cc_sessions.title`.

**Model**: `RemoteModelSync(model="gpt-oss:latest")` via conduit.

**Trim strategy**: Use the first 30 and last 10 turns with non-empty `content_text`, distributed within a 25k token budget (tiktoken `cl100k_base` encoding). Each selected turn is truncated proportionally to fit the budget.

```python
def _build_title_transcript(turns: list[Turn], budget: int = 25_000) -> str:
    """
    Select first 30 + last 10 non-empty turns and fit within token budget.

    The first/last split captures problem statement (head) and resolution (tail)
    while skipping iterative middle turns that add noise to title generation.

    NOTE: first_n and last_n are tunable. Reduce them if you encounter silent
    context window overflows with the backing LLM (e.g. gpt-oss:latest via Ollama).
    The 25k budget already applies a conservative multiplier vs. the nominal
    context window, but model-specific limits may require further adjustment.
    """
```

**Timing**: Runs at end of ingest. Title is never regenerated once set.

**Resolution script**: `ch-resolve-titles` iterates all sessions with `title IS NULL` and attempts generation. Run manually when the ML server is back online.

**Prompt template** (Jinja2):

```
You are given a conversation transcript from a Claude Code session.
Generate a concise title (5–10 words) capturing the main task or topic.
Respond with only the title, no punctuation at the end.

<transcript>
{{ transcript }}
</transcript>
```

## Search Modes

All queries accept `include_subagents BOOL` (default `FALSE`) and `LIMIT / OFFSET` for pagination.

**FTS** — keyword, no API call:
```sql
SELECT s.session_id, s.project_name, s.title,
       t.role, t.content_text AS content, t.ts
FROM cc_turns t
JOIN cc_sessions s USING (session_id)
WHERE t.content_tsv @@ plainto_tsquery('english', $1)
  AND ($2 OR NOT s.is_subagent)
ORDER BY t.ts DESC
LIMIT $3 OFFSET $4;
```

**Vector** — semantic, embeds query string first via `quick_embedding`:
```sql
SELECT s.session_id, s.project_name, s.title,
       t.role, t.content_text AS content,
       1 - (t.embedding <=> $1::vector) AS score
FROM cc_turns t
JOIN cc_sessions s USING (session_id)
WHERE t.embedding IS NOT NULL
  AND ($2 OR NOT s.is_subagent)
ORDER BY t.embedding <=> $1::vector
LIMIT $3 OFFSET $4;
```

**Hybrid** — RRF fusion of FTS rank + vector rank:
- Pull `min((offset + limit) * 10, 200)` candidates from each side independently.
- Score each candidate via `1 / (60 + rank)`.
- Sum scores per turn `id`, sort descending, apply `LIMIT / OFFSET`.
- Turns that appear in only one result set receive a score from that side only (standard RRF behaviour).
- If no embeddings exist yet, falls back to FTS-only automatically.

## Skill Interface (`skill.py`)

`skill.py` is a **pure library** — no argparse, no `__main__`, no side effects on import. It exposes three functions that the Claude Code skill's own runner script calls directly. All functions are synchronous.

```python
def search(
    query: str,
    mode: Literal["fts", "semantic", "hybrid"] = "hybrid",
    limit: int = 20,
    offset: int = 0,
    include_subagents: bool = False,
) -> list[dict]: ...
# Returns: [{session_id, project_name, title, role, content_text, content_raw, ts, score}]

def list_sessions(
    project: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    include_subagents: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]: ...
# Returns: [{session_id, project_name, title, started_at, ended_at,
#            turn_count, is_subagent, first_message_snippet}]
# turn_count: computed via COUNT(*) join
# first_message_snippet: first 300 chars of content_text of first user turn; None if absent

def get_session_turns(
    session_id: str,
    offset: int = 0,
    limit: int = 50,
) -> list[dict]: ...
# Returns: [{id, seq, role, content_text, content_raw, ts}]
# Paginate with offset to handle large sessions
```

The Claude Code skill (a separate `~/.claude/skills/claude-history/` directory created later) owns:
- `SKILL.md` — trigger phrases, instructions for which function to call, pagination guidance
- A runner script that imports from `claude_history.skill`, handles argparse, and serializes results to JSON on stdout

`list_sessions` uses a LATERAL join for `first_message_snippet`:
```sql
SELECT s.*,
       COUNT(t.id)          AS turn_count,
       snippet.content_text AS first_message_snippet
FROM cc_sessions s
LEFT JOIN cc_turns t USING (session_id)
LEFT JOIN LATERAL (
    SELECT LEFT(content_text, 300) AS content_text
    FROM cc_turns
    WHERE session_id = s.session_id
      AND role = 'user'
      AND content_text <> ''
    ORDER BY seq
    LIMIT 1
) snippet ON true
WHERE ($1 IS NULL OR s.project_name = $1)
  AND ($2 IS NULL OR s.started_at >= $2)
  AND ($3 IS NULL OR s.started_at <= $3)
  AND ($4 OR NOT s.is_subagent)
GROUP BY s.session_id, snippet.content_text
ORDER BY s.started_at DESC
LIMIT $5 OFFSET $6;
```

## Modules

| Module        | Role |
|---------------|------|
| `models.py`   | `Session` and `Turn` dataclasses (see shapes below) |
| `parser.py`   | JSONL → `(Session, list[Turn])`; streaming; assigns `seq`; extracts metadata from first record |
| `db.py`       | `create_schema()`, upserts, search queries, `list_sessions` |
| `embed.py`    | Chunked batch embeddings via conduit (`all-MiniLM-L6-v2`, 384d); see Embeddings Batching |
| `titler.py`   | ConduitRemote title generation; tiktoken trim to 25k budget; first 30 + last 10 turns |
| `ingest.py`   | `ingest_session(path)` for hook; `ingest_all(transcripts_dir)` for CLI backfill; both run parse → upsert → embed → title |
| `search.py`   | FTS / vector / hybrid search with RRF fusion |
| `skill.py`    | Pure library: `search`, `list_sessions`, `get_session_turns`; no CLI, no side effects on import |
| `hook.py`     | `SessionEnd` entry point; reads stdin JSON, calls `ingest_session()`; one-line warnings on failure |
| `cli.py`      | `ch-ingest` calls `ingest_all()`; `ch-ingest --embed` also backfills embeddings; `ch-search`; `ch-resolve-titles` |

### `models.py` dataclass shapes

```python
@dataclass
class Session:
    session_id:   str
    project_path: str
    project_name: str        # last component of project_path
    git_branch:   str | None
    cc_version:   str | None
    started_at:   datetime | None
    ended_at:     datetime | None
    is_subagent:  bool
    title:        str | None = None   # populated later by titler.py

@dataclass
class Turn:
    session_id:   str
    seq:          int          # zero-indexed, stable across re-parses
    role:         str          # 'user' | 'assistant' | 'system'
    content_raw:  list | dict  # full JSONB payload; serialised to JSON on upsert
    content_text: str          # extracted plain text; '' for tool-only turns
    ts:           datetime | None
    # Fields below are NOT set by parser.py; populated after DB upsert / embedding
    embedding:    list[float] | None = None
```

`Turn` does not carry a DB `id` — embedding write-back uses `(session_id, seq)` as the natural key. `parser.py` never sets `embedding`; `embed.py` writes directly to the database.

## Concurrency Model

Every entry point is a short-lived subprocess with a single synchronous psycopg2 connection via `get_postgres_client(client_type="context_db", dbname="claude_history")`. No shared pool or async coordination needed.

Multiple `SessionEnd` hooks firing simultaneously operate on distinct session IDs — no row-level contention. The one race (`ch-resolve-titles` and a hook both attempting to title the same session) is harmless: the second `UPDATE ... WHERE title IS NULL` matches 0 rows.

**All upserts use explicit conflict handling:**

```sql
-- Sessions: update mutable fields on re-ingest
INSERT INTO cc_sessions (session_id, project_path, project_name, git_branch,
                         cc_version, started_at, ended_at, is_subagent)
VALUES (...)
ON CONFLICT (session_id) DO UPDATE
  SET ended_at   = EXCLUDED.ended_at,
      git_branch = EXCLUDED.git_branch;

-- Turns: fully idempotent
INSERT INTO cc_turns (session_id, seq, role, content_raw, content_text, ts)
VALUES (...)
ON CONFLICT (session_id, seq) DO NOTHING;

-- Title: set once, never overwrite
UPDATE cc_sessions
   SET title = $1
 WHERE session_id = $2
   AND title IS NULL;

-- Embedding write-back: keyed on natural key, not BIGSERIAL
UPDATE cc_turns
   SET embedding = $1
 WHERE session_id = $2
   AND seq = $3;
```

## Embeddings Batching

`conduit.embeddings.generate_embeddings` has no internal chunking — one HTTP request per call. `validate_model()` also fires a server round-trip on every call. We own all batching logic.

### `embed.py` interface

```python
EMBED_BATCH_SIZE = 64  # override via CH_EMBED_BATCH_SIZE

def embed_turns(turns: list[Turn]) -> list[list[float]]:
    """
    Embed turns in chunks of EMBED_BATCH_SIZE.
    Skips turns with empty content_text (returns None for those positions).
    Validates model once before the loop.
    Returns embeddings in input order; None for skipped turns.
    """
```

### Call sites

| Site | Behaviour |
|------|-----------|
| `ingest.py` (hook) | `embed_turns(session_turns)` once per session. Typically one HTTP call. |
| `ch-ingest --embed` | Server-side cursor over all `cc_turns WHERE embedding IS NULL`; `embed_turns` per page. |
| `search.py` | `quick_embedding(query)` directly — single call, no chunking. |

### Server-side cursor for backfill

```python
with get_postgres_client(client_type="context_db", dbname=db_name)() as conn:
    with conn.cursor("embed_backfill") as cur:   # named → server-side
        cur.itersize = EMBED_BATCH_SIZE
        cur.execute(
            "SELECT session_id, seq, content_text FROM cc_turns"
            " WHERE embedding IS NULL AND content_text <> '' ORDER BY id"
        )
        batch = cur.fetchmany(EMBED_BATCH_SIZE)
        while batch:
            # embed batch, write back via (session_id, seq), fetch next page
            batch = cur.fetchmany(EMBED_BATCH_SIZE)
```

One Postgres fetch ≈ one headwater request. Memory stays flat regardless of backlog size.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CH_EMBED_BATCH_SIZE` | `64` | Turns per HTTP request to headwater |
| `TRANSCRIPTS_DIR` | `~/.claude/projects/` | Override transcript scan root |
| `CH_DB` | `claude_history` | Postgres database name |

Postgres: `get_postgres_client(client_type="context_db", dbname="claude_history")` from `dbclients`.
Embeddings: `generate_embeddings(ids, documents, model="sentence-transformers/all-MiniLM-L6-v2")` from `conduit`.
Title LLM: `RemoteModelSync(model="gpt-oss:latest")` from `conduit`.

## Open Decisions

- **Tool result content in `content_raw`**: `tool_result` blocks (responses from tools) are stored in full. Very large tool outputs (e.g. Read on a 5000-line file) will be stored verbatim. A future `--trim-tool-output N` flag could cap these.
- **HNSW index**: defer until >10k turns; sequential scan is fine at small scale.
- **tiktoken encoding**: `cl100k_base` used as conservative proxy for token counting across models.
