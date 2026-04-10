# Claude History

Claude History is a local storage and search engine for Claude Code session transcripts. It ingests JSONL transcripts into a Postgres database to provide full-text search, semantic vector search, and hybrid ranking.

## Quick Start

### Prerequisites

*   Python 3.12+
*   PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension
*   A running instance of the `conduit` and `dbclients` internal packages

### Installation

```bash
# Clone and install the package
pip install .

# Set the database connection string if different from default
export CH_DB="claude_history"
```

### Initial Ingest

Import existing transcripts from the default Claude Code directory:

```bash
ch-ingest --embed
```

## Core Functionality

### Hybrid Search
Find specific interactions across all past sessions using Reciprocal Rank Fusion (RRF) which combines keyword matching with semantic meaning.

```bash
ch-history search "how did I implement the postgres connection pool" --mode hybrid
```

### Session Visualization
Browse a high-level overview of recent work, including auto-generated titles and turn counts.

```bash
ch-history sessions --limit 10
```

### Automation Hook
Automatically ingest every Claude Code session upon exit. Add this to your shell configuration or Claude Code settings:

```bash
# Invoked by Claude Code on session exit
uv run python -m claude_history.hook
```

## CLI Reference

### ch-history
The primary interface for browsing and searching the transcript database.

| Command | Description | Key Options |
| :--- | :--- | :--- |
| `sessions` | List recent sessions with durations and titles. | `--project`, `--limit`, `--json` |
| `turns` | Display the full conversation for a specific session. | `SESSION_ID`, `--limit` |
| `search` | Execute search across all turns. | `--mode [fts\|semantic\|hybrid]`, `--subagents` |

### Maintenance Tools
*   `ch-ingest`: Scans the filesystem for new `.jsonl` files. Use `--embed` to generate vector embeddings for new content.
*   `ch-resolve-titles`: Uses an LLM to generate descriptive titles for sessions that are currently untitled.

## Configuration

Environment variables control database connectivity and file discovery.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `CH_DB` | Postgres database name. | `claude_history` |
| `TRANSCRIPTS_DIR` | Root directory for Claude Code JSONL files. | `~/.claude/projects` |
| `CH_EMBED_BATCH_SIZE` | Number of turns to embed in a single batch. | `64` |

## Architecture

1.  **Parser**: Converts Claude Code's JSONL format into structured `Session` and `Turn` objects.
2.  **Database**: Postgres stores raw content, metadata, and TSVector data for FTS.
3.  **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` via the `conduit` service to populate `vector(384)` columns.
4.  **Titler**: Employs a Jinja2-templated prompt and a local LLM to summarize conversation intent into a 5–10 word title.
5.  **Search Engine**: Implements RRF to merge results from Postgres GIN indexes and IVFFlat/HNSW vector indexes.
