from __future__ import annotations

import logging
import os
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

log = logging.getLogger(__name__)
console = Console()


def _db_name() -> str:
    return os.environ.get("CH_DB", "claude_history")


def _open_conn(db_name: str):
    from dbclients.clients.postgres import get_postgres_client

    return get_postgres_client(client_type="context_db", dbname=db_name)()


@click.command("ch-ingest")
@click.option("--embed", "backfill_embeddings", is_flag=True, default=False,
              help="Backfill all missing embeddings after ingesting.")
@click.option("-v", "--verbose", is_flag=True)
def ingest(backfill_embeddings: bool, verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    from claude_history.ingest import ingest_all

    transcripts_dir = Path(
        os.environ.get("TRANSCRIPTS_DIR", str(Path.home() / ".claude" / "projects"))
    )
    ingest_all(transcripts_dir=transcripts_dir, backfill_embeddings=backfill_embeddings)


@click.command("ch-search")
@click.argument("query")
@click.option("--fts", "mode", flag_value="fts", help="Full-text search.")
@click.option("--semantic", "mode", flag_value="semantic", help="Semantic/vector search.")
@click.option("--hybrid", "mode", flag_value="hybrid", default=True,
              help="Hybrid RRF (default).")
@click.option("--limit", default=10, show_default=True)
@click.option("--offset", default=0, show_default=True)
@click.option("--subagents", is_flag=True, default=False,
              help="Include subagent sessions.")
def search(query: str, mode: str, limit: int, offset: int, subagents: bool) -> None:
    logging.basicConfig(level=logging.WARNING)
    from claude_history import search as search_mod

    if mode == "fts":
        results = search_mod.fts(query, include_subagents=subagents, limit=limit, offset=offset)
    elif mode == "semantic":
        results = search_mod.semantic(query, include_subagents=subagents, limit=limit, offset=offset)
    else:
        results = search_mod.hybrid(query, include_subagents=subagents, limit=limit, offset=offset)

    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Project", style="cyan", no_wrap=True)
    table.add_column("Title", style="dim")
    table.add_column("Role", width=9)
    table.add_column("Content")
    table.add_column("Date", width=12)

    for row in results:
        ts = str(row.get("ts", ""))[:10]
        table.add_row(
            row.get("project_name", ""),
            (row.get("title") or "")[:40],
            row.get("role", ""),
            row.get("content_text", "")[:200],
            ts,
        )

    console.print(table)


@click.command("ch-resolve-titles")
@click.option("-v", "--verbose", is_flag=True)
def resolve_titles(verbose: bool) -> None:
    """Generate titles for all sessions that still have title IS NULL."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    from claude_history import db
    from claude_history import titler

    db_name = _db_name()
    try:
        with _open_conn(db_name) as conn:
            session_ids = db.sessions_without_titles(conn)
    except Exception as exc:
        console.print(f"[red]DB unavailable: {exc}[/red]")
        return

    if not session_ids:
        console.print("All sessions already have titles.")
        return

    console.print(f"Resolving titles for {len(session_ids)} sessions...")
    resolved = 0

    for session_id in session_ids:
        try:
            with _open_conn(db_name) as conn:
                turns_raw = db.get_turns_for_session(conn, str(session_id), limit=500)
        except Exception as exc:
            log.warning("DB error fetching turns for %s: %s", session_id, exc)
            continue

        from claude_history.models import Turn

        turns = [
            Turn(
                session_id=str(session_id),
                seq=r["seq"],
                role=r["role"],
                content_raw=r["content_raw"],
                content_text=r["content_text"] or "",
                ts=r["ts"],
            )
            for r in turns_raw
        ]

        try:
            title = titler.generate_title(turns)
        except Exception as exc:
            log.warning("title generation failed for %s: %s", session_id, exc)
            continue

        if not title:
            continue

        try:
            with _open_conn(db_name) as conn:
                db.update_title(conn, str(session_id), title)
            resolved += 1
            log.info("titled %s: %s", session_id, title)
        except Exception as exc:
            log.warning("DB write failed for %s: %s", session_id, exc)

    console.print(f"Done. Resolved {resolved}/{len(session_ids)} titles.")
