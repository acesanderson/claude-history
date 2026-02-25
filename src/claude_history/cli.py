from __future__ import annotations

import logging
import os

import click
from rich.console import Console
from rich.table import Table

from claude_history import ingest, search

log = logging.getLogger(__name__)
console = Console()


def _get_conn():
    import psycopg2
    dbname = os.environ.get("CH_DB", "claude_history")
    return psycopg2.connect(dbname=dbname)


@click.command("ch-ingest")
@click.option("--embed", "with_embeddings", is_flag=True, default=False,
              help="Backfill embeddings after ingesting.")
@click.option("--no-subagents", is_flag=True, default=False,
              help="Skip subagent transcript files.")
@click.option("-v", "--verbose", is_flag=True)
def ingest_cmd(with_embeddings: bool, no_subagents: bool, verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    conn = _get_conn()
    try:
        ingest.run(
            conn,
            with_embeddings=with_embeddings,
            include_subagents=not no_subagents,
        )
    finally:
        conn.close()


@click.command("ch-search")
@click.argument("query")
@click.option("--fts", "mode", flag_value="fts", help="Full-text search (default).")
@click.option("--semantic", "mode", flag_value="semantic", help="Semantic/vector search.")
@click.option("--hybrid", "mode", flag_value="hybrid", default=True,
              help="Hybrid RRF (default).")
@click.option("--limit", default=10, show_default=True)
def search_cmd(query: str, mode: str, limit: int) -> None:
    logging.basicConfig(level=logging.WARNING)
    conn = _get_conn()
    try:
        if mode == "fts":
            results = search.fts(conn, query, limit=limit)
        elif mode == "semantic":
            results = search.semantic(conn, query, limit=limit)
        else:
            results = search.hybrid(conn, query, limit=limit)
    finally:
        conn.close()

    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Project", style="cyan", no_wrap=True)
    table.add_column("Role", width=9)
    table.add_column("Content")
    table.add_column("Date", width=12)

    for row in results:
        ts = str(row.get("ts", ""))[:10]
        table.add_row(
            row.get("project_name", ""),
            row.get("role", ""),
            row.get("content", "")[:200],
            ts,
        )

    console.print(table)
