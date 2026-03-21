from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext

import click
from rich.console import Console
from rich.table import Table

from claude_history import db
from claude_history import search as search_mod

_IS_TTY = sys.stdout.isatty()
_console = Console()


def _db_name() -> str:
    return os.environ.get("CH_DB", "claude_history")


def _open_conn(db_name: str):
    from dbclients.clients.postgres import get_postgres_client

    return get_postgres_client(client_type="context_db", dbname=db_name)()


def _fmt_dt(dt) -> str:
    if dt is None:
        return ""
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d %H:%M")
    return str(dt)[:16]


def _fmt_duration(started_at, ended_at) -> str:
    if not started_at or not ended_at:
        return ""
    try:
        minutes = int((ended_at - started_at).total_seconds() / 60)
    except TypeError:
        return ""
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}m"


@click.group()
def main():
    """Browse and search Claude Code session history."""


@main.command("sessions")
@click.option("--project", "-p", default=None, help="Filter by project name.")
@click.option("--limit", "-n", default=20, type=int, help="Max sessions to show.")
@click.option("--subagents", is_flag=True, help="Include subagent sessions.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of table.")
def sessions_cmd(project, limit, subagents, as_json):
    """List recent sessions."""
    with _open_conn(_db_name()) as conn:
        rows = db.list_sessions(
            conn,
            project=project,
            include_subagents=subagents,
            limit=limit,
        )

    if as_json or not _IS_TTY:
        click.echo(json.dumps(rows, default=str))
        return

    if not rows:
        _console.print("[yellow]No sessions found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("#", width=3, justify="right")
    table.add_column("Project", style="cyan", width=16)
    table.add_column("Title / First Message", width=42)
    table.add_column("Turns", justify="right", width=5)
    table.add_column("Dur", justify="right", width=7)
    table.add_column("Started", width=16)

    for i, r in enumerate(rows, 1):
        label = r.get("title") or (r.get("first_message_snippet") or "")
        label = label.replace("\n", " ")
        if len(label) > 42:
            label = label[:39] + "..."
        table.add_row(
            str(i),
            (r.get("project_name") or "")[:16],
            label,
            str(r.get("turn_count") or 0),
            _fmt_duration(r.get("started_at"), r.get("ended_at")),
            _fmt_dt(r.get("started_at")),
        )

    _console.print(table)


@main.command("turns")
@click.argument("session_id")
@click.option("--limit", "-n", default=50, type=int)
@click.option("--offset", "-o", default=0, type=int)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of formatted output.")
def turns_cmd(session_id, limit, offset, as_json):
    """Show turns for SESSION_ID (full UUID or unique prefix)."""
    with _open_conn(_db_name()) as conn:
        if len(session_id) < 36:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT session_id FROM cc_sessions"
                    " WHERE session_id::text LIKE %s LIMIT 1",
                    (session_id + "%",),
                )
                row = cur.fetchone()
            if not row:
                _console.print(
                    f"[red]No session matching '{session_id}'[/red]",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            session_id = str(row[0])
        rows = db.get_turns_for_session(conn, session_id, offset=offset, limit=limit)

    if as_json or not _IS_TTY:
        click.echo(json.dumps(rows, default=str))
        return

    if not rows:
        _console.print("[yellow]No turns found.[/yellow]")
        return

    for r in rows:
        role = r.get("role", "")
        text = (r.get("content_text") or "").strip()
        if not text:
            continue
        color = {"user": "green", "assistant": "blue"}.get(role, "dim")
        _console.print(f"[bold {color}]{role}[/bold {color}]  [dim]{_fmt_dt(r.get('ts'))}[/dim]")
        lines = text.splitlines()
        for line in lines[:8]:
            _console.print(f"  {line}")
        if len(lines) > 8:
            _console.print(f"  [dim]…({len(lines) - 8} more lines)[/dim]")
        _console.print()


@main.command("search")
@click.argument("query")
@click.option(
    "--mode", "-m",
    type=click.Choice(["fts", "hybrid", "semantic"]),
    default="hybrid",
    show_default=True,
)
@click.option("--limit", "-n", default=10, type=int)
@click.option("--subagents", is_flag=True)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of table.")
def search_cmd(query, mode, limit, subagents, as_json):
    """Search turns by keyword or semantic similarity."""
    fn = {
        "fts": search_mod.fts,
        "hybrid": search_mod.hybrid,
        "semantic": search_mod.semantic,
    }[mode]
    ctx = _console.status("Searching...") if _IS_TTY else nullcontext()
    with ctx:
        rows = fn(query, include_subagents=subagents, limit=limit)

    if as_json or not _IS_TTY:
        click.echo(json.dumps(rows, default=str))
        return

    if not rows:
        _console.print("[yellow]No results.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Project", style="cyan", width=14)
    table.add_column("Title", width=32)
    table.add_column("Role", width=9)
    table.add_column("Snippet", width=54)
    table.add_column("Score", justify="right", width=6)

    for r in rows:
        title = (r.get("title") or "")[:32]
        snippet = (r.get("content_text") or "").replace("\n", " ")[:54]
        score = r.get("score")
        table.add_row(
            (r.get("project_name") or "")[:14],
            title,
            r.get("role", ""),
            snippet,
            f"{score:.3f}" if score else "",
        )

    _console.print(table)
