from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tiktoken
from jinja2 import Template

if TYPE_CHECKING:
    from claude_history.models import Turn

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = Template(
    "You are given a conversation transcript from a Claude Code session.\n"
    "Generate a concise title (5–10 words) capturing the main task or topic.\n"
    "Respond with only the title, no punctuation at the end.\n\n"
    "<transcript>\n"
    "{{ transcript }}\n"
    "</transcript>"
)


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
    enc = tiktoken.get_encoding("cl100k_base")

    nonempty = [t for t in turns if t.content_text]
    if not nonempty:
        return ""

    head = nonempty[:30]
    tail = [t for t in nonempty[-10:] if t not in head]
    selected = head + tail

    if not selected:
        return ""

    per_turn_budget = budget // len(selected)

    parts = []
    for t in selected:
        tokens = enc.encode(t.content_text)
        if len(tokens) > per_turn_budget:
            text = enc.decode(tokens[:per_turn_budget])
        else:
            text = t.content_text
        parts.append(f"{t.role.upper()}: {text}")

    return "\n\n".join(parts)


def generate_title(turns: list[Turn]) -> str | None:
    from conduit.remote import RemoteModelSync

    transcript = _build_title_transcript(turns)
    if not transcript:
        return None

    prompt = _PROMPT_TEMPLATE.render(transcript=transcript)
    model = RemoteModelSync(model="gpt-oss:latest")
    result = model.complete(prompt)
    title = result.content.strip()
    return title or None
