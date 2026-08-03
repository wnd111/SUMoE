from __future__ import annotations

from .scrolls import NormalizedExample
from .tasks import TASKS

SYSTEM_MESSAGE = (
    "You are a precise assistant for long-document understanding and reasoning."
)


def render_prompt(example: NormalizedExample) -> list[dict[str, str]]:
    instruction = TASKS[example.task].instruction
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": f"{instruction}\n\nSOURCE:\n{example.source}",
        },
        {"role": "assistant", "content": example.target},
    ]


def render_source_text(example: NormalizedExample) -> tuple[str, int, int]:
    """Return the deterministic training prefix and exact source character span."""
    instruction = TASKS[example.task].instruction
    prefix = (
        f"SYSTEM: {SYSTEM_MESSAGE}\n\n"
        f"USER: {instruction}\n\nSOURCE:\n"
    )
    suffix = "\n\nASSISTANT:\n"
    text = prefix + example.source + suffix
    return text, len(prefix), len(prefix) + len(example.source)
