from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from .tasks import TASKS


@dataclass(frozen=True)
class NormalizedExample:
    example_id: str
    source_id: str
    task: str
    split: str
    family: str
    source: str
    target: str
    references: tuple[str, ...]
    subset: str = "standard"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _strings(value: object, field: str) -> tuple[str, ...]:
    values: tuple[str, ...]
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(str(item) for item in value)
    else:
        raise ValueError(f"SCROLLS {field} must be a string or sequence of strings")
    cleaned = tuple(item.strip() for item in values if item.strip())
    if not cleaned:
        raise ValueError(f"SCROLLS {field} contains no non-empty text")
    return cleaned


def normalize_scrolls_example(
    task: str, split: str, row: Mapping[str, object]
) -> tuple[NormalizedExample, ...]:
    """Normalize one canonical ``tau/scrolls`` row without changing its text."""
    if task not in TASKS:
        raise ValueError(f"unsupported SCROLLS task: {task}")
    source = str(row.get("input", "")).strip()
    if not source:
        raise ValueError("SCROLLS input must be non-empty")
    references = _strings(row.get("output"), "output")
    raw_id = row.get("id", row.get("pid"))
    if raw_id is None or not str(raw_id).strip():
        raise ValueError("SCROLLS row must contain id or pid")
    source_id = f"{task}::{split}::{raw_id}"
    subset = "quality_hard" if task == "quality" and (
        bool(row.get("is_hard", False)) or "hard" in split.lower()
    ) else "standard"

    def make(target: str, suffix: str) -> NormalizedExample:
        return NormalizedExample(
            example_id=f"{source_id}::{suffix}",
            source_id=source_id,
            task=task,
            split=split,
            family=TASKS[task].family,
            source=source,
            target=target,
            references=references,
            subset=subset,
        )

    if split == "train":
        return tuple(make(reference, f"ref{index}") for index, reference in enumerate(references))
    return (make(references[0], "eval"),)
