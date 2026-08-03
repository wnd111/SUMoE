from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    family: str
    instruction: str
    metric: str
    prediction_head: str = "generation"
    class_labels: tuple[str, ...] = ()


TASKS: dict[str, TaskSpec] = {
    "gov_report": TaskSpec(
        "gov_report",
        "summarization",
        "Write a concise summary of the following government report.",
        "rouge",
    ),
    "summ_screen_fd": TaskSpec(
        "summ_screen_fd",
        "summarization",
        "Write a concise summary of the following television screenplay.",
        "rouge",
    ),
    "qmsum": TaskSpec(
        "qmsum",
        "qa",
        "Answer the query using only the meeting transcript.",
        "rouge",
    ),
    "qasper": TaskSpec(
        "qasper",
        "qa",
        "Answer the question using only the scientific paper.",
        "f1",
    ),
    "narrative_qa": TaskSpec(
        "narrative_qa",
        "qa",
        "Answer the question using only the narrative.",
        "f1",
    ),
    "quality": TaskSpec(
        "quality",
        "reasoning",
        "Select the correct answer option and return only its option label.",
        "exact_match",
        prediction_head="classification",
        class_labels=("A", "B", "C", "D"),
    ),
    "contract_nli": TaskSpec(
        "contract_nli",
        "reasoning",
        "Classify the hypothesis as Entailment, Contradiction, or Not mentioned.",
        "accuracy",
        prediction_head="classification",
        class_labels=("Entailment", "Contradiction", "Not mentioned"),
    ),
}


def quality_choices(source: str) -> tuple[str, str, str, str]:
    """Extract the first consecutive (A)-(D) option block from a SCROLLS input."""
    matches = list(re.finditer(r"(?m)^[ \t]*\(([A-Da-d])\)[ \t]+(.+?)[ \t]*$", source))
    for start in range(max(0, len(matches) - 3)):
        group = matches[start : start + 4]
        if tuple(match.group(1).upper() for match in group) == ("A", "B", "C", "D"):
            return (
                group[0].group(2).strip(),
                group[1].group(2).strip(),
                group[2].group(2).strip(),
                group[3].group(2).strip(),
            )
    raise ValueError("QuALITY source does not contain one consecutive (A)-(D) option block")


def classification_label_index(task: str, target: str, source: str | None = None) -> int:
    spec = TASKS[task]
    if spec.prediction_head != "classification":
        raise ValueError(f"task does not use a classification head: {task}")
    if task == "quality":
        match = re.fullmatch(r"\s*\(?([A-Da-d])\)?[\s.]*", target)
        if match is not None:
            normalized = match.group(1).upper()
        else:
            if source is None:
                raise ValueError("QuALITY answer text requires its source option block")
            choices = quality_choices(source)
            normalized_target = " ".join(target.split())
            matching = [
                index
                for index, choice in enumerate(choices)
                if " ".join(choice.split()) == normalized_target
            ]
            if len(matching) != 1:
                raise ValueError(f"QuALITY target does not uniquely match an option: {target!r}")
            return matching[0]
    else:
        compact = re.sub(r"[\s_.-]+", "", target.strip().lower())
        aliases = {
            "entailment": "Entailment",
            "contradiction": "Contradiction",
            "notmentioned": "Not mentioned",
        }
        if compact not in aliases:
            raise ValueError(f"unrecognized ContractNLI class label: {target!r}")
        normalized = aliases[compact]
    return spec.class_labels.index(normalized)


def classification_label_text(task: str, index: int, source: str | None = None) -> str:
    spec = TASKS[task]
    if spec.prediction_head != "classification":
        raise ValueError(f"task does not use a classification head: {task}")
    if not 0 <= index < len(spec.class_labels):
        raise ValueError("classification index lies outside the task label set")
    if task == "quality" and source is not None:
        return quality_choices(source)[index]
    return spec.class_labels[index]
