from __future__ import annotations

import pytest

from sumoe.data.scrolls import normalize_scrolls_example
from sumoe.data.tasks import TASKS


@pytest.mark.parametrize(
    "task",
    [
        "gov_report",
        "summ_screen_fd",
        "qmsum",
        "qasper",
        "narrative_qa",
        "quality",
        "contract_nli",
    ],
)
def test_every_paper_task_has_instruction_and_family(task: str) -> None:
    spec = TASKS[task]
    assert spec.family in {"summarization", "qa", "reasoning"}
    assert spec.instruction.endswith(".")


def test_training_references_expand_but_validation_references_stay_grouped() -> None:
    row = {"id": "doc-1", "input": "long source", "output": ["a", "b"]}
    train = normalize_scrolls_example("qasper", "train", row)
    validation = normalize_scrolls_example("qasper", "validation", row)
    assert [item.target for item in train] == ["a", "b"]
    assert len(validation) == 1
    assert validation[0].references == ("a", "b")
    assert train[0].source_id == "qasper::train::doc-1"


def test_paper_tasks_select_generation_or_classification_heads() -> None:
    assert TASKS["qasper"].prediction_head == "generation"
    assert TASKS["quality"].prediction_head == "classification"
    assert TASKS["quality"].class_labels == ("A", "B", "C", "D")
    assert TASKS["contract_nli"].class_labels == (
        "Entailment",
        "Contradiction",
        "Not mentioned",
    )
