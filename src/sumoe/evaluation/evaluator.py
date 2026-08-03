from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from sumoe.data.tasks import TASKS

from .metrics import (
    contract_nli_accuracy,
    extract_contract_label,
    extract_quality_label,
    qa_f1,
    quality_exact_match,
    rouge_metrics,
)


@dataclass(frozen=True)
class EvaluationReport:
    task: str
    sample_count: int
    invalid_prediction_count: int
    metrics: dict[str, float]
    scrolls_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_predictions(
    task: str,
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> EvaluationReport:
    if task not in TASKS:
        raise ValueError(f"unsupported SCROLLS task: {task}")
    if len(predictions) != len(references) or not predictions:
        raise ValueError("predictions and references must have the same non-zero length")
    metric = TASKS[task].metric
    invalid = 0
    if metric == "rouge":
        values = rouge_metrics(predictions, references)
        score = mean(values.values())
    elif metric == "f1":
        scores = [
            qa_f1(prediction, sample_references)
            for prediction, sample_references in zip(predictions, references, strict=True)
        ]
        invalid = sum(not prediction.strip() for prediction in predictions)
        values = {"f1": 100.0 * mean(scores)}
        score = values["f1"]
    elif metric == "exact_match":
        label_references = all(
            extract_quality_label(sample_references[0]) is not None
            for sample_references in references
        )
        invalid = (
            sum(extract_quality_label(prediction) is None for prediction in predictions)
            if label_references
            else sum(not prediction.strip() for prediction in predictions)
        )
        scores = [
            quality_exact_match(prediction, sample_references[0])
            for prediction, sample_references in zip(predictions, references, strict=True)
        ]
        values = {"exact_match": 100.0 * mean(scores)}
        score = values["exact_match"]
    elif metric == "accuracy":
        invalid = sum(extract_contract_label(prediction) is None for prediction in predictions)
        scores = [
            contract_nli_accuracy(prediction, sample_references[0])
            for prediction, sample_references in zip(predictions, references, strict=True)
        ]
        values = {"accuracy": 100.0 * mean(scores)}
        score = values["accuracy"]
    else:
        raise RuntimeError(f"task metric is not implemented: {metric}")
    return EvaluationReport(task, len(predictions), invalid, values, score)
