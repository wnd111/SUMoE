from __future__ import annotations

import pytest

from sumoe.evaluation.metrics import (
    contract_nli_accuracy,
    qa_f1,
    quality_exact_match,
    rouge_metrics,
)


def test_qa_f1_uses_squad_normalization_and_best_reference() -> None:
    assert qa_f1("The Eiffel Tower.", ["tower", "Eiffel Tower"]) == pytest.approx(1.0)


def test_contract_nli_rejects_unrecognized_generation() -> None:
    assert contract_nli_accuracy("uncertain", "entailment") == 0.0


def test_quality_requires_one_exact_option_label() -> None:
    assert quality_exact_match("(B)", 1) == 1.0
    assert quality_exact_match("B because it is correct", 1) == 0.0


def test_quality_accepts_scrolls_answer_text_references() -> None:
    assert quality_exact_match("The supported answer.", "The supported answer.") == 1.0
    assert quality_exact_match("B", "The supported answer.") == 0.0


def test_rouge_identical_text_is_one_hundred() -> None:
    scores = rouge_metrics(["alpha beta"], [["alpha beta"]])
    assert scores == {"rouge1": 100.0, "rouge2": 100.0, "rougeL": 100.0}
