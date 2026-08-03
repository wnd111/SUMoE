from __future__ import annotations

from sumoe.evaluation.evaluator import evaluate_predictions


def test_quality_report_counts_invalid_predictions() -> None:
    report = evaluate_predictions("quality", ["A", "long invalid answer"], [["A"], ["B"]])
    assert report.sample_count == 2
    assert report.invalid_prediction_count == 1
    assert report.metrics["exact_match"] == 50.0


def test_quality_report_evaluates_scrolls_answer_text() -> None:
    report = evaluate_predictions(
        "quality",
        ["The supported answer.", "wrong"],
        [["The supported answer."], ["Another answer."]],
    )
    assert report.invalid_prediction_count == 0
    assert report.metrics["exact_match"] == 50.0
