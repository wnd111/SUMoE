from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

from rouge_score import rouge_scorer


def rouge_metrics(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> dict[str, float]:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("predictions and references must have the same non-zero length")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for prediction, sample_references in zip(predictions, references, strict=True):
        if not sample_references:
            raise ValueError("every prediction requires at least one reference")
        candidates = [scorer.score(reference, prediction) for reference in sample_references]
        for metric in totals:
            totals[metric] += max(candidate[metric].fmeasure for candidate in candidates)
    return {metric: 100.0 * total / len(predictions) for metric, total in totals.items()}


def _normalize_squad(text: str) -> list[str]:
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return without_articles.split()


def qa_f1(prediction: str, references: Sequence[str]) -> float:
    predicted_tokens = _normalize_squad(prediction)
    best = 0.0
    for reference in references:
        reference_tokens = _normalize_squad(reference)
        if not predicted_tokens or not reference_tokens:
            score = float(predicted_tokens == reference_tokens)
        else:
            common = Counter(predicted_tokens) & Counter(reference_tokens)
            overlap = sum(common.values())
            if overlap == 0:
                score = 0.0
            else:
                precision = overlap / len(predicted_tokens)
                recall = overlap / len(reference_tokens)
                score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def extract_quality_label(prediction: str) -> str | None:
    match = re.fullmatch(r"\s*\(?([A-Da-d])\)?[\s.]*", prediction)
    return match.group(1).upper() if match else None


def quality_exact_match(prediction: str, answer_index: int | str) -> float:
    if isinstance(answer_index, int):
        if not 0 <= answer_index <= 3:
            raise ValueError("QuALITY answer index must be zero-based in [0, 3]")
        expected = chr(ord("A") + answer_index)
    else:
        expected = answer_index.strip()
        if expected.upper() not in {"A", "B", "C", "D"}:
            return float(prediction.strip() == expected)
        expected = expected.upper()
    return float(extract_quality_label(prediction) == expected)


_CONTRACT_LABELS = {
    "entailment": "entailment",
    "contradiction": "contradiction",
    "notmentioned": "notmentioned",
    "not mentioned": "notmentioned",
    "not_mentioned": "notmentioned",
}


def extract_contract_label(prediction: str) -> str | None:
    normalized = prediction.strip().lower().rstrip(".").strip()
    return _CONTRACT_LABELS.get(normalized)


def contract_nli_accuracy(prediction: str, label: str) -> float:
    expected = _CONTRACT_LABELS.get(label.strip().lower())
    if expected is None:
        raise ValueError(f"unrecognized ContractNLI reference label: {label}")
    return float(extract_contract_label(prediction) == expected)
