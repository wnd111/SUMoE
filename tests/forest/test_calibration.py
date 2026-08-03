from __future__ import annotations

import math

import pytest
import torch

from sumoe.forest.calibration import (
    ParserCalibration,
    fit_temperature,
    merge_and_select,
    tree_signature,
)
from sumoe.forest.types import CandidateTree


def candidate(
    parser: str,
    heads: tuple[int, ...],
    labels: tuple[str, ...],
    score: float,
) -> CandidateTree:
    return CandidateTree(parser=parser, heads=heads, labels=labels, raw_score=score)


def fixed_calibrations() -> dict[str, ParserCalibration]:
    return {
        "stanza": ParserCalibration(temperature=1.0, prior=0.4),
        "spacy": ParserCalibration(temperature=1.0, prior=0.3),
        "transition": ParserCalibration(temperature=1.0, prior=0.3),
    }


def test_duplicate_trees_merge_probability_before_top_k() -> None:
    trees = [
        candidate("stanza", (-1, 0), ("root", "obj"), -0.2),
        candidate("spacy", (-1, 0), ("root", "obj"), -0.4),
        candidate("transition", (1, -1), ("nsubj", "root"), -0.1),
    ]
    selected = merge_and_select(trees, fixed_calibrations(), top_k=5)
    assert len(selected) == 2
    assert sum(tree.posterior for tree in selected) == pytest.approx(1.0)
    assert selected[0].parser == "merged"
    assert selected[0].posterior == pytest.approx(0.7)
    assert selected[1].posterior == pytest.approx(0.3)


def test_tree_signature_contains_heads_and_labels() -> None:
    tree = candidate("stanza", (-1, 0), ("root", "obj"), -0.2)
    assert tree_signature(tree) == ((-1, "root"), (0, "obj"))


def test_temperature_fit_returns_positive_finite_value() -> None:
    temperature = fit_temperature(
        torch.tensor([3.0, 2.0, -2.0, -3.0]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )
    assert temperature > 0.0
    assert math.isfinite(temperature)

