from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import LBFGS  # type: ignore[attr-defined]

from .types import CandidateTree


@dataclass(frozen=True)
class ParserCalibration:
    temperature: float
    prior: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("parser temperature must be positive and finite")
        if not math.isfinite(self.prior) or self.prior <= 0:
            raise ValueError("parser prior must be positive and finite")


def tree_signature(tree: CandidateTree) -> tuple[tuple[int, str], ...]:
    return tuple(zip(tree.heads, tree.labels, strict=True))


def fit_temperature(scores: torch.Tensor, correct: torch.Tensor) -> float:
    if scores.ndim != 1 or correct.shape != scores.shape:
        raise ValueError("scores and correct must be equal one-dimensional tensors")
    scores = scores.detach().to(dtype=torch.float64)
    correct = correct.detach().to(dtype=torch.float64)
    if torch.any((correct < 0) | (correct > 1)):
        raise ValueError("correct values must be within [0, 1]")
    raw = torch.nn.Parameter(torch.tensor(0.5413248546, dtype=torch.float64))
    optimizer = LBFGS([raw], max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = F.softplus(raw) + 1e-6
        loss = F.binary_cross_entropy_with_logits(scores / temperature, correct)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float((F.softplus(raw.detach()) + 1e-6).item())


def merge_and_select(
    candidates: Sequence[CandidateTree],
    calibrations: Mapping[str, ParserCalibration],
    top_k: int,
) -> tuple[CandidateTree, ...]:
    if not candidates:
        raise ValueError("at least one candidate tree is required")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    grouped: dict[str, list[CandidateTree]] = defaultdict(list)
    for candidate in candidates:
        if candidate.parser not in calibrations:
            raise ValueError(f"missing calibration for parser {candidate.parser}")
        grouped[candidate.parser].append(candidate)

    masses: dict[tuple[tuple[int, str], ...], float] = defaultdict(float)
    representatives: dict[tuple[tuple[int, str], ...], CandidateTree] = {}
    sources: dict[tuple[tuple[int, str], ...], set[str]] = defaultdict(set)
    for parser, parser_candidates in grouped.items():
        calibration = calibrations[parser]
        logits = torch.tensor(
            [candidate.raw_score for candidate in parser_candidates], dtype=torch.float64
        ) / calibration.temperature
        probabilities = torch.softmax(logits, dim=0).tolist()
        for candidate, probability in zip(parser_candidates, probabilities, strict=True):
            signature = tree_signature(candidate)
            masses[signature] += calibration.prior * float(probability)
            representatives.setdefault(signature, candidate)
            sources[signature].add(parser)

    ordered = sorted(masses, key=lambda item: (-masses[item], item))[:top_k]
    total = sum(masses[signature] for signature in ordered)
    selected: list[CandidateTree] = []
    for signature in ordered:
        representative = representatives[signature]
        parser_name = representative.parser if len(sources[signature]) == 1 else "merged"
        selected.append(
            representative.with_posterior(masses[signature] / total, parser=parser_name)
        )
    return tuple(selected)
