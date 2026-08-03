from __future__ import annotations

import math
from typing import Protocol, cast

import torch

from ..calibration import tree_signature
from ..types import CandidateTree
from .actions import Action, ParserState, apply_action, valid_actions
from .dataset import DependencySentence


class StateScorer(Protocol):
    @property
    def labels(self) -> tuple[str, ...]:
        raise NotImplementedError

    def score_state(
        self, sentence: DependencySentence, state: ParserState
    ) -> dict[Action, float]:
        raise NotImplementedError


def beam_parse(
    model: StateScorer,
    sentence: DependencySentence,
    beam_size: int = 16,
    n_best: int = 5,
) -> tuple[CandidateTree, ...]:
    if beam_size < 1 or n_best < 1:
        raise ValueError("beam_size and n_best must be positive")
    labels = tuple(model.labels)
    beam: list[tuple[ParserState, float]] = [(ParserState.initial(len(sentence.words)), 0.0)]
    completed: list[tuple[ParserState, float]] = []
    for _ in range(2 * len(sentence.words) + 1):
        expanded: list[tuple[ParserState, float]] = []
        for state, score in beam:
            if state.complete:
                completed.append((state, score))
                continue
            valid = valid_actions(state, labels)
            if not valid:
                continue
            state_scores = model.score_state(sentence, state)
            logits = torch.tensor([state_scores[action] for action in valid], dtype=torch.float64)
            log_probabilities = torch.log_softmax(logits, dim=0).tolist()
            for action, log_probability in zip(valid, log_probabilities, strict=True):
                expanded.append((apply_action(state, action), score + float(log_probability)))
        if not expanded:
            break
        expanded.sort(
            key=lambda item: (
                -item[1],
                tuple(action.key() for action in item[0].actions),
            )
        )
        unique: dict[tuple[object, ...], tuple[ParserState, float]] = {}
        for item in expanded:
            state = item[0]
            key = (state.stack, state.buffer, state.heads, state.labels)
            unique.setdefault(key, item)
        beam = list(unique.values())[:beam_size]
    completed.extend(item for item in beam if item[0].complete)
    trees: list[CandidateTree] = []
    seen: set[tuple[tuple[int, str], ...]] = set()
    for state, score in sorted(completed, key=lambda item: -item[1]):
        if any(head is None for head in state.heads):
            raise RuntimeError("complete beam state contains an unresolved head")
        heads = tuple(cast(int, head) for head in state.heads)
        dependency_labels = tuple(str(label) for label in state.labels)
        tree = CandidateTree(
            parser="transition",
            heads=heads,
            labels=dependency_labels,
            raw_score=score / max(len(state.actions), 1),
        )
        signature = tree_signature(tree)
        if signature in seen or not math.isfinite(tree.raw_score):
            continue
        seen.add(signature)
        trees.append(tree)
        if len(trees) == n_best:
            break
    if not trees:
        raise RuntimeError("beam search did not produce a complete dependency tree")
    return tuple(trees)
