from __future__ import annotations

from sumoe.forest.calibration import tree_signature
from sumoe.forest.transition.actions import Action, ActionKind
from sumoe.forest.transition.beam import beam_parse
from sumoe.forest.transition.dataset import DependencySentence


class DeterministicScorer:
    labels = ("root", "nsubj", "obj")

    def score_state(self, sentence, state):
        scores = {Action(ActionKind.SHIFT): 0.0}
        for label in self.labels:
            scores[Action(ActionKind.LEFT_ARC, label)] = -1.0
            scores[Action(ActionKind.RIGHT_ARC, label)] = -1.0
        if not state.buffer and len(state.stack) == 2:
            dependent = state.stack[-1]
            label = "root" if state.stack[-2] == -1 else ("nsubj" if dependent == 0 else "obj")
            scores[Action(ActionKind.RIGHT_ARC, label)] = 2.0
        return scores


def test_beam_returns_unique_complete_trees_in_score_order() -> None:
    sentence = DependencySentence(
        words=("A", "works"),
        upos=("NOUN", "VERB"),
        heads=(1, -1),
        labels=("nsubj", "root"),
    )
    trees = beam_parse(DeterministicScorer(), sentence, beam_size=16, n_best=5)
    assert 1 <= len(trees) <= 5
    assert len({tree_signature(tree) for tree in trees}) == len(trees)
    assert [tree.raw_score for tree in trees] == sorted(
        [tree.raw_score for tree in trees], reverse=True
    )

