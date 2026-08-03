from __future__ import annotations

from sumoe.forest.transition.actions import execute_actions
from sumoe.forest.transition.oracle import oracle_actions


def test_oracle_actions_reconstruct_projective_gold_tree() -> None:
    heads = (1, -1, 1)
    labels = ("nsubj", "root", "obj")
    actions = oracle_actions(heads, labels)
    rebuilt_heads, rebuilt_labels = execute_actions(3, actions)
    assert rebuilt_heads == heads
    assert rebuilt_labels == labels


def test_oracle_never_left_arcs_the_root_symbol() -> None:
    actions = oracle_actions((-1, 0), ("root", "obj"))
    assert all(
        not (action.kind.value == "left_arc" and action.label == "root")
        for action in actions
    )
