from __future__ import annotations

from collections.abc import Sequence

from .actions import Action, ActionKind, ParserState, apply_action


def _all_children_attached(token: int, state: ParserState, gold_heads: Sequence[int]) -> bool:
    return all(
        gold_head != token or state.heads[dependent] is not None
        for dependent, gold_head in enumerate(gold_heads)
    )


def oracle_actions(heads: Sequence[int], labels: Sequence[str]) -> tuple[Action, ...]:
    if len(heads) != len(labels) or not heads:
        raise ValueError("gold heads and labels must have equal nonzero length")
    state = ParserState.initial(len(heads))
    while not state.complete:
        action: Action | None = None
        if len(state.stack) >= 2:
            left = state.stack[-2]
            right = state.stack[-1]
            if (
                left != -1
                and heads[left] == right
                and _all_children_attached(left, state, heads)
            ):
                action = Action(ActionKind.LEFT_ARC, labels[left])
            elif (
                right != -1
                and heads[right] == left
                and _all_children_attached(right, state, heads)
            ):
                action = Action(ActionKind.RIGHT_ARC, labels[right])
        if action is None and state.buffer:
            action = Action(ActionKind.SHIFT)
        if action is None:
            raise ValueError("gold dependency tree is non-projective for arc-standard parsing")
        state = apply_action(state, action)
    return state.actions

