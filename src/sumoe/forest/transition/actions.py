from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import cast


class ActionKind(str, Enum):
    SHIFT = "shift"
    LEFT_ARC = "left_arc"
    RIGHT_ARC = "right_arc"


@dataclass(frozen=True, order=True)
class Action:
    kind: ActionKind
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind == ActionKind.SHIFT and self.label:
            raise ValueError("SHIFT must not have a dependency label")
        if self.kind != ActionKind.SHIFT and not self.label:
            raise ValueError("arc actions require a dependency label")

    def key(self) -> str:
        if self.kind == ActionKind.SHIFT:
            return cast(str, self.kind.value)
        return f"{self.kind.value}:{self.label}"

    @classmethod
    def from_key(cls, value: str) -> Action:
        kind, separator, label = value.partition(":")
        return cls(ActionKind(kind), label if separator else "")


class ActionVocabulary:
    def __init__(self, labels: Iterable[str]) -> None:
        unique = tuple(dict.fromkeys(str(label) for label in labels))
        if "root" not in unique:
            unique = ("root",) + unique
        self.labels = unique
        self.actions = (
            Action(ActionKind.SHIFT),
            *(Action(ActionKind.LEFT_ARC, label) for label in unique),
            *(Action(ActionKind.RIGHT_ARC, label) for label in unique),
        )
        self._ids = {action: index for index, action in enumerate(self.actions)}

    def __len__(self) -> int:
        return len(self.actions)

    def id(self, action: Action) -> int:
        return self._ids[action]

    def action(self, index: int) -> Action:
        return self.actions[index]

    @property
    def shift_id(self) -> int:
        return 0

    def to_dict(self) -> dict[str, object]:
        return {"labels": list(self.labels)}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ActionVocabulary:
        return cls(data["labels"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class ParserState:
    length: int
    stack: tuple[int, ...]
    buffer: tuple[int, ...]
    heads: tuple[int | None, ...]
    labels: tuple[str | None, ...]
    actions: tuple[Action, ...] = ()

    @classmethod
    def initial(cls, length: int) -> ParserState:
        if length < 1:
            raise ValueError("sentence length must be positive")
        return cls(
            length=length,
            stack=(-1,),
            buffer=tuple(range(length)),
            heads=(None,) * length,
            labels=(None,) * length,
        )

    @property
    def complete(self) -> bool:
        return not self.buffer and self.stack == (-1,) and all(
            head is not None for head in self.heads
        )


def valid_actions(state: ParserState, labels: Sequence[str]) -> tuple[Action, ...]:
    actions: list[Action] = []
    if state.buffer:
        actions.append(Action(ActionKind.SHIFT))
    if len(state.stack) >= 2:
        left_dependent = state.stack[-2]
        right_dependent = state.stack[-1]
        if left_dependent != -1 and state.heads[left_dependent] is None:
            actions.extend(
                Action(ActionKind.LEFT_ARC, label) for label in labels if label != "root"
            )
        if right_dependent != -1 and state.heads[right_dependent] is None:
            right_head = state.stack[-2]
            if right_head == -1:
                if not state.buffer and len(state.stack) == 2 and "root" in labels:
                    actions.append(Action(ActionKind.RIGHT_ARC, "root"))
            else:
                actions.extend(
                    Action(ActionKind.RIGHT_ARC, label) for label in labels if label != "root"
                )
    return tuple(actions)


def apply_action(state: ParserState, action: Action) -> ParserState:
    if action.kind == ActionKind.SHIFT:
        if not state.buffer:
            raise ValueError("SHIFT is invalid with an empty buffer")
        return replace(
            state,
            stack=state.stack + (state.buffer[0],),
            buffer=state.buffer[1:],
            actions=state.actions + (action,),
        )
    if len(state.stack) < 2:
        raise ValueError("arc action requires two stack items")
    heads = list(state.heads)
    labels = list(state.labels)
    if action.kind == ActionKind.LEFT_ARC:
        dependent = state.stack[-2]
        head = state.stack[-1]
        if dependent == -1 or heads[dependent] is not None:
            raise ValueError("LEFT_ARC has an invalid dependent")
        heads[dependent] = head
        labels[dependent] = action.label
        stack = state.stack[:-2] + (state.stack[-1],)
    else:
        dependent = state.stack[-1]
        head = state.stack[-2]
        if dependent == -1 or heads[dependent] is not None:
            raise ValueError("RIGHT_ARC has an invalid dependent")
        heads[dependent] = head
        labels[dependent] = action.label
        stack = state.stack[:-1]
    return replace(
        state,
        stack=stack,
        heads=tuple(heads),
        labels=tuple(labels),
        actions=state.actions + (action,),
    )


def execute_actions(
    length: int, actions: Sequence[Action]
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    state = ParserState.initial(length)
    for action in actions:
        state = apply_action(state, action)
    if not state.complete:
        raise ValueError("action sequence does not produce a complete dependency tree")
    if any(head is None for head in state.heads) or any(
        label is None for label in state.labels
    ):
        raise RuntimeError("complete parser state contains an unresolved arc")
    return (
        tuple(cast(int, head) for head in state.heads),
        tuple(str(label) for label in state.labels),
    )
