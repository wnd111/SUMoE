from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..types import CandidateTree, TokenSpan


@dataclass(frozen=True)
class ParsedSentence:
    text: str
    start: int
    end: int
    tokens: tuple[TokenSpan, ...]
    upos: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.end <= self.start or len(self.tokens) != len(self.upos):
            raise ValueError("parsed sentence offsets and token fields are inconsistent")


class DependencyParser(Protocol):
    name: str

    def segment_document(self, text: str) -> tuple[ParsedSentence, ...]:
        raise NotImplementedError

    def parse_sentence(self, sentence: ParsedSentence) -> tuple[CandidateTree, ...]:
        raise NotImplementedError

