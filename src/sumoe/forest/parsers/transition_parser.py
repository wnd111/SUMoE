from __future__ import annotations

from ..transition.beam import beam_parse
from ..transition.dataset import DependencySentence
from ..transition.model import StackTransformerParser
from ..types import CandidateTree, TokenSpan
from .base import ParsedSentence


class TransitionParserAdapter:
    name = "transition"

    def __init__(self, model: StackTransformerParser, n_best: int = 5, beam_size: int = 16) -> None:
        self.model = model
        self.n_best = n_best
        self.beam_size = beam_size

    def make_sentence(
        self,
        words: tuple[str, ...],
        upos: tuple[str, ...],
        start: int,
    ) -> ParsedSentence:
        cursor = start
        spans: list[TokenSpan] = []
        for word in words:
            spans.append(TokenSpan(word, cursor, cursor + len(word)))
            cursor += len(word) + 1
        return ParsedSentence(
            text=" ".join(words),
            start=start,
            end=cursor - 1,
            tokens=tuple(spans),
            upos=upos,
        )

    def segment_document(self, text: str) -> tuple[ParsedSentence, ...]:
        raise RuntimeError("transition parser uses Stanza sentence segmentation")

    def parse_sentence(self, sentence: ParsedSentence) -> tuple[CandidateTree, ...]:
        length = len(sentence.tokens)
        placeholder_heads = (-1,) + (0,) * (length - 1)
        placeholder_labels = ("root",) + ("dep",) * (length - 1)
        row = DependencySentence(
            words=tuple(token.text for token in sentence.tokens),
            upos=sentence.upos,
            heads=placeholder_heads,
            labels=placeholder_labels,
        )
        return beam_parse(self.model, row, beam_size=self.beam_size, n_best=self.n_best)

