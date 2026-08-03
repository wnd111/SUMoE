from __future__ import annotations

from typing import Any

from ..types import CandidateTree, TokenSpan
from .base import ParsedSentence


class StanzaParserAdapter:
    name = "stanza"

    def __init__(self, model: str = "en") -> None:
        try:
            import stanza
        except ImportError as error:
            raise RuntimeError("stanza==1.9.2 is required for forest construction") from error
        self.pipeline = stanza.Pipeline(
            lang=model,
            processors="tokenize,pos,lemma,depparse",
            tokenize_no_ssplit=False,
            use_gpu=False,
            verbose=False,
        )

    def segment_document(self, text: str) -> tuple[ParsedSentence, ...]:
        document = self.pipeline(text)
        sentences: list[ParsedSentence] = []
        for stanza_sentence in document.sentences:
            spans: list[TokenSpan] = []
            upos: list[str] = []
            for token in stanza_sentence.tokens:
                word = token.words[0]
                spans.append(TokenSpan(word.text, int(token.start_char), int(token.end_char)))
                upos.append(str(word.upos or "X"))
            sentences.append(
                ParsedSentence(
                    text=text[spans[0].start : spans[-1].end],
                    start=spans[0].start,
                    end=spans[-1].end,
                    tokens=tuple(spans),
                    upos=tuple(upos),
                )
            )
        return tuple(sentences)

    def parse_sentence(self, sentence: ParsedSentence) -> tuple[CandidateTree, ...]:
        document: Any = self.pipeline(sentence.text)
        if len(document.sentences) != 1:
            raise ValueError("Stanza changed the fixed sentence boundary")
        words = document.sentences[0].words
        if len(words) != len(sentence.tokens):
            raise ValueError("Stanza tokenization changed during dependency parsing")
        heads = tuple(-1 if int(word.head) == 0 else int(word.head) - 1 for word in words)
        labels = tuple(str(word.deprel) for word in words)
        return (
            CandidateTree(
                parser=self.name,
                heads=heads,
                labels=labels,
                raw_score=0.0,
            ),
        )

