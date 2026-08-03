from __future__ import annotations

import math
from typing import Any, cast

from ..types import CandidateTree
from .base import ParsedSentence


class SpacyParserAdapter:
    name = "spacy"

    def __init__(
        self,
        model: str = "en_core_web_trf",
        beam_width: int = 16,
        beam_density: float = 0.0001,
        n_best: int = 5,
    ) -> None:
        try:
            import spacy
        except ImportError as error:
            raise RuntimeError("spacy==3.8.2 is required for forest construction") from error
        self.nlp = spacy.load(model)
        self.parser = cast(Any, self.nlp.get_pipe("parser"))
        self.beam_width = beam_width
        self.beam_density = beam_density
        self.n_best = n_best

    def segment_document(self, text: str) -> tuple[ParsedSentence, ...]:
        raise RuntimeError("spaCy parser uses Stanza sentence segmentation")

    def parse_sentence(self, sentence: ParsedSentence) -> tuple[CandidateTree, ...]:
        disabled = [name for name in self.nlp.pipe_names if name == "parser"]
        with self.nlp.select_pipes(disable=disabled):
            document = self.nlp(sentence.text)
        if len(document) != len(sentence.tokens):
            raise ValueError("spaCy tokens do not align with Stanza master tokens")
        beams = self.parser.beam_parse(
            [document], beam_width=self.beam_width, beam_density=self.beam_density
        )
        parses = self.parser.moves.get_beam_parses(beams[0])
        candidates: list[CandidateTree] = []
        for probability, dependencies in parses[: self.n_best]:
            heads = list(range(len(document)))
            labels = ["dep"] * len(document)
            for head, dependent, label in dependencies:
                heads[int(dependent)] = -1 if int(head) == int(dependent) else int(head)
                labels[int(dependent)] = str(label)
            candidates.append(
                CandidateTree(
                    parser=self.name,
                    heads=tuple(heads),
                    labels=tuple(labels),
                    raw_score=math.log(max(float(probability), 1e-12)),
                )
            )
        if not candidates:
            raise RuntimeError("spaCy beam parser returned no complete dependency trees")
        return tuple(candidates)
