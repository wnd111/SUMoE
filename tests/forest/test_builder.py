from __future__ import annotations

import pytest

from sumoe.forest.builder import build_document_forest
from sumoe.forest.calibration import ParserCalibration
from sumoe.forest.parsers.base import ParsedSentence
from sumoe.forest.types import CandidateTree, TokenSpan


class FakeParser:
    def __init__(self, name: str, candidates: tuple[CandidateTree, ...]) -> None:
        self.name = name
        self._candidates = candidates

    def segment_document(self, text: str) -> tuple[ParsedSentence, ...]:
        return (
            ParsedSentence(
                text=text,
                start=0,
                end=len(text),
                tokens=(TokenSpan("Alpha", 0, 5), TokenSpan("works", 6, 11)),
                upos=("NOUN", "VERB"),
            ),
        )

    def parse_sentence(self, sentence: ParsedSentence) -> tuple[CandidateTree, ...]:
        return self._candidates


class OffsetTokenizer:
    name_or_path = "fake-llama"

    def __call__(self, text: str, **kwargs):
        return {"offset_mapping": [(0, 5), (6, 11)], "input_ids": [10, 11]}


def tree(parser: str, reverse: bool, label: str, score: float) -> CandidateTree:
    return CandidateTree(
        parser=parser,
        heads=((1, -1) if not reverse else (-1, 0)),
        labels=((label, "root") if not reverse else ("root", label)),
        raw_score=score,
    )


def calibrations() -> dict[str, ParserCalibration]:
    return {
        "stanza": ParserCalibration(1.0, 0.34),
        "spacy": ParserCalibration(1.0, 0.33),
        "transition": ParserCalibration(1.0, 0.33),
        "bad": ParserCalibration(1.0, 0.1),
    }


def test_builder_pools_three_sources_and_keeps_five_unique_trees() -> None:
    parsers = [
        FakeParser("stanza", (tree("stanza", False, "nsubj", -0.1),)),
        FakeParser(
            "spacy",
            (
                tree("spacy", True, "obj", -0.1),
                tree("spacy", False, "csubj", -0.2),
                tree("spacy", True, "obl", -0.3),
            ),
        ),
        FakeParser(
            "transition",
            (
                tree("transition", False, "advcl", -0.1),
                tree("transition", True, "xcomp", -0.2),
                tree("transition", False, "acl", -0.3),
            ),
        ),
    ]
    result = build_document_forest(
        {"id": "qasper::train::1", "source": "Alpha works"},
        parsers=parsers,
        calibration=calibrations(),
        tokenizer=OffsetTokenizer(),
        top_k=5,
    )
    assert len(result.forest.sentences[0].candidates) == 5
    assert sum(c.posterior for c in result.forest.sentences[0].candidates) == pytest.approx(1.0)
    assert result.aligned.example_id == "qasper::train::1"


def test_unalignable_parser_tokens_are_recorded_and_not_silently_used() -> None:
    invalid = CandidateTree(
        parser="bad",
        heads=(1, -1, 1),
        labels=("dep", "root", "dep"),
        raw_score=-0.1,
    )
    result = build_document_forest(
        {"id": "qasper::train::1", "source": "Alpha works"},
        parsers=[
            FakeParser("stanza", (tree("stanza", False, "nsubj", -0.1),)),
            FakeParser("bad", (invalid,)),
        ],
        calibration=calibrations(),
        tokenizer=OffsetTokenizer(),
        top_k=5,
    )
    assert result.diagnostics[0].code == "TOKEN_ALIGNMENT_FAILED"
    assert result.diagnostics[0].example_id == "qasper::train::1"

