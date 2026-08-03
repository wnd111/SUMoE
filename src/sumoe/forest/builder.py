from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .alignment import AlignedDocumentForest, align_document_forest
from .calibration import ParserCalibration, merge_and_select
from .parsers.base import DependencyParser
from .types import CandidateTree, DocumentForest, SentenceForest


@dataclass(frozen=True)
class BuildDiagnostic:
    code: str
    example_id: str
    sentence_index: int
    parser: str
    detail: str


@dataclass(frozen=True)
class BuildResult:
    forest: DocumentForest
    aligned: AlignedDocumentForest
    parse_seconds: float
    forest_seconds: float
    diagnostics: tuple[BuildDiagnostic, ...]


def build_document_forest(
    record: Mapping[str, Any],
    parsers: Sequence[DependencyParser],
    calibration: Mapping[str, ParserCalibration],
    tokenizer: Any,
    top_k: int = 5,
    max_length: int = 4096,
    segmenter: DependencyParser | None = None,
) -> BuildResult:
    if not parsers:
        raise ValueError("three-parser construction requires parser adapters")
    identifier = record.get("source_id", record.get("id"))
    if identifier is None:
        raise ValueError("forest record requires source_id or id")
    example_id = str(identifier)
    source = str(record["source"])
    parse_start = time.perf_counter()
    sentences = (segmenter or parsers[0]).segment_document(source)
    sentence_candidates: list[list[CandidateTree]] = [[] for _ in sentences]
    diagnostics: list[BuildDiagnostic] = []
    for sentence_index, sentence in enumerate(sentences):
        for parser in parsers:
            try:
                candidates = parser.parse_sentence(sentence)
            except (ValueError, RuntimeError) as error:
                diagnostics.append(
                    BuildDiagnostic(
                        code="PARSER_FAILED",
                        example_id=example_id,
                        sentence_index=sentence_index,
                        parser=parser.name,
                        detail=str(error),
                    )
                )
                continue
            for candidate in candidates:
                if len(candidate.heads) != len(sentence.tokens):
                    diagnostics.append(
                        BuildDiagnostic(
                            code="TOKEN_ALIGNMENT_FAILED",
                            example_id=example_id,
                            sentence_index=sentence_index,
                            parser=parser.name,
                            detail=(
                                f"candidate has {len(candidate.heads)} tokens; "
                                f"master sentence has {len(sentence.tokens)}"
                            ),
                        )
                    )
                    continue
                sentence_candidates[sentence_index].append(candidate)
    parse_seconds = time.perf_counter() - parse_start

    forest_start = time.perf_counter()
    forest_sentences: list[SentenceForest] = []
    for sentence_index, sentence in enumerate(sentences):
        if not sentence_candidates[sentence_index]:
            raise RuntimeError(
                f"no aligned dependency candidates for {example_id} sentence {sentence_index}"
            )
        selected = merge_and_select(
            sentence_candidates[sentence_index], calibration, top_k=top_k
        )
        forest_sentences.append(
            SentenceForest(
                text_start=sentence.start,
                text_end=sentence.end,
                tokens=sentence.tokens,
                candidates=selected,
            )
        )
    forest = DocumentForest(example_id=example_id, sentences=tuple(forest_sentences))
    tokenized = tokenizer(
        source,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offsets: list[tuple[int, int]] = [
        (int(pair[0]), int(pair[1])) for pair in tokenized["offset_mapping"]
    ]
    aligned = align_document_forest(forest, offsets, [1] * len(offsets))
    forest_seconds = time.perf_counter() - forest_start
    return BuildResult(
        forest=forest,
        aligned=aligned,
        parse_seconds=parse_seconds,
        forest_seconds=forest_seconds,
        diagnostics=tuple(diagnostics),
    )
