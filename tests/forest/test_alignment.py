from __future__ import annotations

from sumoe.forest.alignment import align_document_forest
from sumoe.forest.types import (
    CandidateTree,
    DocumentForest,
    SentenceForest,
    TokenSpan,
)


def split_word_forest() -> DocumentForest:
    return DocumentForest(
        example_id="gov_report::train::1",
        sentences=(
            SentenceForest(
                text_start=0,
                text_end=19,
                tokens=(TokenSpan("unbelievable", 0, 12), TokenSpan("result", 13, 19)),
                candidates=(
                    CandidateTree(
                        parser="stanza",
                        heads=(1, -1),
                        labels=("amod", "root"),
                        raw_score=-0.1,
                        posterior=1.0,
                    ),
                ),
            ),
        ),
    )


def test_incoming_edge_targets_first_overlapping_subword() -> None:
    aligned = align_document_forest(
        split_word_forest(),
        offset_mapping=[(0, 2), (2, 7), (7, 12), (13, 19)],
        source_mask=[1, 1, 1, 1],
    )
    assert aligned.edges_for_candidate(0) == {(0, 3), (1, 1), (2, 2), (3, 3)}


def test_head_removed_by_truncation_becomes_self_loop() -> None:
    aligned = align_document_forest(
        split_word_forest(),
        offset_mapping=[(0, 12)],
        source_mask=[1],
    )
    assert aligned.edges_for_candidate(0) == {(0, 0)}


def test_target_tokens_are_never_added_to_forest() -> None:
    aligned = align_document_forest(
        split_word_forest(),
        offset_mapping=[(0, 12), (13, 19)],
        source_mask=[1, 0],
    )
    assert aligned.edges_for_candidate(0) == {(0, 0)}

