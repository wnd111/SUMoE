from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .types import DocumentForest, TokenSpan


@dataclass(frozen=True)
class AlignedEdge:
    dependent: int
    head: int
    sentence: int
    candidate: int
    posterior: float


@dataclass(frozen=True)
class AlignedDocumentForest:
    example_id: str
    sequence_length: int
    edges: tuple[AlignedEdge, ...]
    candidate_posterior: tuple[float, ...]
    candidate_sentence: tuple[int, ...]

    def edges_for_candidate(self, candidate: int) -> set[tuple[int, int]]:
        return {
            (edge.dependent, edge.head)
            for edge in self.edges
            if edge.candidate == candidate
        }


def _overlapping_subwords(
    token: TokenSpan,
    offset_mapping: Sequence[tuple[int, int]],
    source_mask: Sequence[int | bool],
) -> tuple[int, ...]:
    indices: list[int] = []
    pairs = zip(offset_mapping, source_mask, strict=True)
    for index, ((start, end), is_source) in enumerate(pairs):
        if not is_source or end <= start:
            continue
        if min(token.end, end) > max(token.start, start):
            indices.append(index)
    return tuple(indices)


def align_document_forest(
    forest: DocumentForest,
    offset_mapping: Sequence[tuple[int, int]],
    source_mask: Sequence[int | bool],
) -> AlignedDocumentForest:
    if len(offset_mapping) != len(source_mask):
        raise ValueError("offset_mapping and source_mask lengths differ")
    edges: list[AlignedEdge] = []
    candidate_posterior: list[float] = []
    candidate_sentence: list[int] = []
    candidate_index = 0
    for sentence_index, sentence in enumerate(forest.sentences):
        word_subwords = [
            _overlapping_subwords(token, offset_mapping, source_mask) for token in sentence.tokens
        ]
        for candidate in sentence.candidates:
            candidate_posterior.append(candidate.posterior)
            candidate_sentence.append(sentence_index)
            candidate_edges: set[tuple[int, int]] = set()
            for word_index, subwords in enumerate(word_subwords):
                if not subwords:
                    continue
                dependent = subwords[0]
                head_word = candidate.heads[word_index]
                if head_word == -1 or not word_subwords[head_word]:
                    head = dependent
                else:
                    head = word_subwords[head_word][0]
                candidate_edges.add((dependent, head))
                for continuation in subwords[1:]:
                    candidate_edges.add((continuation, continuation))
            for dependent, head in sorted(candidate_edges):
                edges.append(
                    AlignedEdge(
                        dependent=dependent,
                        head=head,
                        sentence=sentence_index,
                        candidate=candidate_index,
                        posterior=candidate.posterior,
                    )
                )
            candidate_index += 1
    if not edges:
        raise ValueError(f"document {forest.example_id} has no aligned source edges")
    return AlignedDocumentForest(
        example_id=forest.example_id,
        sequence_length=len(offset_mapping),
        edges=tuple(edges),
        candidate_posterior=tuple(candidate_posterior),
        candidate_sentence=tuple(candidate_sentence),
    )
