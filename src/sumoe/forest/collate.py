from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch

from .alignment import AlignedDocumentForest


@dataclass(frozen=True)
class ForestBatch:
    batch_size: int
    sequence_length: int
    edge_dependent: torch.Tensor
    edge_head: torch.Tensor
    edge_marginal: torch.Tensor
    edge_document: torch.Tensor
    edge_sentence: torch.Tensor
    tree_edge_dependent: torch.Tensor
    tree_edge_head: torch.Tensor
    tree_edge_candidate: torch.Tensor
    tree_edge_sentence: torch.Tensor
    candidate_posterior: torch.Tensor
    candidate_document: torch.Tensor
    candidate_sentence: torch.Tensor

    def to(self, device: torch.device | str) -> ForestBatch:
        values: dict[str, object] = {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
        }
        for item in fields(self):
            if item.name in values:
                continue
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return ForestBatch(**values)  # type: ignore[arg-type]


def _long(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def collate_aligned_forests(
    forests: Sequence[AlignedDocumentForest],
    sequence_length: int,
) -> ForestBatch:
    if not forests:
        raise ValueError("at least one aligned forest is required")
    if any(forest.sequence_length > sequence_length for forest in forests):
        raise ValueError("an aligned forest exceeds the batch sequence length")

    tree_dependent: list[int] = []
    tree_head: list[int] = []
    tree_candidate: list[int] = []
    tree_sentence: list[int] = []
    candidate_posterior: list[float] = []
    candidate_document: list[int] = []
    candidate_sentence: list[int] = []
    coalesced: dict[tuple[int, int, int, int], float] = {}
    candidate_offset = 0
    sentence_offset = 0

    for document_index, forest in enumerate(forests):
        token_offset = document_index * sequence_length
        local_sentence_count = max(forest.candidate_sentence, default=-1) + 1
        for posterior, local_sentence in zip(
            forest.candidate_posterior, forest.candidate_sentence, strict=True
        ):
            candidate_posterior.append(posterior)
            candidate_document.append(document_index)
            candidate_sentence.append(sentence_offset + local_sentence)
        for edge in forest.edges:
            dependent = token_offset + edge.dependent
            head = token_offset + edge.head
            global_candidate = candidate_offset + edge.candidate
            global_sentence = sentence_offset + edge.sentence
            tree_dependent.append(dependent)
            tree_head.append(head)
            tree_candidate.append(global_candidate)
            tree_sentence.append(global_sentence)
            key = (dependent, head, document_index, global_sentence)
            coalesced[key] = coalesced.get(key, 0.0) + edge.posterior
        candidate_offset += len(forest.candidate_posterior)
        sentence_offset += local_sentence_count

    edge_keys = sorted(coalesced)
    return ForestBatch(
        batch_size=len(forests),
        sequence_length=sequence_length,
        edge_dependent=_long([key[0] for key in edge_keys]),
        edge_head=_long([key[1] for key in edge_keys]),
        edge_marginal=torch.tensor([coalesced[key] for key in edge_keys], dtype=torch.float32),
        edge_document=_long([key[2] for key in edge_keys]),
        edge_sentence=_long([key[3] for key in edge_keys]),
        tree_edge_dependent=_long(tree_dependent),
        tree_edge_head=_long(tree_head),
        tree_edge_candidate=_long(tree_candidate),
        tree_edge_sentence=_long(tree_sentence),
        candidate_posterior=torch.tensor(candidate_posterior, dtype=torch.float32),
        candidate_document=_long(candidate_document),
        candidate_sentence=_long(candidate_sentence),
    )

