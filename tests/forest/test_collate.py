from __future__ import annotations

import torch

from sumoe.forest.alignment import align_document_forest
from sumoe.forest.collate import collate_aligned_forests
from tests.forest.test_alignment import split_word_forest


def aligned_one():
    return align_document_forest(
        split_word_forest(), [(0, 12), (13, 19)], [1, 1]
    )


def test_document_batch_never_creates_cross_document_edges() -> None:
    batch = collate_aligned_forests([aligned_one(), aligned_one()], sequence_length=4)
    dependent_document = torch.div(batch.edge_dependent, 4, rounding_mode="floor")
    head_document = torch.div(batch.edge_head, 4, rounding_mode="floor")
    assert torch.equal(dependent_document, head_document)
    assert torch.equal(dependent_document, batch.edge_document)


def test_candidate_posteriors_are_preserved_per_document() -> None:
    batch = collate_aligned_forests([aligned_one(), aligned_one()], sequence_length=4)
    assert batch.candidate_posterior.tolist() == [1.0, 1.0]
    assert batch.candidate_document.tolist() == [0, 1]
    assert batch.tree_edge_candidate.min().item() == 0
    assert batch.tree_edge_candidate.max().item() == 1


def test_coalesced_edge_marginals_sum_duplicate_candidate_edges() -> None:
    batch = collate_aligned_forests([aligned_one()], sequence_length=4)
    assert torch.allclose(batch.edge_marginal, torch.ones_like(batch.edge_marginal))

