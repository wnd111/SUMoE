from __future__ import annotations

import torch

from sumoe.forest.collate import ForestBatch
from sumoe.model.tree_readout import ForestTreeReadout


def two_sentence_forest() -> ForestBatch:
    return ForestBatch(
        batch_size=1,
        sequence_length=4,
        edge_dependent=torch.tensor([0, 1, 2, 3]),
        edge_head=torch.tensor([0, 0, 2, 2]),
        edge_marginal=torch.ones(4),
        edge_document=torch.zeros(4, dtype=torch.long),
        edge_sentence=torch.tensor([0, 0, 1, 1]),
        tree_edge_dependent=torch.tensor([0, 1, 0, 1, 2, 3]),
        tree_edge_head=torch.tensor([0, 0, 1, 1, 2, 2]),
        tree_edge_candidate=torch.tensor([0, 0, 1, 1, 2, 2]),
        tree_edge_sentence=torch.tensor([0, 0, 0, 0, 1, 1]),
        candidate_posterior=torch.tensor([0.75, 0.25, 1.0]),
        candidate_document=torch.tensor([0, 0, 0]),
        candidate_sentence=torch.tensor([0, 0, 1]),
    )


def test_tree_readout_applies_candidate_posteriors_then_sentence_mean() -> None:
    torch.manual_seed(1)
    readout = ForestTreeReadout(hidden_size=8, node_size=8, num_heads=4, num_layers=2)
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    result = readout(hidden, two_sentence_forest())
    assert result.shape == (1, 8)
    result.sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_tree_readout_returns_zero_for_empty_forest() -> None:
    forest = two_sentence_forest()
    forest = ForestBatch(
        **{
            **forest.__dict__,
            "tree_edge_dependent": torch.empty(0, dtype=torch.long),
            "tree_edge_head": torch.empty(0, dtype=torch.long),
            "tree_edge_candidate": torch.empty(0, dtype=torch.long),
            "tree_edge_sentence": torch.empty(0, dtype=torch.long),
        }
    )
    readout = ForestTreeReadout(hidden_size=8, node_size=8, num_heads=4, num_layers=2)
    result = readout(torch.randn(1, 4, 8), forest)
    assert torch.equal(result, torch.zeros_like(result))
