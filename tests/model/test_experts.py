from __future__ import annotations

import torch

from sumoe.model import experts
from sumoe.model.experts import SparseExpertPool
from sumoe.model.router import RoutingOutput


def routing() -> RoutingOutput:
    return RoutingOutput(
        logits=torch.zeros(2, 3),
        probabilities=torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]),
        indices=torch.tensor([[0, 1], [1, 2]]),
        weights=torch.full((2, 2), 0.5),
        dense_assignments=torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]),
        selection_assignments=torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        hard_mask=torch.tensor([[True, True, False], [False, True, True]]),
    )


def test_sparse_expert_pool_preserves_shape_and_gradients() -> None:
    pool = SparseExpertPool(hidden_size=8, intermediate_size=12, num_experts=3, dropout=0.0)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    output = pool(hidden, routing())
    assert output.shape == hidden.shape
    output.square().mean().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_transformer_expert_uses_prior_token_context() -> None:
    torch.manual_seed(17)
    expert = experts.TransformerExpert(
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        dropout=0.0,
    ).eval()
    baseline = torch.zeros(1, 3, 8)
    changed = baseline.clone()
    changed[0, 0, 0] = 1.0

    baseline_output = expert(baseline, torch.ones(1, 3, dtype=torch.bool))
    changed_output = expert(changed, torch.ones(1, 3, dtype=torch.bool))

    assert not torch.allclose(baseline_output[:, 1], changed_output[:, 1])
