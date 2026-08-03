from __future__ import annotations

import pytest
import torch

from sumoe.model.losses import (
    effective_batch_load_balance_loss,
    load_balance_loss,
    target_cross_entropy,
)


def test_uniform_expert_use_minimizes_negative_entropy_term() -> None:
    uniform = torch.eye(8)
    collapsed = torch.zeros(8, 8)
    collapsed[:, 0] = 1.0
    assert load_balance_loss(uniform, 8, 0.05) < load_balance_loss(collapsed, 8, 0.05)


def test_load_balance_loss_matches_paper_equation_10() -> None:
    selected_experts = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    coefficient = 0.05
    usage = torch.tensor([0.25, 0.50, 0.25, 0.0])
    nonzero_usage = usage[usage > 0]
    expected = coefficient * torch.sum(nonzero_usage * torch.log(nonzero_usage)) / 4

    actual = load_balance_loss(selected_experts, 4, coefficient, top_k=2)

    assert torch.allclose(actual, expected)


def test_effective_batch_balance_distinguishes_collapsed_from_balanced_usage() -> None:
    one_document = torch.tensor([[1.0, 1.0, 0.0, 0.0]], requires_grad=True)
    collapsed = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], requires_grad=True)
    balanced = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], requires_grad=True)

    single_loss = effective_batch_load_balance_loss(one_document, 4, top_k=2)
    collapsed_loss = effective_batch_load_balance_loss(collapsed, 4, top_k=2)
    balanced_loss = effective_batch_load_balance_loss(balanced, 4, top_k=2)

    assert single_loss.item() == pytest.approx(collapsed_loss.item())
    assert balanced_loss < collapsed_loss


def test_load_balance_loss_rejects_weighted_mixture_assignments() -> None:
    weighted_assignments = torch.tensor([[0.75, 0.25, 0.0, 0.0]])

    with pytest.raises(ValueError, match="binary top-k indicators"):
        load_balance_loss(weighted_assignments, 4, top_k=1)


def test_load_balance_loss_rejects_nonbinary_integer_assignments() -> None:
    signed_assignments = torch.tensor([[-1.0, 2.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="binary top-k indicators"):
        load_balance_loss(signed_assignments, 4, top_k=1)


def test_target_cross_entropy_uses_causal_shift_and_ignores_source() -> None:
    logits = torch.tensor([[[9.0, -9.0], [-9.0, 9.0], [-9.0, 9.0]]], requires_grad=True)
    labels = torch.tensor([[-100, 0, 1]])
    loss = target_cross_entropy(logits, labels)
    assert loss < 1e-5
    loss.backward()
    assert logits.grad is not None
