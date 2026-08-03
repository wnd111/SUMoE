from __future__ import annotations

import torch

from sumoe.model.losses import load_balance_loss
from sumoe.model.router import StructureSemanticRouter


def test_router_uses_exactly_two_experts_and_breaks_ties_by_index() -> None:
    router = StructureSemanticRouter(
        hidden_size=8, structural_size=4, router_size=6, num_experts=4, top_k=2
    )
    for parameter in router.parameters():
        torch.nn.init.zeros_(parameter)
    output = router(
        torch.randn(1, 3, 8),
        torch.tensor([[True, True, False]]),
        torch.randn(1, 4),
    )
    assert output.indices.tolist() == [[0, 1]]
    assert torch.allclose(output.weights, torch.full((1, 2), 0.25))
    assert torch.count_nonzero(output.hard_mask, dim=-1).tolist() == [2]
    assert torch.allclose(output.dense_assignments.sum(-1), torch.full((1,), 0.5))
    assert torch.equal(
        output.selection_assignments,
        output.hard_mask.to(dtype=output.selection_assignments.dtype),
    )


def test_selection_assignments_keep_a_straight_through_router_gradient() -> None:
    router = StructureSemanticRouter(
        hidden_size=8, structural_size=4, router_size=6, num_experts=4, top_k=2
    )
    for parameter in router.parameters():
        torch.nn.init.zeros_(parameter)
    output = router(
        torch.randn(1, 3, 8),
        torch.tensor([[True, True, False]]),
        torch.randn(1, 4),
    )

    loss = load_balance_loss(output.selection_assignments, num_experts=4, top_k=2)
    loss.backward()

    assert router.classifier.bias.grad is not None
    assert torch.count_nonzero(router.classifier.bias.grad) > 0


def test_task_routing_uses_one_softmax_ste_gradient() -> None:
    router = StructureSemanticRouter(
        hidden_size=8, structural_size=4, router_size=6, num_experts=4, top_k=2
    )
    output = router(
        torch.randn(2, 3, 8),
        torch.tensor([[True, True, False], [True, False, False]]),
        torch.randn(2, 4),
    )
    coefficients = torch.tensor([[-1.0, 0.5, 2.0, 3.0], [1.0, -2.0, 0.25, 4.0]])

    actual = torch.autograd.grad(
        (output.dense_assignments * coefficients).sum(), output.logits, retain_graph=True
    )[0]
    expected = torch.autograd.grad((output.probabilities * coefficients).sum(), output.logits)[0]

    torch.testing.assert_close(actual, expected)


def test_selection_assignments_are_exact_hard_indicators_in_bfloat16() -> None:
    router = StructureSemanticRouter(
        hidden_size=8, structural_size=4, router_size=6, num_experts=4, top_k=2
    ).to(dtype=torch.bfloat16)
    for parameter in router.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        router.classifier.bias.copy_(torch.tensor([4.5, 0.0, -8.0, -12.0], dtype=torch.bfloat16))
    output = router(
        torch.zeros(1, 3, 8, dtype=torch.bfloat16),
        torch.tensor([[True, True, False]]),
        torch.zeros(1, 4, dtype=torch.bfloat16),
    )

    assert torch.equal(
        output.selection_assignments,
        output.hard_mask.to(dtype=output.selection_assignments.dtype),
    )
    loss = load_balance_loss(output.selection_assignments, num_experts=4, top_k=2)
    loss.backward()
    assert router.classifier.bias.grad is not None
    assert torch.count_nonzero(router.classifier.bias.grad) > 0


def test_router_ignores_masked_tokens_in_semantic_max_pool() -> None:
    router = StructureSemanticRouter(
        hidden_size=2, structural_size=2, router_size=2, num_experts=2, top_k=1
    )
    hidden = torch.tensor([[[1.0, 2.0], [1000.0, 1000.0]]])
    pooled = router.semantic_max_pool(hidden, torch.tensor([[True, False]]))
    assert torch.equal(pooled, torch.tensor([[1.0, 2.0]]))
