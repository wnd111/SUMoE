from __future__ import annotations

import torch
from torch import nn

from sumoe.forest.collate import ForestBatch
from sumoe.model.forest_encoder import SparseForestInjection


def forest_with_edges_only_for_tokens_0_and_1() -> ForestBatch:
    return ForestBatch(
        batch_size=1,
        sequence_length=4,
        edge_dependent=torch.tensor([0, 1]),
        edge_head=torch.tensor([1, 1]),
        edge_marginal=torch.tensor([1.0, 1.0]),
        edge_document=torch.tensor([0, 0]),
        edge_sentence=torch.tensor([0, 0]),
        tree_edge_dependent=torch.tensor([0, 1]),
        tree_edge_head=torch.tensor([1, 1]),
        tree_edge_candidate=torch.tensor([0, 0]),
        tree_edge_sentence=torch.tensor([0, 0]),
        candidate_posterior=torch.tensor([1.0]),
        candidate_document=torch.tensor([0]),
        candidate_sentence=torch.tensor([0]),
    )


def forest_with_two_heads(marginals: torch.Tensor) -> ForestBatch:
    return ForestBatch(
        batch_size=1,
        sequence_length=3,
        edge_dependent=torch.tensor([0, 0]),
        edge_head=torch.tensor([1, 2]),
        edge_marginal=marginals,
        edge_document=torch.tensor([0, 0]),
        edge_sentence=torch.tensor([0, 0]),
        tree_edge_dependent=torch.tensor([0, 0]),
        tree_edge_head=torch.tensor([1, 2]),
        tree_edge_candidate=torch.tensor([0, 1]),
        tree_edge_sentence=torch.tensor([0, 0]),
        candidate_posterior=torch.tensor([0.5, 0.5]),
        candidate_document=torch.tensor([0, 0]),
        candidate_sentence=torch.tensor([0, 0]),
    )


def test_equation_4_normalizes_over_every_source_position() -> None:
    module = SparseForestInjection(hidden_size=2, ffn_size=2, tau=1.0)
    module.structural_ffn = nn.Identity()
    with torch.no_grad():
        module.value.weight.copy_(torch.eye(2))

    hidden = torch.tensor([[[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]])
    output = module(
        hidden,
        forest_with_two_heads(torch.tensor([0.0, torch.log(torch.tensor(2.0))])),
        torch.ones(1, 3, dtype=torch.bool),
    )

    # A[0] = [0, 0, log(2)], so Eq. (4) gives alpha = [1, 1, 2] / 4.
    torch.testing.assert_close(output.structural_states[0, 0], torch.tensor([5.0, 0.0]))


def test_equation_5_applies_ffn_before_all_token_aggregation() -> None:
    class Square(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs.square()

    module = SparseForestInjection(hidden_size=2, ffn_size=2, tau=1.0)
    module.structural_ffn = Square()
    with torch.no_grad():
        module.value.weight.copy_(torch.eye(2))

    hidden = torch.tensor([[[2.0, 0.0], [1.0, 0.0], [3.0, 0.0]]])
    output = module(
        hidden,
        forest_with_two_heads(torch.tensor([0.0, 0.0])),
        torch.ones(1, 3, dtype=torch.bool),
    )

    torch.testing.assert_close(output.structural_states[0, 0], torch.tensor([14.0 / 3.0, 0.0]))


def test_equation_4_does_not_use_query_key_compatibility() -> None:
    module = SparseForestInjection(hidden_size=2, ffn_size=2, tau=1.0)
    module.structural_ffn = nn.Identity()
    with torch.no_grad():
        module.value.weight.copy_(torch.eye(2))

    hidden = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]])
    output = module(
        hidden,
        forest_with_two_heads(torch.tensor([0.0, 0.0])),
        torch.ones(1, 3, dtype=torch.bool),
    )

    expected = torch.tensor([1.0 / 3.0, 0.0])
    torch.testing.assert_close(output.structural_states[0, 0], expected)


def test_equation_4_uses_uniform_rows_for_source_tokens_without_edges() -> None:
    module = SparseForestInjection(hidden_size=2, ffn_size=2, tau=1.0)
    module.structural_ffn = nn.Identity()
    with torch.no_grad():
        module.value.weight.copy_(torch.eye(2))
    hidden = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]])
    output = module(
        hidden,
        forest_with_edges_only_for_tokens_0_and_1(),
        torch.ones(1, 4, dtype=torch.bool),
    )

    assert output.valid_mask.tolist() == [[True, True, True, True]]
    torch.testing.assert_close(output.structural_states[0, 2], torch.tensor([2.25, 0.0]))


def test_positions_outside_the_source_sequence_remain_unchanged() -> None:
    torch.manual_seed(3)
    module = SparseForestInjection(hidden_size=32, ffn_size=32, tau=1.0)
    hidden = torch.randn(1, 4, 32, requires_grad=True)
    output = module(
        hidden,
        forest_with_edges_only_for_tokens_0_and_1(),
        torch.tensor([[True, True, False, False]]),
    )

    assert torch.equal(output.hidden_states[:, 2:], hidden[:, 2:])
    assert output.valid_mask.tolist() == [[True, True, False, False]]
    output.hidden_states.sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_forest_loss_is_finite_over_source_tokens() -> None:
    module = SparseForestInjection(hidden_size=32, ffn_size=32, tau=1.0)
    hidden = torch.randn(1, 4, 32)
    output = module(
        hidden,
        forest_with_edges_only_for_tokens_0_and_1(),
        torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
    )
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)


def test_equation_11_forest_loss_uses_only_tokens_with_retained_edges() -> None:
    module = SparseForestInjection(hidden_size=2, ffn_size=2, tau=1.0)
    module.structural_ffn = nn.Identity()
    with torch.no_grad():
        module.value.weight.copy_(torch.eye(2))

    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]])
    output = module(
        hidden,
        forest_with_edges_only_for_tokens_0_and_1(),
        torch.ones(1, 4, dtype=torch.bool),
    )

    # Only dependent positions 0 and 1 belong to Omega in Eq. (11).
    torch.testing.assert_close(output.loss, torch.tensor(0.5))
