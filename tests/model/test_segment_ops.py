from __future__ import annotations

import torch

from sumoe.model.segment_ops import segment_max, segment_softmax, segment_sum


def test_segment_softmax_normalizes_each_segment_and_column() -> None:
    logits = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    segments = torch.tensor([0, 0, 1])
    weights = segment_softmax(logits, segments, num_segments=2)
    assert torch.allclose(weights[segments == 0].sum(0), torch.ones(2))
    assert torch.allclose(weights[segments == 1].sum(0), torch.ones(2))


def test_segment_sum_and_max_match_hand_derived_values() -> None:
    values = torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 1.0]])
    segments = torch.tensor([0, 0, 1])
    assert torch.equal(segment_sum(values, segments, 2), torch.tensor([[4.0, 6.0], [5.0, 1.0]]))
    assert torch.equal(segment_max(values, segments, 2), torch.tensor([[3.0, 4.0], [5.0, 1.0]]))

