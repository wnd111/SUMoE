from __future__ import annotations

import torch


def _validate_segments(
    values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> None:
    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    if segment_ids.ndim != 1 or segment_ids.shape[0] != values.shape[0]:
        raise ValueError("segment_ids must be one-dimensional and match values")
    if num_segments < 0:
        raise ValueError("num_segments must be non-negative")
    if segment_ids.numel() and (
        int(segment_ids.min()) < 0 or int(segment_ids.max()) >= num_segments
    ):
        raise ValueError("segment id lies outside [0, num_segments)")


def segment_sum(
    values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Sum rows of ``values`` that have the same segment id."""
    _validate_segments(values, segment_ids, num_segments)
    output = values.new_zeros((num_segments, *values.shape[1:]))
    if values.shape[0]:
        output.index_add_(0, segment_ids, values)
    return output


def segment_max(
    values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Compute a column-wise maximum for every segment."""
    _validate_segments(values, segment_ids, num_segments)
    if not (values.is_floating_point() or values.is_complex()):
        fill_value: float | int = torch.iinfo(values.dtype).min
    else:
        fill_value = -torch.inf
    output = torch.full(
        (num_segments, *values.shape[1:]),
        fill_value,
        dtype=values.dtype,
        device=values.device,
    )
    if values.shape[0]:
        expanded_ids = segment_ids.reshape(
            (segment_ids.shape[0],) + (1,) * (values.ndim - 1)
        ).expand_as(values)
        output.scatter_reduce_(0, expanded_ids, values, reduce="amax", include_self=True)
    return output


def segment_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Numerically stable softmax independently within each segment."""
    _validate_segments(logits, segment_ids, num_segments)
    if logits.shape[0] == 0:
        return logits.clone()
    maxima = segment_max(logits, segment_ids, num_segments)
    shifted = logits - maxima.index_select(0, segment_ids)
    exponentials = shifted.exp()
    denominators = segment_sum(exponentials, segment_ids, num_segments)
    return exponentials / denominators.index_select(0, segment_ids).clamp_min(epsilon)
