from __future__ import annotations

import torch
from torch.nn import functional as F


def target_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Causal language-model loss; source positions are represented by -100 labels."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("labels must match the first two logits dimensions")
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(-100)
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def _validate_selection_assignments(
    selection_assignments: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    if selection_assignments.ndim != 2:
        raise ValueError("assignments must have shape [batch, experts]")
    if selection_assignments.shape[1] != num_experts:
        raise ValueError("assignment width does not match num_experts")
    if selection_assignments.shape[0] == 0:
        raise ValueError("assignments batch must not be empty")
    if not 1 <= top_k <= num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    detached_assignments = selection_assignments.detach()
    close_to_zero = torch.isclose(
        detached_assignments,
        torch.zeros_like(detached_assignments),
        rtol=0.0,
        atol=1e-6,
    )
    close_to_one = torch.isclose(
        detached_assignments,
        torch.ones_like(detached_assignments),
        rtol=0.0,
        atol=1e-6,
    )
    if not torch.logical_or(close_to_zero, close_to_one).all():
        raise ValueError("assignments must be binary top-k indicators")
    selected_per_sample = detached_assignments.sum(dim=1)
    expected = torch.full_like(selected_per_sample, float(top_k))
    if not torch.allclose(selected_per_sample, expected, rtol=0.0, atol=1e-6):
        raise ValueError("each sample must contain exactly top_k selections")
    return detached_assignments


def _negative_usage_entropy(
    utilization: torch.Tensor, num_experts: int, coefficient: float
) -> torch.Tensor:
    return (coefficient / num_experts) * torch.sum(
        utilization * torch.log(utilization.clamp_min(1e-12))
    )


def load_balance_loss(
    selection_assignments: torch.Tensor,
    num_experts: int,
    coefficient: float = 0.05,
    top_k: int = 1,
) -> torch.Tensor:
    """Paper Eq. (10) over one physical forward batch."""
    _validate_selection_assignments(selection_assignments, num_experts, top_k)
    utilization = selection_assignments.sum(dim=0) / (selection_assignments.shape[0] * top_k)
    return _negative_usage_entropy(utilization, num_experts, coefficient)


def effective_batch_load_balance_loss(
    selection_assignments: torch.Tensor,
    num_experts: int,
    coefficient: float = 0.05,
    top_k: int = 1,
) -> torch.Tensor:
    """Eq. (10) over accumulated local documents and all distributed ranks."""
    detached = _validate_selection_assignments(selection_assignments, num_experts, top_k)
    global_counts = detached.sum(dim=0)
    global_documents = torch.tensor(
        float(selection_assignments.shape[0]),
        device=selection_assignments.device,
        dtype=selection_assignments.dtype,
    )
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(global_counts)
        torch.distributed.all_reduce(global_documents)
        world_size = torch.distributed.get_world_size()

    local_counts = selection_assignments.sum(dim=0)
    straight_through_counts = global_counts + world_size * (local_counts - local_counts.detach())
    utilization = straight_through_counts / (global_documents * top_k)
    return _negative_usage_entropy(utilization, num_experts, coefficient)
