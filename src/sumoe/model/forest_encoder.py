from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from sumoe.forest.collate import ForestBatch


@dataclass(frozen=True)
class ForestInjectionOutput:
    hidden_states: torch.Tensor
    structural_states: torch.Tensor
    valid_mask: torch.Tensor
    loss: torch.Tensor


class SparseForestInjection(nn.Module):
    """Paper Eq. (4)-(6): posterior-weighted forest aggregation and gated fusion."""

    def __init__(
        self,
        hidden_size: int,
        ffn_size: int | None = None,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.hidden_size = hidden_size
        self.tau = tau
        inner_size = ffn_size or hidden_size

        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.structural_ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Linear(inner_size, hidden_size),
        )
        self.gate = nn.Linear(2 * hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forest: ForestBatch,
        source_mask: torch.Tensor,
    ) -> ForestInjectionOutput:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError("hidden size does not match this injection module")
        if (batch_size, sequence_length) != (forest.batch_size, forest.sequence_length):
            raise ValueError("forest dimensions do not match hidden_states")
        if source_mask.shape != hidden_states.shape[:2]:
            raise ValueError("source_mask must have shape [batch, sequence]")

        active_source_mask = source_mask.to(device=hidden_states.device, dtype=torch.bool)
        if not active_source_mask.any(dim=1).all():
            raise ValueError("every document must contain at least one source token")

        flat_hidden = hidden_states.reshape(-1, hidden_size)
        dependents = forest.edge_dependent.to(hidden_states.device)
        heads = forest.edge_head.to(hidden_states.device)
        marginals = forest.edge_marginal.to(hidden_states.device, torch.float32)
        flat_source_mask = active_source_mask.reshape(-1)
        num_flat_tokens = flat_hidden.shape[0]
        if dependents.numel():
            if dependents.min() < 0 or dependents.max() >= num_flat_tokens:
                raise ValueError("forest dependent index lies outside the batch")
            if heads.min() < 0 or heads.max() >= num_flat_tokens:
                raise ValueError("forest head index lies outside the batch")
            if not flat_source_mask.index_select(0, dependents).all():
                raise ValueError("forest dependents must be source tokens")
            if not flat_source_mask.index_select(0, heads).all():
                raise ValueError("forest heads must be source tokens")

        projected_values = self.structural_ffn(self.value(flat_hidden)).float()
        document_for_token = torch.arange(
            batch_size, device=hidden_states.device
        ).repeat_interleave(sequence_length)
        source_documents = document_for_token[flat_source_mask]
        document_value_sums = projected_values.new_zeros((batch_size, hidden_size))
        document_value_sums.index_add_(0, source_documents, projected_values[flat_source_mask])
        source_counts = active_source_mask.sum(dim=1).to(dtype=torch.float32)

        # Eq. (4) assigns logit A_ij / tau to every source-token pair. Missing
        # sparse edges have A_ij = 0, so their unnormalized weight is one. The
        # algebra below is exactly the dense softmax/weighted sum from Eqs. (4)
        # and (5), evaluated without materializing a [B, T, T] tensor.
        row_max = marginals.new_zeros(num_flat_tokens)
        if dependents.numel():
            scaled_marginals = marginals / self.tau
            row_max.scatter_reduce_(
                0, dependents, scaled_marginals, reduce="amax", include_self=True
            )
        else:
            scaled_marginals = marginals
        base_scale = torch.exp(-row_max)
        row_documents = document_for_token
        numerator = base_scale.unsqueeze(-1) * document_value_sums.index_select(0, row_documents)
        denominator = base_scale * source_counts.index_select(0, row_documents)
        if dependents.numel():
            edge_scale = torch.exp(scaled_marginals - row_max.index_select(0, dependents))
            correction = edge_scale - base_scale.index_select(0, dependents)
            numerator = numerator.index_add(
                0,
                dependents,
                correction.unsqueeze(-1) * projected_values.index_select(0, heads),
            )
            denominator = denominator.index_add(0, dependents, correction)
        flat_structural = (numerator / denominator.clamp_min(1e-12).unsqueeze(-1)).to(
            hidden_states.dtype
        )
        flat_structural = torch.where(
            flat_source_mask.unsqueeze(-1), flat_structural, torch.zeros_like(flat_structural)
        )

        structural_states = flat_structural.view(batch_size, sequence_length, hidden_size)
        valid_mask = active_source_mask
        gate = torch.sigmoid(self.gate(torch.cat((hidden_states, structural_states), dim=-1)))
        fused = gate * structural_states + (1.0 - gate) * hidden_states
        fused = torch.where(valid_mask.unsqueeze(-1), fused, hidden_states)

        edge_connected = torch.zeros_like(flat_source_mask)
        if dependents.numel():
            edge_connected.index_fill_(0, dependents, True)
        loss_mask = valid_mask & edge_connected.view(batch_size, sequence_length)
        if loss_mask.any():
            similarities = F.cosine_similarity(
                hidden_states[loss_mask].float(), structural_states[loss_mask].float(), dim=-1
            )
            forest_loss = (1.0 - similarities).mean().to(hidden_states.dtype)
        else:
            forest_loss = hidden_states.sum() * 0.0
        return ForestInjectionOutput(
            hidden_states=fused,
            structural_states=structural_states,
            valid_mask=valid_mask,
            loss=forest_loss,
        )
