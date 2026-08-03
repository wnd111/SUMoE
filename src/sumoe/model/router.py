from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RoutingOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor
    dense_assignments: torch.Tensor
    selection_assignments: torch.Tensor
    hard_mask: torch.Tensor


class StructureSemanticRouter(nn.Module):
    """Document-level router combining semantic and forest representations."""

    def __init__(
        self,
        hidden_size: int,
        structural_size: int = 256,
        router_size: int = 1024,
        num_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.semantic_projection = nn.Linear(hidden_size, router_size)
        self.structural_projection = nn.Linear(structural_size, router_size)
        self.fusion = nn.Sequential(
            nn.Linear(router_size, router_size),
            nn.GELU(),
        )
        self.classifier = nn.Linear(router_size, num_experts)

    @staticmethod
    def semantic_max_pool(hidden_states: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or source_mask.shape != hidden_states.shape[:2]:
            raise ValueError("source_mask must match the first two hidden-state dimensions")
        mask = source_mask.to(device=hidden_states.device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("every document must contain at least one source token")
        masked = hidden_states.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        return masked.max(dim=1).values

    def forward(
        self,
        hidden_states: torch.Tensor,
        source_mask: torch.Tensor,
        structural_summary: torch.Tensor,
    ) -> RoutingOutput:
        semantic = self.semantic_max_pool(hidden_states, source_mask)
        return self.route_summaries(semantic, structural_summary)

    def route_summaries(
        self,
        semantic_summary: torch.Tensor,
        structural_summary: torch.Tensor,
    ) -> RoutingOutput:
        if semantic_summary.ndim != 2:
            raise ValueError("semantic_summary must have shape [batch, hidden]")
        if structural_summary.ndim != 2 or structural_summary.shape[0] != semantic_summary.shape[0]:
            raise ValueError("structural_summary batch dimension does not match semantic_summary")
        fused = self.semantic_projection(semantic_summary) + self.structural_projection(
            structural_summary
        )
        logits = self.classifier(self.fusion(fused))
        probabilities = torch.softmax(logits, dim=-1)

        order = torch.argsort(probabilities, dim=-1, descending=True, stable=True)
        indices = order[:, : self.top_k]
        weights = probabilities.gather(1, indices)
        hard_weights = torch.zeros_like(probabilities).scatter(1, indices, weights)
        hard_mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter(1, indices, True)
        dense_assignments = hard_weights.detach() + probabilities - probabilities.detach()
        selection_assignments = hard_mask.to(dtype=probabilities.dtype) + (
            probabilities - probabilities.detach()
        )
        return RoutingOutput(
            logits=logits,
            probabilities=probabilities,
            indices=indices,
            weights=weights,
            dense_assignments=dense_assignments,
            selection_assignments=selection_assignments,
            hard_mask=hard_mask,
        )
