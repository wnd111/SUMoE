from __future__ import annotations

import torch
from torch import nn

from .router import RoutingOutput


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.epsilon).to(hidden_states.dtype)
        return normalized * self.weight


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.gate_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_projection = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden_states)
        gated = torch.nn.functional.silu(self.gate_projection(normalized))
        update = self.down_projection(gated * self.up_projection(normalized))
        return hidden_states + self.dropout(update)


class TransformerExpert(nn.Module):
    """Lightweight causal transformer block used as one SUMoE expert."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by expert num_heads")
        self.attention_norm = RMSNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn = SwiGLUExpert(hidden_size, intermediate_size, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        batch_size, sequence_length, _ = hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must match the expert sequence")
        normalized = self.attention_norm(hidden_states)
        causal_mask = torch.triu(
            torch.ones(
                (sequence_length, sequence_length),
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            diagonal=1,
        )
        attention_output, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=~attention_mask.to(dtype=torch.bool),
            need_weights=False,
        )
        hidden_states = hidden_states + self.attention_dropout(attention_output)
        return self.ffn(hidden_states)


class SparseExpertPool(nn.Module):
    """Evaluate only the selected document-expert pairs."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = 4096,
        num_experts: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                TransformerExpert(hidden_size, intermediate_size, num_heads, dropout)
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing: RoutingOutput,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if routing.hard_mask.shape != (hidden_states.shape[0], self.num_experts):
            raise ValueError("routing mask does not match batch and expert dimensions")
        combined = torch.zeros_like(hidden_states)
        for expert_index, expert in enumerate(self.experts):
            document_indices = torch.nonzero(
                routing.hard_mask[:, expert_index], as_tuple=False
            ).flatten()
            if document_indices.numel() == 0:
                continue
            expert_input = hidden_states.index_select(0, document_indices)
            expert_mask = (
                attention_mask.index_select(0, document_indices)
                if attention_mask is not None
                else None
            )
            expert_output = expert(expert_input, expert_mask)
            weights = routing.dense_assignments[document_indices, expert_index].view(-1, 1, 1)
            combined.index_add_(0, document_indices, expert_output * weights)
        return combined
