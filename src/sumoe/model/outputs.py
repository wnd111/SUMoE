from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.utils import ModelOutput

from .router import RoutingOutput


@dataclass
class SumoeCausalLMOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    past_key_values: Any | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    routing: RoutingOutput | None = None
    routing_semantic: torch.Tensor | None = None
    routing_structural: torch.Tensor | None = None
    expert_history: torch.Tensor | None = None
    task_logits: torch.Tensor | None = None
    span_logits: torch.Tensor | None = None
    task_loss: torch.Tensor | None = None
    lm_loss: torch.Tensor | None = None
    classification_loss: torch.Tensor | None = None
    span_loss: torch.Tensor | None = None
    balance_loss: torch.Tensor | None = None
    forest_loss: torch.Tensor | None = None
