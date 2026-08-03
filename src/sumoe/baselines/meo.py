from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.func import functional_call  # type: ignore[attr-defined]
from torch.nn import functional as F

from sumoe.model.experts import SwiGLUExpert, TransformerExpert


def initialize_meo_from_sumoe_state_dict(
    meo_model: nn.Module, sumoe_state: Mapping[str, torch.Tensor]
) -> None:
    """Load the trained SUMoE backbone, semantic router, heads, and experts into MEO."""
    settings = getattr(meo_model, "settings", None)
    if getattr(settings, "variant", None) != "meo":
        raise ValueError("SUMoE-to-MEO initialization requires a meo model")
    if not any(".injection." in key for key in sumoe_state):
        raise ValueError("source state is not a forest-injected SUMoE checkpoint")

    mapped: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for target_key, target_value in meo_model.state_dict().items():
        source_key = target_key
        if target_key.startswith("backbone.model.layers."):
            parts = target_key.split(".")
            source_key = ".".join((*parts[:4], "decoder_layer", *parts[4:]))
        source_value = sumoe_state.get(source_key)
        if source_value is None:
            missing.append(source_key)
            continue
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"SUMoE-to-MEO shape mismatch for {source_key}: "
                f"{tuple(source_value.shape)} != {tuple(target_value.shape)}"
            )
        mapped[target_key] = source_value
    if missing:
        raise KeyError(f"SUMoE checkpoint lacks MEO parameters: {', '.join(missing[:5])}")
    meo_model.load_state_dict(mapped, strict=True)


@dataclass(frozen=True)
class MergedSwiGLUParameters:
    norm_weight: torch.Tensor
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor
    norm_epsilon: float


def _stack_parameters(experts: Sequence[SwiGLUExpert], attribute: str) -> torch.Tensor:
    return torch.stack([getattr(expert, attribute).weight for expert in experts], dim=0)


def merge_swiglu_parameters(
    experts: Sequence[SwiGLUExpert], weights: torch.Tensor
) -> MergedSwiGLUParameters:
    if not experts:
        raise ValueError("at least one expert is required")
    if weights.ndim != 2 or weights.shape[1] != len(experts):
        raise ValueError("weights must have shape [batch, experts]")
    if not torch.allclose(
        weights.sum(dim=-1), torch.ones(weights.shape[0], device=weights.device), atol=1e-5
    ):
        raise ValueError("MEO weights must sum to one for every document")
    reference = experts[0]
    for expert in experts[1:]:
        if expert.gate_projection.weight.shape != reference.gate_projection.weight.shape:
            raise ValueError("all experts must have identical dimensions")
    norm = torch.stack([expert.norm.weight for expert in experts], dim=0)
    return MergedSwiGLUParameters(
        norm_weight=torch.einsum("be,ed->bd", weights, norm),
        gate_weight=torch.einsum(
            "be,eid->bid", weights, _stack_parameters(experts, "gate_projection")
        ),
        up_weight=torch.einsum("be,eid->bid", weights, _stack_parameters(experts, "up_projection")),
        down_weight=torch.einsum(
            "be,edi->bdi", weights, _stack_parameters(experts, "down_projection")
        ),
        norm_epsilon=reference.norm.epsilon,
    )


def explicit_merged_expert(
    hidden_states: torch.Tensor,
    parameters: MergedSwiGLUParameters,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
    normalized = hidden_states * torch.rsqrt(variance + parameters.norm_epsilon).to(
        hidden_states.dtype
    )
    normalized = normalized * parameters.norm_weight[:, None, :]
    gate = F.silu(torch.einsum("btd,bid->bti", normalized, parameters.gate_weight))
    up = torch.einsum("btd,bid->bti", normalized, parameters.up_weight)
    update = torch.einsum("bti,bdi->btd", gate * up, parameters.down_weight)
    return hidden_states + F.dropout(update, p=dropout, training=training)


def functional_merged_expert(
    hidden_states: torch.Tensor,
    experts: Sequence[SwiGLUExpert],
    weights: torch.Tensor,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    parameters = merge_swiglu_parameters(experts, weights)
    return explicit_merged_expert(hidden_states, parameters, dropout, training)


def functional_merged_transformer_expert(
    hidden_states: torch.Tensor,
    experts: Sequence[TransformerExpert],
    weights: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate one parameter-merged transformer expert per document."""
    if not experts:
        raise ValueError("at least one expert is required")
    if weights.shape != (hidden_states.shape[0], len(experts)):
        raise ValueError("weights must have shape [batch, experts]")
    if not torch.allclose(
        weights.sum(dim=-1),
        torch.ones(weights.shape[0], device=weights.device),
        atol=1e-5,
    ):
        raise ValueError("MEO weights must sum to one for every document")
    parameter_maps = [dict(expert.named_parameters()) for expert in experts]
    names = tuple(parameter_maps[0])
    if any(tuple(parameters) != names for parameters in parameter_maps[1:]):
        raise ValueError("all transformer experts must have identical parameters")

    outputs = []
    for document_index in range(hidden_states.shape[0]):
        merged_parameters = {}
        for name in names:
            stacked = torch.stack([parameters[name] for parameters in parameter_maps], dim=0)
            broadcast_shape = (len(experts),) + (1,) * (stacked.ndim - 1)
            merged_parameters[name] = (stacked * weights[document_index].view(broadcast_shape)).sum(
                dim=0
            )
        document_mask = (
            attention_mask[document_index : document_index + 1]
            if attention_mask is not None
            else None
        )
        outputs.append(
            functional_call(
                experts[0],
                merged_parameters,
                (hidden_states[document_index : document_index + 1], document_mask),
            )
        )
    return torch.cat(outputs, dim=0)
