from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaForCausalLM

from sumoe.baselines.dcc import DCCState, route_dcc, route_dcc_training_assignments
from sumoe.baselines.meo import functional_merged_transformer_expert
from sumoe.data.tasks import TASKS
from sumoe.forest.collate import ForestBatch

from .experts import SparseExpertPool
from .forest_encoder import ForestInjectionOutput, SparseForestInjection
from .losses import (
    effective_batch_load_balance_loss,
    load_balance_loss,
    target_cross_entropy,
)
from .outputs import SumoeCausalLMOutput
from .router import RoutingOutput, StructureSemanticRouter
from .tree_readout import ForestTreeReadout


@dataclass(frozen=True)
class SumoeModelSettings:
    injection_layers: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32)
    forest_temperature: float = 1.0
    gat_node_size: int = 256
    gat_heads: int = 4
    router_size: int = 1024
    num_experts: int = 8
    top_k: int = 2
    expert_heads: int = 8
    expert_intermediate_size: int = 4096
    expert_dropout: float = 0.1
    balance_coefficient: float = 0.05
    forest_coefficient: float = 1.0
    dcc_centroids_file: str | None = None
    variant: str = "sumoe"


class ForestInjectedDecoderLayer(nn.Module):
    def __init__(self, decoder_layer: nn.Module, injection: SparseForestInjection) -> None:
        super().__init__()
        self.decoder_layer = decoder_layer
        self.injection = injection
        self.forest_batch: ForestBatch | None = None
        self.source_mask: torch.Tensor | None = None
        self.last_output: ForestInjectionOutput | None = None

    def set_context(
        self, forest_batch: ForestBatch | None, source_mask: torch.Tensor | None
    ) -> None:
        self.forest_batch = forest_batch
        self.source_mask = source_mask
        self.last_output = None

    def forward(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        output = self.decoder_layer(*args, **kwargs)
        if not isinstance(output, tuple):
            raise TypeError("LLaMA decoder layer must return a tuple")
        if self.forest_batch is None or self.source_mask is None:
            return output
        injected = self.injection(output[0], self.forest_batch, self.source_mask)
        self.last_output = injected
        return (injected.hidden_states, *output[1:])


class SumoeForCausalLM(nn.Module):
    """LLaMA causal LM with paper-aligned forest injections and document MoE."""

    def __init__(self, backbone: LlamaForCausalLM, settings: SumoeModelSettings) -> None:
        super().__init__()
        self.backbone = backbone
        self.settings = settings
        self.config = backbone.config
        self.last_generation_routing: RoutingOutput | None = None
        hidden_size = int(backbone.config.hidden_size)
        self.classification_heads = nn.ModuleDict(
            {
                task: nn.Linear(hidden_size, len(spec.class_labels))
                for task, spec in TASKS.items()
                if spec.prediction_head == "classification"
            }
        )
        self.span_head = nn.Linear(hidden_size, 2)
        num_layers = len(backbone.model.layers)
        if any(layer < 1 or layer > num_layers for layer in settings.injection_layers):
            raise ValueError("injection layer lies outside the LLaMA decoder")
        if settings.variant not in {
            "llama",
            "llama_forest",
            "llama_moe",
            "dcc",
            "meo",
            "sumoe",
        }:
            raise ValueError(f"unsupported variant: {settings.variant}")

        self.uses_forest = settings.variant in {"llama_forest", "sumoe"}
        self.uses_moe = settings.variant in {"llama_moe", "sumoe", "dcc", "meo"}
        self.tree_readout: ForestTreeReadout | None
        self.router: StructureSemanticRouter | None
        self.expert_pool: SparseExpertPool | None
        self.dcc_state: DCCState | None = None
        if settings.variant == "dcc" and settings.dcc_centroids_file is not None:
            self.dcc_state = DCCState.load(Path(settings.dcc_centroids_file))
        self.injection_wrappers: list[ForestInjectedDecoderLayer] = []
        if self.uses_forest:
            for one_based_layer in settings.injection_layers:
                index = one_based_layer - 1
                wrapper = ForestInjectedDecoderLayer(
                    backbone.model.layers[index],
                    SparseForestInjection(
                        hidden_size=hidden_size,
                        ffn_size=hidden_size,
                        tau=settings.forest_temperature,
                    ),
                )
                backbone.model.layers[index] = wrapper
                self.injection_wrappers.append(wrapper)

        if self.uses_moe:
            self.tree_readout = (
                ForestTreeReadout(
                    hidden_size=hidden_size,
                    node_size=settings.gat_node_size,
                    num_heads=settings.gat_heads,
                    num_layers=2,
                )
                if settings.variant == "sumoe"
                else None
            )
            self.router = (
                StructureSemanticRouter(
                    hidden_size=hidden_size,
                    structural_size=settings.gat_node_size,
                    router_size=settings.router_size,
                    num_experts=settings.num_experts,
                    top_k=settings.top_k,
                )
                if settings.variant != "dcc"
                else None
            )
            self.expert_pool = SparseExpertPool(
                hidden_size=hidden_size,
                intermediate_size=settings.expert_intermediate_size,
                num_experts=settings.num_experts,
                num_heads=settings.expert_heads,
                dropout=settings.expert_dropout,
            )
        else:
            self.tree_readout = None
            self.router = None
            self.expert_pool = None

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        settings: SumoeModelSettings,
        **kwargs: Any,
    ) -> SumoeForCausalLM:
        backbone = LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        return cls(backbone, settings)

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.backbone.gradient_checkpointing_disable()

    def set_dcc_state(self, state: DCCState) -> None:
        if self.settings.variant != "dcc":
            raise ValueError("DCC state can only be attached to the dcc variant")
        if state.centroids.shape != (
            self.settings.num_experts,
            int(self.config.hidden_size),
        ):
            raise ValueError("DCC centroids must have shape [num_experts, hidden_size]")
        self.dcc_state = state

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        task_names: Sequence[str] | None = None,
        source_ids: Sequence[str] | None = None,
        classification_labels: torch.Tensor | None = None,
        span_start_positions: torch.Tensor | None = None,
        span_end_positions: torch.Tensor | None = None,
        forest_batch: ForestBatch | None = None,
        past_key_values: Any | None = None,
        routing_cache: RoutingOutput | None = None,
        expert_history: torch.Tensor | None = None,
        defer_balance_loss: bool = False,
        use_cache: bool | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> SumoeCausalLMOutput:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds is required")
        if input_ids is not None:
            current_shape = input_ids.shape
        else:
            if inputs_embeds is None:
                raise RuntimeError("validated inputs_embeds is unexpectedly absent")
            current_shape = inputs_embeds.shape[:2]
        if source_mask is None:
            if attention_mask is None or attention_mask.shape != current_shape:
                raise ValueError(
                    "source_mask is required when attention_mask includes cached tokens"
                )
            source_mask = attention_mask.to(dtype=torch.bool)
        if source_mask.shape != current_shape:
            raise ValueError("source_mask must match the current input sequence")

        if input_ids is not None:
            device = input_ids.device
        else:
            if inputs_embeds is None:
                raise RuntimeError("validated inputs_embeds is unexpectedly absent")
            device = inputs_embeds.device
        active_forest = forest_batch.to(device) if forest_batch is not None else None
        if self.uses_forest and past_key_values is None and active_forest is None:
            raise ValueError("forest_batch is required for the first forest-enabled forward pass")
        for wrapper in self.injection_wrappers:
            wrapper.set_context(active_forest, source_mask if active_forest is not None else None)

        model_output = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = model_output.last_hidden_state
        current_sequence_length = hidden_states.shape[1]
        next_expert_history = None

        zero = hidden_states.sum() * 0.0
        forest_losses = [
            wrapper.last_output.loss
            for wrapper in self.injection_wrappers
            if wrapper.last_output is not None
        ]
        forest_loss = torch.stack(forest_losses).mean() if forest_losses else zero

        routing = routing_cache
        routing_semantic = None
        routing_structural = None
        balance_loss = zero
        if self.uses_moe:
            if routing is None:
                if self.settings.variant == "dcc":
                    if self.dcc_state is None:
                        raise ValueError(
                            "dcc variant requires dcc_centroids_file or set_dcc_state()"
                        )
                    if self.training:
                        if source_ids is None:
                            raise ValueError("DCC training requires source_ids for fixed routing")
                        routing = route_dcc_training_assignments(
                            tuple(source_ids),
                            self.dcc_state,
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    else:
                        semantic = StructureSemanticRouter.semantic_max_pool(
                            hidden_states, source_mask
                        )
                        routing = route_dcc(semantic, self.dcc_state, self.settings.top_k)
                elif self.uses_forest:
                    if active_forest is None:
                        raise ValueError("routing cache is required for a cached forest step")
                    if self.tree_readout is None:
                        raise RuntimeError("tree readout was not initialized")
                    structural = self.tree_readout(hidden_states, active_forest)
                else:
                    structural = hidden_states.new_zeros(
                        (hidden_states.shape[0], self.settings.gat_node_size)
                    )
                if self.settings.variant != "dcc":
                    router = self.router
                    if router is None:
                        raise RuntimeError("router was not initialized")
                    semantic = StructureSemanticRouter.semantic_max_pool(hidden_states, source_mask)
                    routing_semantic = semantic.detach()
                    routing_structural = structural.detach()
                    routing = router.route_summaries(semantic, structural)
            if self.expert_pool is None or routing is None:
                raise RuntimeError("expert pool was not initialized")
            if past_key_values is not None and expert_history is None:
                raise ValueError("expert_history is required for a cached MoE forward pass")
            if expert_history is not None:
                expected_shape = (hidden_states.shape[0], int(self.config.hidden_size))
                if (
                    expert_history.ndim != 3
                    or expert_history.shape[0] != expected_shape[0]
                    or expert_history.shape[2] != expected_shape[1]
                ):
                    raise ValueError(
                        "expert_history must have shape [batch, cached_sequence, hidden]"
                    )
                if expert_history.device != hidden_states.device:
                    raise ValueError("expert_history must be on the same device as hidden states")
                expert_input = torch.cat((expert_history, hidden_states), dim=1)
            else:
                expert_input = hidden_states
            if attention_mask is not None and attention_mask.shape[1] < expert_input.shape[1]:
                raise ValueError("attention_mask is shorter than the expert sequence")
            expert_attention_mask = (
                attention_mask[:, -expert_input.shape[1] :].to(dtype=torch.bool)
                if attention_mask is not None
                else None
            )
            if self.settings.variant == "meo":
                hidden_states = functional_merged_transformer_expert(
                    expert_input,
                    list(self.expert_pool.experts),
                    routing.probabilities,
                    expert_attention_mask,
                )
            else:
                hidden_states = self.expert_pool(expert_input, routing, expert_attention_mask)
            hidden_states = hidden_states[:, -current_sequence_length:]
            if model_output.past_key_values is not None:
                next_expert_history = expert_input
            if not defer_balance_loss and self.settings.variant != "dcc":
                balance_loss = load_balance_loss(
                    routing.selection_assignments,
                    self.settings.num_experts,
                    coefficient=self.settings.balance_coefficient,
                    top_k=self.settings.top_k,
                )

        logits = self.backbone.lm_head(hidden_states).float()
        task_logits = None
        classification_loss = zero
        classification_supervised = False
        if task_names is not None:
            if len(task_names) != hidden_states.shape[0]:
                raise ValueError("task_names must contain one task per document")
            unknown_tasks = sorted(set(task_names) - set(TASKS))
            if unknown_tasks:
                raise ValueError(f"unsupported task names: {', '.join(unknown_tasks)}")
            max_classes = max(
                len(spec.class_labels)
                for spec in TASKS.values()
                if spec.prediction_head == "classification"
            )
            task_logits = hidden_states.new_full(
                (hidden_states.shape[0], max_classes), -torch.inf
            ).float()
            classification_losses = []
            for task, head in self.classification_heads.items():
                document_indices = torch.tensor(
                    [index for index, name in enumerate(task_names) if name == task],
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                if document_indices.numel() == 0:
                    continue
                pooled = StructureSemanticRouter.semantic_max_pool(
                    hidden_states.index_select(0, document_indices),
                    source_mask.index_select(0, document_indices),
                )
                one_task_logits = head(pooled).float()
                task_logits[document_indices, : one_task_logits.shape[1]] = one_task_logits
                if classification_labels is None:
                    continue
                selected_labels = classification_labels.to(hidden_states.device).index_select(
                    0, document_indices
                )
                valid = selected_labels.ne(-100)
                if valid.any():
                    classification_losses.append(
                        F.cross_entropy(one_task_logits[valid], selected_labels[valid])
                    )
                    classification_supervised = True
            if classification_losses:
                classification_loss = torch.stack(classification_losses).mean()

        span_logits = None
        span_loss = zero
        span_supervised = False
        if span_start_positions is not None or span_end_positions is not None:
            if span_start_positions is None or span_end_positions is None:
                raise ValueError("both span start and end positions are required")
            if span_start_positions.shape != (hidden_states.shape[0],) or (
                span_end_positions.shape != (hidden_states.shape[0],)
            ):
                raise ValueError("span positions must have shape [batch]")
            span_logits = self.span_head(hidden_states).float()
            start_positions = span_start_positions.to(hidden_states.device)
            end_positions = span_end_positions.to(hidden_states.device)
            if start_positions.ne(-100).any() or end_positions.ne(-100).any():
                span_loss = 0.5 * (
                    F.cross_entropy(span_logits[:, :, 0], start_positions, ignore_index=-100)
                    + F.cross_entropy(span_logits[:, :, 1], end_positions, ignore_index=-100)
                )
                span_supervised = True

        lm_loss = target_cross_entropy(logits, labels) if labels is not None else None
        generation_supervised = labels is not None and bool(labels.ne(-100).any())
        task_loss = None
        if generation_supervised or classification_supervised or span_supervised:
            task_loss = zero
            if generation_supervised and lm_loss is not None:
                task_loss = task_loss + lm_loss
            if classification_supervised:
                task_loss = task_loss + classification_loss
            if span_supervised:
                task_loss = task_loss + span_loss
        total_loss = None
        if task_loss is not None:
            total_loss = task_loss + balance_loss + self.settings.forest_coefficient * forest_loss
        return SumoeCausalLMOutput(
            loss=total_loss,
            logits=logits,
            past_key_values=model_output.past_key_values,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
            routing=routing,
            routing_semantic=routing_semantic,
            routing_structural=routing_structural,
            expert_history=next_expert_history,
            task_logits=task_logits,
            span_logits=span_logits,
            task_loss=task_loss,
            lm_loss=lm_loss,
            classification_loss=classification_loss,
            span_loss=span_loss,
            balance_loss=balance_loss,
            forest_loss=forest_loss,
        )

    def effective_batch_balance_loss(
        self,
        semantic_summaries: torch.Tensor,
        structural_summaries: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate Eq. (10) over stop-gradient feature snapshots for router training."""
        if self.router is None:
            raise RuntimeError("effective-batch balance requires a trainable router")
        routing = self.router.route_summaries(semantic_summaries, structural_summaries)
        return effective_batch_load_balance_loss(
            routing.selection_assignments,
            self.settings.num_experts,
            coefficient=self.settings.balance_coefficient,
            top_k=self.settings.top_k,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_mask: torch.Tensor,
        forest_batch: ForestBatch | None,
        max_new_tokens: int = 512,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Deterministic beam-1 decoding with cached routing and key/value states."""
        if max_new_tokens < 1:
            return input_ids
        generated = input_ids
        mask = attention_mask
        output = self(
            input_ids=input_ids,
            attention_mask=mask,
            source_mask=source_mask,
            forest_batch=forest_batch,
            use_cache=True,
        )
        routing_cache = output.routing
        self.last_generation_routing = routing_cache
        past_key_values = output.past_key_values
        expert_history = output.expert_history
        for token_index in range(max_new_tokens):
            if output.logits is None:
                raise RuntimeError("model did not return logits")
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and next_token.eq(eos_token_id).all():
                break
            if token_index + 1 == max_new_tokens:
                break
            mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
            output = self(
                input_ids=next_token,
                attention_mask=mask,
                source_mask=torch.zeros_like(next_token, dtype=torch.bool),
                past_key_values=past_key_values,
                routing_cache=routing_cache,
                expert_history=expert_history,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            expert_history = output.expert_history
        return generated

    def settings_dict(self) -> dict[str, Any]:
        return asdict(self.settings)
