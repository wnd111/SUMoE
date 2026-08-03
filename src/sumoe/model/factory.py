from __future__ import annotations

from transformers import LlamaConfig, LlamaForCausalLM

from sumoe.config import ExperimentConfig

from .sumoe_model import SumoeForCausalLM, SumoeModelSettings


def settings_from_experiment(config: ExperimentConfig) -> SumoeModelSettings:
    return SumoeModelSettings(
        injection_layers=config.model.injection_layers,
        forest_temperature=config.forest.edge_temperature,
        gat_node_size=config.model.gat_node_dim,
        gat_heads=config.model.gat_heads,
        router_size=config.model.router_dim,
        num_experts=config.model.num_experts,
        top_k=config.model.top_k,
        expert_heads=config.model.expert_heads,
        expert_intermediate_size=config.model.expert_intermediate_size,
        expert_dropout=config.model.expert_dropout,
        balance_coefficient=config.training.balance_coefficient,
        forest_coefficient=config.forest.forest_coefficient,
        dcc_centroids_file=config.model.dcc_centroids_file,
        variant=config.variant,
    )


def build_model(config: ExperimentConfig, **from_pretrained_kwargs: object) -> SumoeForCausalLM:
    return SumoeForCausalLM.from_pretrained(
        config.model.base_model,
        settings_from_experiment(config),
        **from_pretrained_kwargs,
    )


def build_tiny_model(llama_config: LlamaConfig, settings: SumoeModelSettings) -> SumoeForCausalLM:
    return SumoeForCausalLM(LlamaForCausalLM(llama_config), settings)
