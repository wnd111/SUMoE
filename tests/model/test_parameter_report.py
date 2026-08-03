from __future__ import annotations

from transformers import LlamaConfig

from scripts.count_parameters import build_meta_model, paper_backbone_config
from sumoe.model.factory import build_tiny_model
from sumoe.model.parameter_report import count_parameters
from sumoe.model.sumoe_model import SumoeModelSettings


def build_tiny_sumoe():
    backbone_config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        _attn_implementation="eager",
    )
    settings = SumoeModelSettings(
        injection_layers=(1, 2, 3, 4),
        gat_node_size=16,
        gat_heads=4,
        router_size=16,
        num_experts=4,
        top_k=2,
        expert_intermediate_size=32,
        expert_dropout=0.0,
        variant="sumoe",
    )
    return build_tiny_model(backbone_config, settings)


def test_parameter_report_partitions_every_parameter_once() -> None:
    model = build_tiny_sumoe()

    report = count_parameters(model)

    assert (
        report.backbone,
        report.forest_injections,
        report.tree_readout,
        report.router,
        report.experts,
        report.task_heads,
    ) == (41248, 20864, 2208, 1140, 29440, 297)
    categorized = (
        report.backbone
        + report.forest_injections
        + report.tree_readout
        + report.router
        + report.experts
        + report.task_heads
    )
    assert categorized == report.total
    assert report.total == sum(parameter.numel() for parameter in model.parameters())
    assert report.trainable == report.total


def test_parameter_report_detects_frozen_parameters() -> None:
    model = build_tiny_sumoe()
    next(model.backbone.parameters()).requires_grad_(False)

    report = count_parameters(model)

    assert report.trainable < report.total


def test_meta_device_parameter_report_counts_tied_embeddings_once() -> None:
    backbone_config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        _attn_implementation="eager",
    )
    settings = SumoeModelSettings(
        injection_layers=(1, 2, 3, 4),
        gat_node_size=16,
        gat_heads=4,
        router_size=16,
        num_experts=4,
        top_k=2,
        expert_intermediate_size=32,
        expert_dropout=0.0,
        variant="sumoe",
    )
    normal_model = build_tiny_model(backbone_config, settings)
    meta_model = build_meta_model(backbone_config, settings)

    assert meta_model.backbone.model.embed_tokens.weight is meta_model.backbone.lm_head.weight
    assert count_parameters(meta_model).total == count_parameters(normal_model).total


def test_paper_backbone_config_is_pinned_offline_llama_31_8b() -> None:
    config = paper_backbone_config("meta-llama/Meta-Llama-3.1-8B-Instruct")

    assert config.vocab_size == 128256
    assert config.hidden_size == 4096
    assert config.intermediate_size == 14336
    assert config.num_hidden_layers == 32
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
