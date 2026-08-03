from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

from transformers import LlamaConfig

from sumoe.config import ForestConfig, load_config
from sumoe.model.factory import build_tiny_model
from sumoe.model.forest_encoder import SparseForestInjection
from sumoe.model.sumoe_model import (
    ForestInjectedDecoderLayer,
    SumoeModelSettings,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER_INJECTION_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


def paper_depth_tiny_model():
    backbone = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        _attn_implementation="eager",
    )
    settings = SumoeModelSettings(
        gat_node_size=16,
        router_size=16,
        expert_intermediate_size=32,
        expert_dropout=0.0,
        variant="llama_forest",
    )
    return build_tiny_model(backbone, settings)


def test_eq4_forest_aggregation_has_no_obsolete_multihead_setting() -> None:
    assert "forest_heads" not in {field.name for field in fields(ForestConfig)}
    assert "forest_heads" not in {field.name for field in fields(SumoeModelSettings)}
    assert "num_heads" not in inspect.signature(SparseForestInjection).parameters

    config = load_config(ROOT / "configs" / "model" / "sumoe.yaml")
    assert not hasattr(config.forest, "forest_heads")


def test_default_configuration_matches_the_finley_experiment_contract() -> None:
    config = load_config(ROOT / "configs" / "model" / "sumoe.yaml")

    assert config.forest.candidate_top_k == 5
    assert config.forest.edge_temperature == 1.0
    assert config.model.injection_layers == PAPER_INJECTION_LAYERS
    assert config.model.num_experts == 8
    assert config.model.top_k == 2
    assert config.model.expert_heads == 8
    assert config.model.expert_intermediate_size == 4096
    assert config.model.expert_dropout == 0.1
    assert config.model.router_dim == 1024
    assert config.model.gat_node_dim == 256
    assert config.model.gat_heads == 4
    assert config.training.global_batch_size == 64
    assert config.training.learning_rate == 3e-5
    assert config.training.beta1 == 0.9
    assert config.training.beta2 == 0.98
    assert config.training.weight_decay == 0.01
    assert config.training.balance_coefficient == 0.05
    assert config.training.seeds == (13, 21, 42)


def test_default_model_settings_use_the_eight_paper_injection_locations() -> None:
    assert SumoeModelSettings().injection_layers == PAPER_INJECTION_LAYERS


def test_model_wraps_the_eight_paper_decoder_blocks() -> None:
    model = paper_depth_tiny_model()

    wrapped = tuple(
        index
        for index, layer in enumerate(model.backbone.model.layers, start=1)
        if isinstance(layer, ForestInjectedDecoderLayer)
    )

    assert wrapped == PAPER_INJECTION_LAYERS


def test_forest_injection_modules_have_independent_parameters() -> None:
    model = paper_depth_tiny_model()
    injections = [wrapper.injection for wrapper in model.injection_wrappers]

    assert len(injections) == 8
    assert len({id(injection) for injection in injections}) == 8


def test_public_docs_do_not_describe_an_alternative_cross_attention_formula() -> None:
    for relative_path in (Path("README.md"), Path("docs/IMPLEMENTATION_NOTES.md")):
        content = (ROOT / relative_path).read_text(encoding="utf-8").casefold()
        assert "cross-attention" not in content, relative_path
