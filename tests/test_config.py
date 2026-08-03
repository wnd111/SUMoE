from __future__ import annotations

from pathlib import Path

import pytest

from sumoe.config import ExperimentConfig, ModelConfig, deep_merge, load_config

PAPER_YAML = """
model:
  base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
  hidden_size: 4096
  num_layers: 32
  injection_layers: [4, 8, 12, 16, 20, 24, 28, 32]
  num_experts: 8
  top_k: 2
  expert_heads: 8
  expert_intermediate_size: 4096
  expert_dropout: 0.1
  router_dim: 1024
  gat_node_dim: 256
  gat_heads: 4
forest:
  candidate_top_k: 5
  edge_temperature: 1.0
  forest_coefficient: 1.0
training:
  global_batch_size: 64
  world_size: 8
  micro_batch_size: 1
  gradient_accumulation: 8
  learning_rate: 0.00003
  beta1: 0.9
  beta2: 0.98
  weight_decay: 0.01
  balance_coefficient: 0.05
  max_length: 4096
  max_target_length: 512
  seeds: [13, 21, 42]
data:
  tasks: [gov_report, summ_screen_fd, qmsum, qasper, narrative_qa, quality, contract_nli]
evaluation:
  max_new_tokens: 512
  num_beams: 1
  do_sample: false
variant: sumoe
"""


def minimal_mapping() -> dict[str, object]:
    import yaml

    value = yaml.safe_load(PAPER_YAML)
    assert isinstance(value, dict)
    return value


def test_paper_configuration_accepts_exact_defaults(tmp_path: Path) -> None:
    path = tmp_path / "sumoe.yaml"
    path.write_text(PAPER_YAML, encoding="utf-8")
    cfg = load_config(path)
    assert cfg.model.injection_layers == (4, 8, 12, 16, 20, 24, 28, 32)
    assert cfg.model.num_experts == 8
    assert cfg.model.top_k == 2
    assert cfg.training.global_batch_size == 64
    assert cfg.training.seeds == (13, 21, 42)


def test_model_config_default_uses_the_eight_paper_injection_locations() -> None:
    config = ModelConfig(base_model="meta-llama/Meta-Llama-3.1-8B-Instruct")
    assert config.injection_layers == (4, 8, 12, 16, 20, 24, 28, 32)


def test_unknown_configuration_key_is_rejected() -> None:
    data = minimal_mapping()
    model = data["model"]
    assert isinstance(model, dict)
    model["invented_key"] = 1
    with pytest.raises(ValueError, match="Unknown model keys: invented_key"):
        ExperimentConfig.from_mapping(data)


def test_global_batch_arithmetic_is_exact() -> None:
    data = minimal_mapping()
    training = data["training"]
    assert isinstance(training, dict)
    training["gradient_accumulation"] = 4
    with pytest.raises(ValueError, match="global batch size must equal 64"):
        ExperimentConfig.from_mapping(data)


def test_expert_attention_heads_must_divide_hidden_size() -> None:
    data = minimal_mapping()
    model = data["model"]
    assert isinstance(model, dict)
    model["expert_heads"] = 7
    with pytest.raises(ValueError, match="expert_heads must divide hidden_size"):
        ExperimentConfig.from_mapping(data)


def test_overlay_is_recursive_without_mutating_base() -> None:
    base = {"model": {"num_experts": 8, "top_k": 2}, "variant": "sumoe"}
    merged = deep_merge(base, {"model": {"top_k": 1}})
    assert merged == {"model": {"num_experts": 8, "top_k": 1}, "variant": "sumoe"}
    assert base["model"] == {"num_experts": 8, "top_k": 2}
