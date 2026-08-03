from __future__ import annotations

import json
from pathlib import Path

from sumoe.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "configs/model/sumoe.yaml"


def test_canonical_config_matches_paper_constants() -> None:
    config = load_config(CANONICAL)
    assert config.model.base_model == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert config.model.injection_layers == (4, 8, 12, 16, 20, 24, 28, 32)
    assert (
        config.forest.candidate_top_k,
        config.model.num_experts,
        config.model.top_k,
    ) == (5, 8, 2)
    assert config.forest.forest_coefficient == 1.0
    assert config.training.seeds == (13, 21, 42)
    assert config.training.max_length == 4096


def test_every_controlled_and_sensitivity_overlay_loads() -> None:
    overlays = sorted((ROOT / "configs/ablation").glob("*.yaml"))
    assert len(overlays) == 16
    for overlay in overlays:
        load_config(CANONICAL, [overlay])


def test_deepspeed_configuration_matches_global_batch_arithmetic() -> None:
    value = json.loads((ROOT / "configs/deepspeed/zero3_bf16.json").read_text(encoding="utf-8"))
    assert value["train_batch_size"] == 64
    assert value["train_micro_batch_size_per_gpu"] == 1
    assert value["gradient_accumulation_steps"] == 8
    assert value["zero_optimization"]["stage"] == 3
    assert value["bf16"]["enabled"] is True
