from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from accelerate import init_empty_weights
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sumoe.config import load_config
from sumoe.model.factory import settings_from_experiment
from sumoe.model.parameter_report import count_parameters
from sumoe.model.sumoe_model import SumoeForCausalLM, SumoeModelSettings

PAPER_BACKBONE = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def paper_backbone_config(base_model: str) -> LlamaConfig:
    """Return the pinned public architecture metadata; no Hub access is required."""
    if base_model != PAPER_BACKBONE:
        raise ValueError(
            f"offline parameter counting is pinned to {PAPER_BACKBONE!r}, got {base_model!r}"
        )
    return LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
    )


def build_meta_model(
    backbone_config: LlamaConfig, settings: SumoeModelSettings
) -> SumoeForCausalLM:
    with init_empty_weights():
        backbone = LlamaForCausalLM(backbone_config)
        model = SumoeForCausalLM(backbone, settings)
    backbone.tie_weights()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report model-derived parameter counts without downloading model weights"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.overlay)
    backbone_config = paper_backbone_config(config.model.base_model)
    model = build_meta_model(backbone_config, settings_from_experiment(config))
    print(json.dumps(count_parameters(model).to_dict(), indent=2))


if __name__ == "__main__":
    main()
