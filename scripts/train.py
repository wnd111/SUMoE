from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, LlamaConfig

from sumoe.config import ExperimentConfig, load_config
from sumoe.data.collator import SumoeDataCollator
from sumoe.data.scrolls import NormalizedExample
from sumoe.forest.collate import ForestBatch
from sumoe.forest.io import read_forests
from sumoe.model.factory import build_model, build_tiny_model, settings_from_experiment
from sumoe.training.manifest import build_run_manifest
from sumoe.training.trainer import SumoeTrainer, TrainerSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the paper-aligned SUMoE implementation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=13, choices=(13, 21, 42))
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def trainer_settings(config: ExperimentConfig, distributed: bool) -> TrainerSettings:
    values = config.training
    return TrainerSettings(
        learning_rate=values.learning_rate,
        beta1=values.beta1,
        beta2=values.beta2,
        weight_decay=values.weight_decay,
        epochs=values.epochs,
        gradient_accumulation=values.gradient_accumulation,
        warmup_ratio=values.warmup_ratio,
        minimum_learning_rate=values.minimum_learning_rate,
        max_grad_norm=values.max_grad_norm,
        save_steps=values.save_steps,
        logging_steps=values.logging_steps,
        distributed=distributed,
    )


def build_training_sampler(examples: Sequence[object], seed: int) -> RandomSampler[object]:
    """Shuffle the unified multitask examples once per epoch without family oversampling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return RandomSampler(examples, replacement=False, generator=generator)


def tiny_forest() -> ForestBatch:
    return ForestBatch(
        batch_size=1,
        sequence_length=6,
        edge_dependent=torch.tensor([0, 1, 2, 3]),
        edge_head=torch.tensor([0, 0, 1, 2]),
        edge_marginal=torch.ones(4),
        edge_document=torch.zeros(4, dtype=torch.long),
        edge_sentence=torch.zeros(4, dtype=torch.long),
        tree_edge_dependent=torch.tensor([0, 1, 2, 3]),
        tree_edge_head=torch.tensor([0, 0, 1, 2]),
        tree_edge_candidate=torch.zeros(4, dtype=torch.long),
        tree_edge_sentence=torch.zeros(4, dtype=torch.long),
        candidate_posterior=torch.ones(1),
        candidate_document=torch.zeros(1, dtype=torch.long),
        candidate_sentence=torch.zeros(1, dtype=torch.long),
    )


def run_smoke(config: ExperimentConfig, output_dir: Path) -> None:
    llama = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        _attn_implementation="eager",
    )
    settings = replace(
        settings_from_experiment(config),
        injection_layers=(1, 2),
        gat_node_size=16,
        gat_heads=4,
        router_size=16,
        num_experts=4,
        top_k=2,
        expert_intermediate_size=32,
        expert_dropout=0.0,
        variant="sumoe",
    )
    model = build_tiny_model(llama, settings)
    batch = {
        "input_ids": torch.tensor([[1, 5, 6, 7, 8, 2]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "source_mask": torch.tensor([[True, True, True, True, False, False]]),
        "labels": torch.tensor([[-100, -100, -100, -100, 8, 2]]),
        "forest_batch": tiny_forest(),
        "example_ids": ["smoke::0"],
    }
    loader = DataLoader([batch], batch_size=None)
    settings_train = TrainerSettings(
        epochs=1,
        gradient_accumulation=1,
        save_steps=1,
        logging_steps=1,
    )
    manifest = build_run_manifest(asdict(config), [], [])
    result = SumoeTrainer(model, loader, output_dir, settings_train, manifest).train(
        max_optimizer_steps=1
    )
    print(json.dumps(asdict(result), default=str))


def load_training_data(
    config: ExperimentConfig,
) -> tuple[list[NormalizedExample], dict[str, object], list[Path], list[Path]]:
    examples: list[NormalizedExample] = []
    forests: dict[str, object] = {}
    data_files: list[Path] = []
    forest_files: list[Path] = []
    for task in config.data.tasks:
        data_path = Path(config.data.data_dir) / task / "train.jsonl"
        forest_path = Path(config.data.forest_dir) / task / "train.jsonl"
        data_files.append(data_path)
        forest_files.append(forest_path)
        with data_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                row["references"] = tuple(row["references"])
                examples.append(NormalizedExample(**row))
        _, task_forests = read_forests(forest_path)
        for forest in task_forests:
            forests[forest.example_id] = forest
    return examples, forests, data_files, forest_files


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overlay)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = args.output_dir or Path("outputs") / config.variant / f"seed-{args.seed}"
    if args.smoke_test:
        run_smoke(config, output_dir)
        return
    if config.variant == "meo":
        raise ValueError(
            "MEO is an evaluation-only condition initialized from a trained SUMoE "
            "checkpoint; use scripts/predict.py or scripts/profile_efficiency.py."
        )

    examples, forests, data_files, forest_files = load_training_data(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = SumoeDataCollator(
        tokenizer,
        forests,  # type: ignore[arg-type]
        config.training.max_length,
        config.training.max_target_length,
    )
    sampler = build_training_sampler(examples, args.seed)
    loader = DataLoader(examples, batch_size=1, sampler=sampler, collate_fn=collator)
    model = build_model(
        config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    manifest = build_run_manifest(config, data_files, forest_files)
    trainer = SumoeTrainer(
        model,
        loader,
        output_dir,
        trainer_settings(config, args.distributed),
        manifest,
    )
    result = trainer.train(args.resume_from_checkpoint)
    if trainer.accelerator is None or trainer.accelerator.is_main_process:
        print(json.dumps(asdict(result), default=str))


if __name__ == "__main__":
    main()
