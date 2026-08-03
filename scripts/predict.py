from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sumoe.baselines.meo import initialize_meo_from_sumoe_state_dict
from sumoe.config import load_config
from sumoe.data.scrolls import NormalizedExample
from sumoe.evaluation.generation import PredictionCollator, greedy_generate
from sumoe.forest.io import read_forests
from sumoe.model.factory import build_model
from sumoe.training.checkpoint import load_checkpoint_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy prediction for one SCROLLS split")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    settings = getattr(model, "settings", None)
    expected_variant = getattr(settings, "variant", None)
    if expected_variant == "meo":
        state = load_checkpoint_model_state(checkpoint, "sumoe")
        initialize_meo_from_sumoe_state_dict(model, state)
    else:
        state = load_checkpoint_model_state(checkpoint, expected_variant)
        model.load_state_dict(state, strict=True)


def normalized_rows(path: Path) -> list[NormalizedExample]:
    examples: list[NormalizedExample] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row: dict[str, Any] = json.loads(line)
            row["references"] = tuple(row["references"])
            examples.append(NormalizedExample(**row))
    return examples


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prediction file: {args.output}")
    config = load_config(args.config, args.overlay)
    if args.task not in config.data.tasks:
        raise ValueError(f"task is not present in the paper configuration: {args.task}")
    examples = normalized_rows(Path(config.data.data_dir) / args.task / f"{args.split}.jsonl")
    _, forest_iterator = read_forests(
        Path(config.data.forest_dir) / args.task / f"{args.split}.jsonl"
    )
    forest_index = {forest.example_id: forest for forest in forest_iterator}
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = PredictionCollator(tokenizer, forest_index, config.training.max_length)
    loader = DataLoader(examples, batch_size=1, shuffle=False, collate_fn=collator)
    model = build_model(
        config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    load_weights(model, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for batch in loader:
            for key in ("input_ids", "attention_mask", "source_mask"):
                batch[key] = batch[key].to(device)
            batch["forest_batch"] = batch["forest_batch"].to(device)
            result = greedy_generate(model, batch, tokenizer, config.evaluation.max_new_tokens)
            stream.write(
                json.dumps(
                    {
                        "example_id": result.example_ids[0],
                        "task": args.task,
                        "prediction": result.predictions[0],
                        "references": batch["references"][0],
                        "assignments": (
                            result.routing_assignments[0]
                            if result.routing_assignments is not None
                            else None
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
