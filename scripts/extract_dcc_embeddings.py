from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sumoe.config import load_config
from sumoe.data.scrolls import NormalizedExample
from sumoe.evaluation.generation import PredictionCollator
from sumoe.model.factory import build_model
from sumoe.model.router import StructureSemanticRouter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract training-source LLaMA max-pool embeddings for DCC"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite DCC embeddings: {args.output}")
    config = load_config(args.config, args.overlay)
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    device = torch.device("cuda")
    model.to(device).eval()
    collator = PredictionCollator(tokenizer, None, config.training.max_length)
    embeddings: list[torch.Tensor] = []
    source_ids: list[str] = []
    seen: set[str] = set()
    with torch.no_grad():
        for task in config.data.tasks:
            path = Path(config.data.data_dir) / task / "train.jsonl"
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    row: dict[str, Any] = json.loads(line)
                    row["references"] = tuple(row["references"])
                    example = NormalizedExample(**row)
                    if example.source_id in seen:
                        continue
                    seen.add(example.source_id)
                    batch = collator([example])
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    source_mask = batch["source_mask"].to(device)
                    hidden = model.backbone.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    ).last_hidden_state
                    pooled = StructureSemanticRouter.semantic_max_pool(hidden, source_mask)
                    embeddings.append(pooled.float().cpu())
                    source_ids.append(example.source_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": torch.cat(embeddings), "source_ids": source_ids}, args.output)


if __name__ == "__main__":
    main()
