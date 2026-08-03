from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from sumoe.baselines.dcc import fit_dcc_centroids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the fixed DCC document centroids")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    embeddings = value["embeddings"] if isinstance(value, dict) else value
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("embedding file must contain a tensor or {'embeddings': tensor}")
    source_ids = tuple(value["source_ids"]) if isinstance(value, dict) else None
    state = fit_dcc_centroids(embeddings, args.num_experts, args.seed, source_ids=source_ids)
    state.save(args.output)
    print(f"saved {state.centroids.shape[0]} centroids to {args.output}")


if __name__ == "__main__":
    main()
