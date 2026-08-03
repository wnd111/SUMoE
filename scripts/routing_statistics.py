from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sumoe.evaluation.routing_stats import routing_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SUMoE routing diagnostics")
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-experts", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite routing report: {args.output}")
    tasks: list[str] = []
    rows: list[list[float]] = []
    with args.assignments.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            tasks.append(str(row["task"]))
            rows.append([float(value) for value in row["assignments"]])
    report = routing_statistics(torch.tensor(rows), tasks, args.num_experts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
