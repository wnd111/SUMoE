from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sumoe.forest.transition.training import train_transition_parser


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper's Stack-Transformer parser")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("transition parser configuration must be a mapping")
    result = train_transition_parser(config, smoke_test=args.smoke_test)
    print(
        json.dumps(
            {
                "optimizer_steps": result.optimizer_steps,
                "best_dev_las": result.best_dev_las,
                "checkpoint": str(result.checkpoint),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

