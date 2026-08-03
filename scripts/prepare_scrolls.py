from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from sumoe.config import PAPER_TASKS
from sumoe.data.scrolls import normalize_scrolls_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize all seven SCROLLS tasks")
    parser.add_argument("--dataset", default="tau/scrolls")
    parser.add_argument("--output-dir", type=Path, default=Path("data/scrolls"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in PAPER_TASKS:
        dataset = load_dataset(args.dataset, task)
        for split, rows in dataset.items():
            output_path = args.output_dir / task / f"{split}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    for example in normalize_scrolls_example(task, split, row):
                        stream.write(
                            json.dumps(example.to_dict(), ensure_ascii=False) + "\n"
                        )
            print(f"wrote {len(rows)} source rows to {output_path}")


if __name__ == "__main__":
    main()
