from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def read_scores(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    values = {"macro_scrolls_score": float(report["macro_scrolls_score"])}
    values.update(
        {
            task: float(task_report["scrolls_score"])
            for task, task_report in report["tasks"].items()
        }
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect paired seed scores from evaluation reports"
    )
    parser.add_argument("--baseline", type=Path, nargs=3, required=True)
    parser.add_argument("--sumoe", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite paired scores: {args.output}")
    baseline = [read_scores(path) for path in args.baseline]
    sumoe = [read_scores(path) for path in args.sumoe]
    keys = set(baseline[0])
    if any(set(row) != keys for row in (*baseline, *sumoe)):
        raise ValueError("all six evaluation reports must contain identical task keys")
    payload = {
        key: {
            "baseline": [row[key] for row in baseline],
            "sumoe": [row[key] for row in sumoe],
        }
        for key in sorted(keys)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
