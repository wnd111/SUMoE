from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sumoe.evaluation.evaluator import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic SCROLLS predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {args.output}")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with args.predictions.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            grouped[str(row["task"])].append(row)
    reports = {}
    for task, rows in sorted(grouped.items()):
        report = evaluate_predictions(
            task,
            [str(row["prediction"]) for row in rows],
            [tuple(str(item) for item in row["references"]) for row in rows],
        )
        reports[task] = report.to_dict()
    payload = {
        "tasks": reports,
        "macro_scrolls_score": mean(
            float(report["scrolls_score"]) for report in reports.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
