from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sumoe.evaluation.significance import holm_adjust, paired_three_seed_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-seed paired tests with Holm correction")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite significance report: {args.output}")
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    names = sorted(scores)
    results = [
        paired_three_seed_test(scores[name]["baseline"], scores[name]["sumoe"])
        for name in names
    ]
    adjusted = holm_adjust([result.p_value for result in results])
    payload = {
        name: {**result.to_dict(), "holm_adjusted_p": corrected}
        for name, result, corrected in zip(names, results, adjusted, strict=True)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
