from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sumoe.forest.calibration import fit_temperature


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit parser temperatures and LAS priors")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[str, list[tuple[float, float]]] = {}
    with args.input.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            grouped.setdefault(str(row["parser"]), []).append(
                (float(row["score"]), float(row["correct"]))
            )
    if set(grouped) != {"stanza", "spacy", "transition"}:
        raise ValueError("calibration input must contain stanza, spacy, and transition rows")
    las = {name: sum(value for _, value in rows) / len(rows) for name, rows in grouped.items()}
    total_las = sum(las.values())
    output: dict[str, object] = {"schema_version": 1, "parsers": {}}
    parser_output = output["parsers"]
    assert isinstance(parser_output, dict)
    for name, rows in sorted(grouped.items()):
        temperature = fit_temperature(
            torch.tensor([score for score, _ in rows]),
            torch.tensor([correct for _, correct in rows]),
        )
        parser_output[name] = {
            "temperature": temperature,
            "prior": las[name] / total_las,
            "las": las[name],
            "count": len(rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

