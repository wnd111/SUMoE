from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sumoe.forest.builder import build_document_forest
from sumoe.forest.calibration import ParserCalibration
from sumoe.forest.io import sha256_file, write_forests
from sumoe.forest.parsers import SpacyParserAdapter, StanzaParserAdapter, TransitionParserAdapter
from sumoe.forest.transition.training import load_transition_checkpoint
from sumoe.forest.types import ForestManifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached SUMoE dependency forests")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--transition-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("stanza", "spacy", "transition"),
        default=("stanza", "spacy", "transition"),
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    calibration_data = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibrations = {
        name: ParserCalibration(float(values["temperature"]), float(values["prior"]))
        for name, values in calibration_data["parsers"].items()
    }
    transition_model = load_transition_checkpoint(args.transition_checkpoint)
    segmenter = StanzaParserAdapter()
    available = {
        "stanza": segmenter,
        "spacy": SpacyParserAdapter(),
        "transition": TransitionParserAdapter(transition_model),
    }
    parsers = [available[name] for name in args.sources]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    results = [
        build_document_forest(
            row,
            parsers,
            calibrations,
            tokenizer,
            top_k=args.top_k,
            segmenter=segmenter,
        )
        for row in rows
    ]
    manifest = ForestManifest(
        schema_version=1,
        data_sha256=sha256_file(args.input),
        tokenizer_revision=str(getattr(tokenizer, "init_kwargs", {}).get("revision", "main")),
        parser_versions={"stanza": "1.9.2", "spacy": "3.8.2", "transition": "1"},
        parser_hashes={"transition": sha256_file(args.transition_checkpoint)},
        calibration_sha256=sha256_file(args.calibration),
    )
    write_forests(args.output, [result.forest for result in results], manifest)
    report = {
        "documents": len(results),
        "parse_seconds": sum(result.parse_seconds for result in results),
        "forest_seconds": sum(result.forest_seconds for result in results),
        "diagnostics": [
            diagnostic.__dict__ for result in results for diagnostic in result.diagnostics
        ],
    }
    args.output.with_suffix(args.output.suffix + ".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
