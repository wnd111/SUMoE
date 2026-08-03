from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sumoe.forest.parsers import SpacyParserAdapter, StanzaParserAdapter, TransitionParserAdapter
from sumoe.forest.transition.dataset import load_conllu
from sumoe.forest.transition.training import load_transition_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect parser scores and labeled attachment on PTB dev"
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--transition-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration rows: {args.output}")
    transition_model = load_transition_checkpoint(args.transition_checkpoint)
    stanza = StanzaParserAdapter()
    adapters = [
        stanza,
        SpacyParserAdapter(),
        TransitionParserAdapter(transition_model, n_best=5, beam_size=16),
    ]
    sentences = load_conllu(args.gold)
    counts = {adapter.name: 0 for adapter in adapters}
    skipped = {adapter.name: 0 for adapter in adapters}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for sentence_index, gold in enumerate(sentences):
            fixed = TransitionParserAdapter(transition_model).make_sentence(
                gold.words, gold.upos, 0
            )
            for adapter in adapters:
                try:
                    candidates = adapter.parse_sentence(fixed)
                except (ValueError, RuntimeError):
                    skipped[adapter.name] += 1
                    continue
                for candidate_index, candidate in enumerate(candidates):
                    correct = sum(
                        predicted_head == gold_head and predicted_label == gold_label
                        for predicted_head, gold_head, predicted_label, gold_label in zip(
                            candidate.heads,
                            gold.heads,
                            candidate.labels,
                            gold.labels,
                            strict=True,
                        )
                    ) / len(gold.words)
                    stream.write(
                        json.dumps(
                            {
                                "sentence": sentence_index,
                                "candidate": candidate_index,
                                "parser": adapter.name,
                                "score": candidate.raw_score,
                                "correct": correct,
                            }
                        )
                        + "\n"
                    )
                    counts[adapter.name] += 1
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"calibration produced no candidates for a parser: {counts}")
    args.output.with_suffix(args.output.suffix + ".report.json").write_text(
        json.dumps({"candidates": counts, "skipped_sentences": skipped}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
