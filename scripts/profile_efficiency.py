from __future__ import annotations

"""Command implementation kept separate to avoid shadowing Python's profile module."""

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sumoe.baselines.meo import initialize_meo_from_sumoe_state_dict
from sumoe.config import ExperimentConfig, load_config
from sumoe.data.scrolls import NormalizedExample
from sumoe.data.templates import render_source_text
from sumoe.evaluation.generation import PredictionCollator
from sumoe.evaluation.profiling import profile_generation, profile_prefill
from sumoe.forest.calibration import ParserCalibration, merge_and_select
from sumoe.forest.collate import ForestBatch
from sumoe.forest.parsers import SpacyParserAdapter, StanzaParserAdapter, TransitionParserAdapter
from sumoe.forest.parsers.base import DependencyParser, ParsedSentence
from sumoe.forest.transition.training import load_transition_checkpoint
from sumoe.forest.types import CandidateTree, DocumentForest, SentenceForest
from sumoe.model.factory import build_model
from sumoe.training.checkpoint import load_checkpoint_model_state


@dataclass(frozen=True)
class PreparedDocument:
    example: NormalizedExample
    input_length: int


@dataclass(frozen=True)
class ParsedDocument:
    prepared: PreparedDocument
    sentences: tuple[ParsedSentence, ...]
    candidates: tuple[tuple[CandidateTree, ...], ...]


class OnlineSumoeCondition:
    def __init__(
        self,
        config: ExperimentConfig,
        example: NormalizedExample,
        tokenizer: Any,
        model: torch.nn.Module,
        parsers: list[DependencyParser],
        calibration: dict[str, ParserCalibration],
        device: torch.device,
    ) -> None:
        self.config = config
        self.example = example
        self.tokenizer = tokenizer
        self.model = model
        self.parsers = parsers
        self.calibration = calibration
        self.device = device
        self.uses_forest = config.variant in {"llama_forest", "sumoe"}

    def prepare(self, length: int) -> PreparedDocument:
        empty_example = replace(self.example, source="")
        empty_prompt, _, _ = render_source_text(empty_example)
        prefix_tokens = len(self.tokenizer(empty_prompt, add_special_tokens=True)["input_ids"])
        source_budget = max(1, length - prefix_tokens)
        all_source_ids = self.tokenizer(
            self.example.source,
            add_special_tokens=False,
        )["input_ids"]
        if len(all_source_ids) < source_budget:
            raise ValueError(
                f"profiling source has {len(all_source_ids)} tokens; {source_budget} required"
            )
        source_ids = all_source_ids[:source_budget]
        source = self.tokenizer.decode(source_ids, skip_special_tokens=True)
        return PreparedDocument(replace(self.example, source=source), length)

    def parse(self, prepared: PreparedDocument) -> ParsedDocument:
        if not self.uses_forest:
            return ParsedDocument(prepared, (), ())
        sentences = self.parsers[0].segment_document(prepared.example.source)
        rows: list[tuple[CandidateTree, ...]] = []
        for sentence in sentences:
            candidates: list[CandidateTree] = []
            for parser in self.parsers:
                parsed = parser.parse_sentence(sentence)
                candidates.extend(
                    candidate
                    for candidate in parsed
                    if len(candidate.heads) == len(sentence.tokens)
                )
            if not candidates:
                raise RuntimeError("profiling parser ensemble produced no aligned candidates")
            rows.append(tuple(candidates))
        return ParsedDocument(prepared, sentences, tuple(rows))

    def _pad(self, batch: dict[str, Any], length: int) -> dict[str, Any]:
        current = batch["input_ids"].shape[1]
        if current > length:
            raise ValueError("profile collator exceeded requested input length")
        padding = length - current
        if padding:
            batch["input_ids"] = torch.nn.functional.pad(
                batch["input_ids"], (0, padding), value=int(self.tokenizer.pad_token_id)
            )
            batch["attention_mask"] = torch.nn.functional.pad(
                batch["attention_mask"], (0, padding), value=0
            )
            batch["source_mask"] = torch.nn.functional.pad(
                batch["source_mask"], (0, padding), value=False
            )
            if "forest_batch" in batch:
                batch["forest_batch"] = replace(batch["forest_batch"], sequence_length=length)
        return batch

    def build_forest(self, parsed: ParsedDocument) -> dict[str, Any]:
        if self.uses_forest:
            sentences = tuple(
                SentenceForest(
                    text_start=sentence.start,
                    text_end=sentence.end,
                    tokens=sentence.tokens,
                    candidates=merge_and_select(
                        candidates,
                        self.calibration,
                        self.config.forest.candidate_top_k,
                    ),
                )
                for sentence, candidates in zip(parsed.sentences, parsed.candidates, strict=True)
            )
            forest = DocumentForest(parsed.prepared.example.source_id, sentences)
            index = {parsed.prepared.example.source_id: forest}
        else:
            index = None
        requested = parsed.prepared.input_length
        batch = PredictionCollator(self.tokenizer, index, requested)([parsed.prepared.example])
        return self._pad(batch, requested)

    def run_model(self, prepared: PreparedDocument, forest: dict[str, Any]) -> torch.Tensor:
        inputs = {
            key: value.to(self.device)
            for key, value in forest.items()
            if key in {"input_ids", "attention_mask", "source_mask"}
        }
        forest_batch: ForestBatch | None = forest.get("forest_batch")
        output = self.model(
            **inputs,
            forest_batch=forest_batch.to(self.device) if forest_batch is not None else None,
            use_cache=False,
        )
        return output.logits

    def generate(
        self,
        prepared: PreparedDocument,
        forest: dict[str, Any],
        max_new_tokens: int,
    ) -> int:
        inputs = {
            key: value.to(self.device)
            for key, value in forest.items()
            if key in {"input_ids", "attention_mask", "source_mask"}
        }
        forest_batch: ForestBatch | None = forest.get("forest_batch")
        generated = self.model.generate(
            **inputs,
            forest_batch=forest_batch.to(self.device) if forest_batch is not None else None,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return int(generated.shape[1] - forest["input_ids"].shape[1])


def load_model_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    settings = getattr(model, "settings", None)
    expected_variant = getattr(settings, "variant", None)
    if expected_variant == "meo":
        state = load_checkpoint_model_state(checkpoint, "sumoe")
        initialize_meo_from_sumoe_state_dict(model, state)
    else:
        state = load_checkpoint_model_state(checkpoint, expected_variant)
        model.load_state_dict(state, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile parsing, forest, model, and end-to-end SUMoE"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--transition-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prefill", "generation"), default="prefill")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--generation-warmups", type=int, default=1)
    parser.add_argument("--generation-iterations", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite profile report: {args.output}")
    config = load_config(args.config, args.overlay)
    row = json.loads(args.example.read_text(encoding="utf-8").splitlines()[0])
    row["references"] = tuple(row["references"])
    example = NormalizedExample(**row)
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    calibration_payload = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration = {
        name: ParserCalibration(float(values["temperature"]), float(values["prior"]))
        for name, values in calibration_payload["parsers"].items()
    }
    transition = load_transition_checkpoint(args.transition_checkpoint)
    parsers: list[DependencyParser] = [
        StanzaParserAdapter(config.parsers.stanza_model),
        SpacyParserAdapter(
            config.parsers.spacy_model,
            config.parsers.spacy_beam_width,
            config.parsers.spacy_beam_density,
        ),
        TransitionParserAdapter(
            transition, n_best=5, beam_size=config.parsers.transition_beam_size
        ),
    ]
    model = build_model(
        config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    load_model_weights(model, args.checkpoint)
    device = torch.device("cuda")
    model.to(device).eval()
    condition = OnlineSumoeCondition(
        config, example, tokenizer, model, parsers, calibration, device
    )
    if args.mode == "prefill":
        report = profile_prefill(
            condition,
            warmups=config.evaluation.warmup_iterations,
            iterations=config.evaluation.timed_iterations,
        )
    else:
        report = profile_generation(
            condition,
            max_new_tokens=(
                config.evaluation.max_new_tokens
                if args.max_new_tokens is None
                else args.max_new_tokens
            ),
            warmups=args.generation_warmups,
            iterations=args.generation_iterations,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
