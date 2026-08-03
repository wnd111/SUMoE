from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW  # type: ignore[attr-defined]

from .actions import ActionVocabulary, ParserState, apply_action
from .beam import beam_parse
from .dataset import DependencySentence, ParserVocabulary, load_conllu
from .model import StackTransformerParser
from .oracle import oracle_actions


@dataclass(frozen=True)
class TransitionTrainResult:
    optimizer_steps: int
    best_dev_las: float
    checkpoint: Path


def build_vocabulary(sentences: Iterable[DependencySentence]) -> ParserVocabulary:
    rows = list(sentences)
    return ParserVocabulary.from_sequences(
        words=[row.words for row in rows],
        upos=[row.upos for row in rows],
        labels=[row.labels for row in rows],
    )


def oracle_states(
    sentence: DependencySentence,
) -> list[tuple[ParserState, object]]:
    actions = oracle_actions(sentence.heads, sentence.labels)
    state = ParserState.initial(len(sentence.words))
    examples: list[tuple[ParserState, object]] = []
    for action in actions:
        examples.append((state, action))
        state = apply_action(state, action)
    return examples


@torch.no_grad()
def labeled_attachment_score(
    model: StackTransformerParser, sentences: Iterable[DependencySentence]
) -> float:
    correct = 0
    total = 0
    model.eval()
    for sentence in sentences:
        prediction = beam_parse(model, sentence, beam_size=16, n_best=1)[0]
        for predicted_head, predicted_label, gold_head, gold_label in zip(
            prediction.heads,
            prediction.labels,
            sentence.heads,
            sentence.labels,
            strict=True,
        ):
            correct += int(predicted_head == gold_head and predicted_label == gold_label)
            total += 1
    return correct / total if total else 0.0


def train_transition_parser(
    config: Mapping[str, Any], smoke_test: bool = False
) -> TransitionTrainResult:
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    train_rows = load_conllu(Path(config["train"]))
    dev_rows = load_conllu(Path(config["dev"]))
    vocabulary = build_vocabulary(train_rows)
    actions = ActionVocabulary(vocabulary.labels)
    architecture_keys = (
        "model_dim",
        "word_dim",
        "upos_dim",
        "char_dim",
        "char_channels",
        "num_heads",
        "ffn_dim",
        "num_layers",
        "dropout",
    )
    architecture = {key: config[key] for key in architecture_keys}
    model = StackTransformerParser(vocabulary, actions, **architecture)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best.pt"
    best_las = -1.0
    patience = 0
    optimizer_steps = 0
    for epoch in range(int(config["max_epochs"])):
        order = list(range(len(train_rows)))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        for row_index in order:
            sentence = train_rows[row_index]
            for state, gold_action in oracle_states(sentence):
                optimizer.zero_grad(set_to_none=True)
                logits = model(model.feature_batch(sentence, state))
                target = torch.tensor([actions.id(gold_action)])  # type: ignore[arg-type]
                loss = F.cross_entropy(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite parser loss at sentence {row_index}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer_steps += 1
                if smoke_test:
                    best_las = 0.0
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "vocabulary": vocabulary.to_dict(),
                            "actions": actions.to_dict(),
                            "architecture": architecture,
                            "seed": seed,
                        },
                        checkpoint,
                    )
                    return TransitionTrainResult(optimizer_steps, best_las, checkpoint)
        dev_las = labeled_attachment_score(model, dev_rows)
        if dev_las > best_las:
            best_las = dev_las
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocabulary": vocabulary.to_dict(),
                    "actions": actions.to_dict(),
                    "architecture": architecture,
                    "seed": seed,
                    "dev_las": dev_las,
                },
                checkpoint,
            )
        else:
            patience += 1
            if patience >= int(config["patience"]):
                break
    return TransitionTrainResult(optimizer_steps, best_las, checkpoint)


def load_transition_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> StackTransformerParser:
    payload = torch.load(path, map_location=device, weights_only=False)
    vocabulary = ParserVocabulary.from_dict(payload["vocabulary"])
    actions = ActionVocabulary.from_dict(payload["actions"])
    model = StackTransformerParser(vocabulary, actions, **payload["architecture"])
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model
