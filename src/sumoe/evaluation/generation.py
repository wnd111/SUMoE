from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from sumoe.data.scrolls import NormalizedExample
from sumoe.data.tasks import TASKS, classification_label_text
from sumoe.data.templates import render_source_text
from sumoe.forest.alignment import AlignedDocumentForest, align_document_forest
from sumoe.forest.collate import collate_aligned_forests
from sumoe.forest.types import DocumentForest


@dataclass(frozen=True)
class GenerationResult:
    example_ids: tuple[str, ...]
    predictions: tuple[str, ...]
    generated_token_ids: torch.Tensor
    routing_assignments: tuple[tuple[float, ...], ...] | None


class PredictionCollator:
    """Encode source-only prompts; references remain metadata and are never tokenized."""

    def __init__(
        self,
        tokenizer: Any,
        forest_index: Mapping[str, DocumentForest] | None,
        max_length: int = 4096,
    ) -> None:
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("prediction tokenizer requires pad and EOS token ids")
        self.tokenizer = tokenizer
        self.forest_index = forest_index
        self.max_length = max_length

    def _encode_one(
        self, example: NormalizedExample
    ) -> tuple[list[int], list[bool], AlignedDocumentForest | None]:
        text, source_start, source_end = render_source_text(example)
        encoded = self.tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        input_ids = [int(item) for item in encoded["input_ids"]][: self.max_length]
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]][: self.max_length]
        source_mask: list[bool] = []
        relative_offsets: list[tuple[int, int]] = []
        for start, end in offsets:
            overlaps = end > start and min(end, source_end) > max(start, source_start)
            source_mask.append(overlaps)
            relative_offsets.append(
                (
                    max(0, start - source_start),
                    min(source_end, end) - source_start,
                )
                if overlaps
                else (0, 0)
            )
        if not any(source_mask):
            raise ValueError(f"source was fully truncated for {example.example_id}")
        aligned = None
        if self.forest_index is not None:
            if example.source_id not in self.forest_index:
                raise KeyError(f"missing cached forest for {example.source_id}")
            aligned = align_document_forest(
                self.forest_index[example.source_id], relative_offsets, source_mask
            )
        return input_ids, source_mask, aligned

    def __call__(self, examples: Sequence[NormalizedExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("cannot collate an empty prediction batch")
        rows = [self._encode_one(example) for example in examples]
        length = max(len(row[0]) for row in rows)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        source_mask: list[list[bool]] = []
        forests: list[AlignedDocumentForest] = []
        for ids, source, forest in rows:
            padding = length - len(ids)
            input_ids.append(ids + [int(self.tokenizer.pad_token_id)] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            source_mask.append(source + [False] * padding)
            if forest is not None:
                forests.append(forest)
        batch: dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "source_mask": torch.tensor(source_mask, dtype=torch.bool),
            "example_ids": tuple(example.example_id for example in examples),
            "tasks": tuple(example.task for example in examples),
            "sources": tuple(example.source for example in examples),
            "references": tuple(example.references for example in examples),
        }
        if self.forest_index is not None:
            batch["forest_batch"] = collate_aligned_forests(forests, length)
        return batch


@torch.no_grad()
def greedy_generate(
    model: Any,
    batch: Mapping[str, Any],
    tokenizer: Any,
    max_new_tokens: int = 512,
) -> GenerationResult:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    input_ids = batch["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("batch input_ids must be a tensor")
    tasks = tuple(str(task) for task in batch["tasks"])
    prediction_heads = {TASKS[task].prediction_head for task in tasks}
    if len(prediction_heads) != 1:
        raise ValueError("a prediction batch must use one task-head type")
    if prediction_heads == {"classification"}:
        output = model(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"],
            source_mask=batch["source_mask"],
            forest_batch=batch.get("forest_batch"),
            task_names=tasks,
        )
        if output.task_logits is None:
            raise RuntimeError("classification model returned no task logits")
        class_indices = output.task_logits.argmax(dim=-1).tolist()
        sources = tuple(str(source) for source in batch["sources"])
        predictions = tuple(
            classification_label_text(task, int(index), source)
            for task, index, source in zip(tasks, class_indices, sources, strict=True)
        )
        generated_ids = input_ids.new_empty((input_ids.shape[0], 0))
        routing = output.routing
    else:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"],
            source_mask=batch["source_mask"],
            forest_batch=batch.get("forest_batch"),
            max_new_tokens=max_new_tokens,
            eos_token_id=int(tokenizer.eos_token_id),
        )
        generated_ids = generated[:, input_ids.shape[1] :]
        predictions = tuple(
            text.strip()
            for text in tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        routing = getattr(model, "last_generation_routing", None)
    routing_assignments = None
    if routing is not None:
        routing_assignments = tuple(
            tuple(float(value) for value in row)
            for row in routing.dense_assignments.detach().float().cpu().tolist()
        )
    return GenerationResult(
        example_ids=tuple(str(item) for item in batch["example_ids"]),
        predictions=predictions,
        generated_token_ids=generated_ids,
        routing_assignments=routing_assignments,
    )
