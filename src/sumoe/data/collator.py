from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from sumoe.forest.alignment import AlignedDocumentForest, align_document_forest
from sumoe.forest.collate import collate_aligned_forests
from sumoe.forest.types import DocumentForest

from .scrolls import NormalizedExample
from .tasks import TASKS, classification_label_index
from .templates import render_source_text


class SumoeDataCollator:
    """Right-truncate sources after reserving target capacity and align forests."""

    def __init__(
        self,
        tokenizer: Any,
        forest_index: Mapping[str, DocumentForest] | None,
        max_length: int = 4096,
        max_target_length: int = 512,
    ) -> None:
        if max_target_length < 2 or max_target_length >= max_length:
            raise ValueError("max_target_length must be in [2, max_length)")
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be set")
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id must be set")
        self.tokenizer = tokenizer
        self.forest_index = forest_index
        self.max_length = max_length
        self.max_target_length = max_target_length

    def _encode_one(
        self, example: NormalizedExample
    ) -> tuple[list[int], list[int], list[bool], int, AlignedDocumentForest | None]:
        source_text, source_start, source_end = render_source_text(example)
        encoded = self.tokenizer(source_text, add_special_tokens=True, return_offsets_mapping=True)
        source_ids = [int(item) for item in encoded["input_ids"]]
        offsets = [tuple(map(int, item)) for item in encoded["offset_mapping"]]
        task = TASKS[example.task]
        classification_label = -100
        if task.prediction_head == "classification":
            target_ids: list[int] = []
            classification_label = classification_label_index(
                example.task, example.target, example.source
            )
        else:
            target_encoded = self.tokenizer(example.target, add_special_tokens=False)
            target_ids = [int(item) for item in target_encoded["input_ids"]]
            target_ids = target_ids[: self.max_target_length - 1]
            target_ids.append(int(self.tokenizer.eos_token_id))
        available_source = self.max_length - len(target_ids)
        if available_source < 1:
            raise ValueError("target reservation leaves no room for a source token")
        source_ids = source_ids[:available_source]
        offsets = offsets[:available_source]

        source_token_mask: list[bool] = []
        relative_offsets: list[tuple[int, int]] = []
        for start, end in offsets:
            overlaps = end > start and min(end, source_end) > max(start, source_start)
            source_token_mask.append(overlaps)
            if overlaps:
                relative_offsets.append(
                    (max(0, start - source_start), min(source_end, end) - source_start)
                )
            else:
                relative_offsets.append((0, 0))
        if not any(source_token_mask):
            raise ValueError(f"source was fully truncated for {example.example_id}")

        input_ids = source_ids + target_ids
        labels = [-100] * len(source_ids) + target_ids
        source_mask = source_token_mask + [False] * len(target_ids)
        aligned = None
        if self.forest_index is not None:
            if example.source_id not in self.forest_index:
                raise KeyError(f"missing cached forest for {example.source_id}")
            relative_offsets.extend([(0, 0)] * len(target_ids))
            aligned = align_document_forest(
                self.forest_index[example.source_id], relative_offsets, source_mask
            )
        return input_ids, labels, source_mask, classification_label, aligned

    def __call__(self, examples: Sequence[NormalizedExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        encoded = [self._encode_one(example) for example in examples]
        batch_length = max(len(item[0]) for item in encoded)
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        source_rows: list[list[bool]] = []
        classification_labels: list[int] = []
        aligned_forests: list[AlignedDocumentForest] = []
        for input_ids, labels, source_mask, classification_label, aligned in encoded:
            padding = batch_length - len(input_ids)
            input_rows.append(input_ids + [int(self.tokenizer.pad_token_id)] * padding)
            label_rows.append(labels + [-100] * padding)
            attention_rows.append([1] * len(input_ids) + [0] * padding)
            source_rows.append(source_mask + [False] * padding)
            classification_labels.append(classification_label)
            if aligned is not None:
                aligned_forests.append(aligned)
        batch: dict[str, Any] = {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "source_mask": torch.tensor(source_rows, dtype=torch.bool),
            "classification_labels": torch.tensor(classification_labels, dtype=torch.long),
            "task_names": tuple(example.task for example in examples),
            "example_ids": [example.example_id for example in examples],
            "source_ids": [example.source_id for example in examples],
        }
        if self.forest_index is not None:
            batch["forest_batch"] = collate_aligned_forests(
                aligned_forests, sequence_length=batch_length
            )
        return batch
