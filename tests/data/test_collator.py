from __future__ import annotations

import re
from dataclasses import replace

import torch

from sumoe.data.collator import SumoeDataCollator
from sumoe.data.scrolls import NormalizedExample
from sumoe.forest.types import CandidateTree, DocumentForest, SentenceForest, TokenSpan


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        matches = list(re.finditer(r"\S+", text))
        ids = [10 + index for index in range(len(matches))]
        offsets = [(match.start(), match.end()) for match in matches]
        if add_special_tokens:
            ids.insert(0, 1)
            offsets.insert(0, (0, 0))
        result: dict[str, list[int] | list[tuple[int, int]]] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def sample() -> NormalizedExample:
    return NormalizedExample(
        example_id="qasper::train::1::ref0",
        source_id="qasper::train::1",
        task="qasper",
        split="train",
        family="qa",
        source="alpha beta",
        target="short answer",
        references=("short answer",),
    )


def forest() -> DocumentForest:
    return DocumentForest(
        example_id="qasper::train::1",
        sentences=(
            SentenceForest(
                text_start=0,
                text_end=10,
                tokens=(TokenSpan("alpha", 0, 5), TokenSpan("beta", 6, 10)),
                candidates=(
                    CandidateTree(
                        parser="stanza",
                        heads=(1, -1),
                        labels=("dep", "root"),
                        raw_score=0.0,
                        posterior=1.0,
                    ),
                ),
            ),
        ),
    )


def test_collator_masks_every_non_target_token() -> None:
    batch = SumoeDataCollator(
        FakeTokenizer(), {"qasper::train::1": forest()}, max_length=64, max_target_length=8
    )([sample()])
    target_positions = batch["labels"] != -100
    assert target_positions.any()
    assert torch.all(batch["source_mask"][target_positions] == 0)
    assert batch["input_ids"].shape[1] <= 64
    assert batch["forest_batch"].batch_size == 1


def test_collator_builds_classification_labels_without_target_tokens() -> None:
    quality_source = """Why did the team stop?

(A) It ran out of fuel.
(B) The weather became unsafe.
(C) The mission was complete.
(D) The crew requested a rest.


DOCUMENT
The weather deteriorated rapidly."""
    example = replace(
        sample(),
        example_id="quality::train::1::ref0",
        source_id="quality::train::1",
        task="quality",
        family="reasoning",
        source=quality_source,
        target="The weather became unsafe.",
        references=("The weather became unsafe.",),
    )
    batch = SumoeDataCollator(
        FakeTokenizer(), forest_index=None, max_length=64, max_target_length=8
    )([example])

    assert batch["task_names"] == ("quality",)
    assert batch["classification_labels"].tolist() == [1]
    assert not batch["labels"].ne(-100).any()
