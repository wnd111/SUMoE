from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace

import torch

from sumoe.data.scrolls import NormalizedExample
from sumoe.evaluation.generation import PredictionCollator, greedy_generate


class RecordingTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def __call__(self, text: str, **_: object) -> dict[str, list[object]]:
        self.encoded_texts.append(text)
        matches = list(re.finditer(r"\S+", text))
        return {
            "input_ids": [1] + [10 + index for index in range(len(matches))],
            "offset_mapping": [(0, 0)] + [(m.start(), m.end()) for m in matches],
        }

    def batch_decode(self, rows: torch.Tensor, **_: object) -> list[str]:
        return ["decoded" for _ in rows]


class RecordingModel:
    def __init__(self) -> None:
        self.keys: set[str] = set()

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.keys = set(kwargs)
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        return torch.cat((input_ids, torch.tensor([[2]])), dim=1)


class ClassificationModel:
    def __init__(self, task_logits: torch.Tensor | None = None) -> None:
        self.keys: set[str] = set()
        self.task_logits = (
            task_logits if task_logits is not None else torch.tensor([[0.0, 1.0, 4.0, 2.0]])
        )

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.keys = set(kwargs)
        return SimpleNamespace(
            task_logits=self.task_logits,
            routing=None,
        )


def example() -> NormalizedExample:
    return NormalizedExample(
        example_id="qasper::validation::1::eval",
        source_id="qasper::validation::1",
        task="qasper",
        split="validation",
        family="qa",
        source="public source",
        target="SECRET_REFERENCE",
        references=("SECRET_REFERENCE",),
    )


def test_prediction_input_contains_no_reference_tokens() -> None:
    tokenizer = RecordingTokenizer()
    batch = PredictionCollator(tokenizer, forest_index=None, max_length=32)([example()])
    model = RecordingModel()
    result = greedy_generate(model, batch, tokenizer, max_new_tokens=16)
    assert all("SECRET_REFERENCE" not in text for text in tokenizer.encoded_texts)
    assert "references" not in model.keys
    assert result.predictions == ("decoded",)


def test_classification_prediction_uses_task_head_label() -> None:
    tokenizer = RecordingTokenizer()
    quality_source = """Which answer is supported?

(A) First answer.
(B) Second answer.
(C) The supported answer text.
(D) Fourth answer.


DOCUMENT
Supporting passage."""
    quality = replace(
        example(),
        example_id="quality::validation::1::eval",
        source_id="quality::validation::1",
        task="quality",
        family="reasoning",
        source=quality_source,
        target="The supported answer text.",
        references=("The supported answer text.",),
    )
    batch = PredictionCollator(tokenizer, forest_index=None, max_length=32)([quality])
    model = ClassificationModel()

    result = greedy_generate(model, batch, tokenizer, max_new_tokens=16)

    assert result.predictions == ("The supported answer text.",)
    assert result.generated_token_ids.shape == (1, 0)
    assert "references" not in model.keys


def test_contract_prediction_uses_dataset_native_label_spelling() -> None:
    tokenizer = RecordingTokenizer()
    contract_nli = replace(
        example(),
        example_id="contract_nli::validation::1::eval",
        source_id="contract_nli::validation::1",
        task="contract_nli",
        family="reasoning",
        target="Not mentioned",
        references=("Not mentioned",),
    )
    batch = PredictionCollator(tokenizer, forest_index=None, max_length=32)([contract_nli])
    model = ClassificationModel(torch.tensor([[0.0, 0.0, 1.0]]))

    result = greedy_generate(model, batch, tokenizer, max_new_tokens=16)

    assert result.predictions == ("Not mentioned",)
