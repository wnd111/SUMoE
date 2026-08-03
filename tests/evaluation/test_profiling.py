from __future__ import annotations

import torch

from sumoe.evaluation.profiling import (
    profile_condition,
    profile_generation,
    profile_prefill,
)


class TinyCondition:
    def prepare(self, length: int) -> torch.Tensor:
        return torch.ones(length)

    def parse(self, prepared: torch.Tensor) -> torch.Tensor:
        return prepared + 1

    def build_forest(self, parsed: torch.Tensor) -> torch.Tensor:
        return parsed * 2

    def run_model(self, prepared: torch.Tensor, forest: torch.Tensor) -> torch.Tensor:
        return prepared @ forest

    def generate(self, prepared: torch.Tensor, forest: torch.Tensor, max_new_tokens: int) -> int:
        return max_new_tokens


def test_prefill_profiler_uses_unambiguous_field_names() -> None:
    report = profile_prefill(TinyCondition(), lengths=(8,), warmups=0, iterations=2)
    item = report.measurements[0]
    assert item.prefill_model_ms >= 0
    assert item.prefill_pipeline_ms >= 0
    assert item.input_tokens_per_second > 0


def test_profile_condition_remains_a_prefill_compatibility_alias() -> None:
    assert profile_condition is profile_prefill


def test_generation_profiler_reports_completed_output_metrics() -> None:
    report = profile_generation(
        TinyCondition(), lengths=(8,), max_new_tokens=3, warmups=0, iterations=2
    )
    item = report.measurements[0]
    assert item.generated_tokens == 3
    assert item.time_to_first_token_ms >= 0
    assert item.generation_ms >= 0
    assert item.end_to_end_generation_ms >= item.generation_ms
    assert item.generated_tokens_per_second > 0
