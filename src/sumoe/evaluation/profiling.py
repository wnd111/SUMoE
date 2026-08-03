from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Protocol

import torch


class ProfileCondition(Protocol):
    def prepare(self, length: int) -> Any:
        raise NotImplementedError

    def parse(self, prepared: Any) -> Any:
        raise NotImplementedError

    def build_forest(self, parsed: Any) -> Any:
        raise NotImplementedError

    def run_model(self, prepared: Any, forest: Any) -> Any:
        raise NotImplementedError


class GenerationProfileCondition(ProfileCondition, Protocol):
    def generate(self, prepared: Any, forest: Any, max_new_tokens: int) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class PrefillProfileMeasurement:
    input_length: int
    parse_ms: float
    forest_ms: float
    prefill_model_ms: float
    prefill_pipeline_ms: float
    input_tokens_per_second: float
    peak_memory_mib: float
    model_flops: int


@dataclass(frozen=True)
class PrefillProfileReport:
    warmup_iterations: int
    timed_iterations: int
    measurements: tuple[PrefillProfileMeasurement, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_iterations": self.warmup_iterations,
            "timed_iterations": self.timed_iterations,
            "measurements": [asdict(item) for item in self.measurements],
        }


@dataclass(frozen=True)
class GenerationProfileMeasurement:
    input_length: int
    generated_tokens: int
    time_to_first_token_ms: float
    generation_ms: float
    end_to_end_generation_ms: float
    generated_tokens_per_second: float
    peak_memory_mib: float


@dataclass(frozen=True)
class GenerationProfileReport:
    warmup_iterations: int
    timed_iterations: int
    measurements: tuple[GenerationProfileMeasurement, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_iterations": self.warmup_iterations,
            "timed_iterations": self.timed_iterations,
            "measurements": [asdict(item) for item in self.measurements],
        }


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure(function: Any, warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        function()
    timings: list[float] = []
    for _ in range(iterations):
        _synchronize()
        start = time.perf_counter_ns()
        function()
        _synchronize()
        timings.append((time.perf_counter_ns() - start) / 1_000_000)
    return mean(timings)


def _model_flops(condition: ProfileCondition, prepared: Any, forest: Any) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
        condition.run_model(prepared, forest)
    return int(sum(int(event.flops or 0) for event in profiler.key_averages()))


def _validate_profile_arguments(lengths: Sequence[int], warmups: int, iterations: int) -> None:
    if warmups < 0 or iterations < 1:
        raise ValueError("warmups must be non-negative and iterations must be positive")
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("all profiling lengths must be positive")


def profile_prefill(
    condition: ProfileCondition,
    lengths: Sequence[int] = (512, 1024, 2048, 4096),
    warmups: int = 20,
    iterations: int = 100,
) -> PrefillProfileReport:
    _validate_profile_arguments(lengths, warmups, iterations)
    measurements: list[PrefillProfileMeasurement] = []
    for length in lengths:
        prepared = condition.prepare(length)
        parsed = condition.parse(prepared)
        forest = condition.build_forest(parsed)
        parse_ms = _measure(lambda active=prepared: condition.parse(active), warmups, iterations)
        forest_ms = _measure(
            lambda active=parsed: condition.build_forest(active), warmups, iterations
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        prefill_model_ms = _measure(
            lambda active=prepared, active_forest=forest: condition.run_model(
                active, active_forest
            ),
            warmups,
            iterations,
        )

        def end_to_end(active: Any = prepared) -> Any:
            current_parsed = condition.parse(active)
            current_forest = condition.build_forest(current_parsed)
            return condition.run_model(active, current_forest)

        prefill_pipeline_ms = _measure(end_to_end, warmups, iterations)
        peak_memory = (
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        )
        flops = _model_flops(condition, prepared, forest)
        measurements.append(
            PrefillProfileMeasurement(
                input_length=int(length),
                parse_ms=parse_ms,
                forest_ms=forest_ms,
                prefill_model_ms=prefill_model_ms,
                prefill_pipeline_ms=prefill_pipeline_ms,
                input_tokens_per_second=1000.0 * length / max(prefill_pipeline_ms, 1e-12),
                peak_memory_mib=peak_memory,
                model_flops=flops,
            )
        )
    return PrefillProfileReport(warmups, iterations, tuple(measurements))


def profile_generation(
    condition: GenerationProfileCondition,
    lengths: Sequence[int] = (512, 1024, 2048, 4096),
    max_new_tokens: int = 512,
    warmups: int = 1,
    iterations: int = 5,
) -> GenerationProfileReport:
    _validate_profile_arguments(lengths, warmups, iterations)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    measurements: list[GenerationProfileMeasurement] = []
    for length in lengths:
        prepared = condition.prepare(length)
        parsed = condition.parse(prepared)
        forest = condition.build_forest(parsed)
        time_to_first_token_ms = _measure(
            lambda active=prepared, active_forest=forest: condition.generate(
                active, active_forest, 1
            ),
            warmups,
            iterations,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        generated_tokens = 0

        def generate(active: Any = prepared, active_forest: Any = forest) -> int:
            nonlocal generated_tokens
            generated_tokens = int(condition.generate(active, active_forest, max_new_tokens))
            return generated_tokens

        generation_ms = _measure(generate, warmups, iterations)

        def end_to_end(active: Any = prepared) -> int:
            current_parsed = condition.parse(active)
            current_forest = condition.build_forest(current_parsed)
            return int(condition.generate(active, current_forest, max_new_tokens))

        end_to_end_generation_ms = _measure(end_to_end, warmups, iterations)
        peak_memory = (
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        )
        measurements.append(
            GenerationProfileMeasurement(
                input_length=int(length),
                generated_tokens=generated_tokens,
                time_to_first_token_ms=time_to_first_token_ms,
                generation_ms=generation_ms,
                end_to_end_generation_ms=end_to_end_generation_ms,
                generated_tokens_per_second=1000.0 * generated_tokens / max(generation_ms, 1e-12),
                peak_memory_mib=peak_memory,
            )
        )
    return GenerationProfileReport(warmups, iterations, tuple(measurements))


profile_condition = profile_prefill

# Compatibility names for consumers that imported the original report types.
ProfileMeasurement = PrefillProfileMeasurement
ProfileReport = PrefillProfileReport
