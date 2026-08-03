from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")
PAPER_TASKS = (
    "gov_report",
    "summ_screen_fd",
    "qmsum",
    "qasper",
    "narrative_qa",
    "quality",
    "contract_nli",
)
VARIANTS = {"llama", "llama_forest", "llama_moe", "dcc", "meo", "sumoe"}


def _strict_keys(name: str, data: Mapping[str, object], cls: type[Any]) -> None:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown {name} keys: {', '.join(unknown)}")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return value


@dataclass(frozen=True)
class ParserConfig:
    sources: tuple[str, ...] = ("stanza", "spacy", "transition")
    stanza_model: str = "en"
    spacy_model: str = "en_core_web_trf"
    transition_checkpoint: str = "checkpoints/transition/best.pt"
    calibration_file: str = "data/parser-calibration.json"
    spacy_beam_width: int = 16
    spacy_beam_density: float = 0.0001
    transition_beam_size: int = 16

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> ParserConfig:
        _strict_keys("parsers", data, cls)
        values = dict(data)
        if "sources" in values:
            values["sources"] = tuple(
                str(item) for item in _sequence(values["sources"], "parsers.sources")
            )
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ForestConfig:
    candidate_top_k: int = 5
    edge_temperature: float = 1.0
    forest_coefficient: float = 1.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> ForestConfig:
        _strict_keys("forest", data, cls)
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ModelConfig:
    base_model: str
    hidden_size: int = 4096
    num_layers: int = 32
    injection_layers: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32)
    num_experts: int = 8
    top_k: int = 2
    expert_heads: int = 8
    expert_intermediate_size: int = 4096
    expert_dropout: float = 0.1
    router_dim: int = 1024
    gat_node_dim: int = 256
    gat_heads: int = 4
    dcc_centroids_file: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> ModelConfig:
        _strict_keys("model", data, cls)
        values = dict(data)
        if "injection_layers" in values:
            values["injection_layers"] = tuple(
                int(str(value))
                for value in _sequence(values["injection_layers"], "model.injection_layers")
            )
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class DataConfig:
    tasks: tuple[str, ...] = PAPER_TASKS
    data_dir: str = "data/scrolls"
    forest_dir: str = "data/forests"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> DataConfig:
        _strict_keys("data", data, cls)
        tasks = tuple(str(item) for item in _sequence(data.get("tasks", PAPER_TASKS), "data.tasks"))
        return cls(
            tasks=tasks,
            data_dir=str(data.get("data_dir", "data/scrolls")),
            forest_dir=str(data.get("forest_dir", "data/forests")),
        )


@dataclass(frozen=True)
class TrainingConfig:
    global_batch_size: int = 64
    world_size: int = 8
    micro_batch_size: int = 1
    gradient_accumulation: int = 8
    learning_rate: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 0.01
    balance_coefficient: float = 0.05
    max_length: int = 4096
    max_target_length: int = 512
    seeds: tuple[int, ...] = (13, 21, 42)
    epochs: int = 3
    warmup_ratio: float = 0.03
    minimum_learning_rate: float = 3e-6
    max_grad_norm: float = 1.0
    save_steps: int = 500
    logging_steps: int = 100

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> TrainingConfig:
        _strict_keys("training", data, cls)
        values = dict(data)
        if "seeds" in values:
            values["seeds"] = tuple(
                int(str(value)) for value in _sequence(values["seeds"], "training.seeds")
            )
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class EvaluationConfig:
    max_new_tokens: int = 512
    num_beams: int = 1
    do_sample: bool = False
    warmup_iterations: int = 20
    timed_iterations: int = 100

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> EvaluationConfig:
        _strict_keys("evaluation", data, cls)
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    forest: ForestConfig
    training: TrainingConfig
    data: DataConfig
    evaluation: EvaluationConfig
    parsers: ParserConfig
    variant: str = "sumoe"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> ExperimentConfig:
        _strict_keys("root", data, cls)
        config = cls(
            model=ModelConfig.from_mapping(_mapping(data.get("model"), "model")),
            forest=ForestConfig.from_mapping(_mapping(data.get("forest"), "forest")),
            training=TrainingConfig.from_mapping(_mapping(data.get("training"), "training")),
            data=DataConfig.from_mapping(_mapping(data.get("data"), "data")),
            evaluation=EvaluationConfig.from_mapping(
                _mapping(data.get("evaluation"), "evaluation")
            ),
            parsers=ParserConfig.from_mapping(_mapping(data.get("parsers", {}), "parsers")),
            variant=str(data.get("variant", "sumoe")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        expected_injection_layers = (4, 8, 12, 16, 20, 24, 28, 32)
        if self.model.injection_layers != expected_injection_layers:
            raise ValueError(
                "injection_layers must equal the paper configuration [4, 8, 12, 16, 20, 24, 28, 32]"
            )
        if any(layer < 1 or layer > self.model.num_layers for layer in self.model.injection_layers):
            raise ValueError("injection layers must be within the decoder depth")
        if self.model.top_k < 1 or self.model.top_k > self.model.num_experts:
            raise ValueError("model top_k must be between 1 and num_experts")
        if self.model.expert_heads < 1 or (self.model.hidden_size % self.model.expert_heads):
            raise ValueError("expert_heads must divide hidden_size")
        if self.forest.candidate_top_k < 1:
            raise ValueError("forest candidate_top_k must be positive")
        if self.forest.edge_temperature <= 0:
            raise ValueError("edge_temperature must be positive")
        calculated = (
            self.training.world_size
            * self.training.micro_batch_size
            * self.training.gradient_accumulation
        )
        if calculated != self.training.global_batch_size or calculated != 64:
            raise ValueError("global batch size must equal 64")
        if self.training.seeds != (13, 21, 42):
            raise ValueError("training seeds must equal [13, 21, 42]")
        if tuple(self.data.tasks) != PAPER_TASKS:
            raise ValueError("data tasks must contain the seven SCROLLS tasks in paper order")
        if self.variant not in VARIANTS:
            raise ValueError(f"unsupported variant: {self.variant}")
        allowed_parsers = {"stanza", "spacy", "transition"}
        if not self.parsers.sources or not set(self.parsers.sources) <= allowed_parsers:
            raise ValueError("parser sources must be a non-empty subset of stanza/spacy/transition")
        if self.evaluation.num_beams != 1 or self.evaluation.do_sample:
            raise ValueError("evaluation must use deterministic greedy decoding")


def deep_merge(base: Mapping[str, object], overlay: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = deepcopy(dict(base))
    for key, value in overlay.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_yaml(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        value: Any = yaml.safe_load(stream)
    return _mapping(value, str(path))


def load_config(path: Path, overlays: Sequence[Path] = ()) -> ExperimentConfig:
    merged: Mapping[str, object] = _read_yaml(path)
    for overlay in overlays:
        merged = deep_merge(merged, _read_yaml(overlay))
    return ExperimentConfig.from_mapping(merged)
