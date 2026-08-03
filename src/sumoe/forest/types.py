from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenSpan:
    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.text or self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid token span: {self.start}:{self.end} {self.text!r}")

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TokenSpan:
        return cls(text=str(data["text"]), start=int(data["start"]), end=int(data["end"]))


@dataclass(frozen=True)
class CandidateTree:
    parser: str
    heads: tuple[int, ...]
    labels: tuple[str, ...]
    raw_score: float
    posterior: float = 0.0

    def __post_init__(self) -> None:
        if not self.parser:
            raise ValueError("parser name must not be empty")
        if not self.heads or len(self.heads) != len(self.labels):
            raise ValueError("heads and labels must have the same nonzero length")
        roots = [index for index, head in enumerate(self.heads) if head == -1]
        if len(roots) != 1:
            raise ValueError("candidate tree must contain exactly one root")
        size = len(self.heads)
        for index, head in enumerate(self.heads):
            if head < -1 or head >= size:
                raise ValueError(f"head index {head} is invalid for token {index}")
            if head == index:
                raise ValueError(f"token {index} cannot be its own syntactic head")
        if not math.isfinite(self.raw_score):
            raise ValueError("raw_score must be finite")
        if not math.isfinite(self.posterior) or not 0.0 <= self.posterior <= 1.0:
            raise ValueError("posterior must be finite and within [0, 1]")
        for index in range(size):
            visited: set[int] = set()
            cursor = index
            while cursor != -1:
                if cursor in visited:
                    raise ValueError("candidate tree contains a dependency cycle")
                visited.add(cursor)
                cursor = self.heads[cursor]

    def with_posterior(self, posterior: float, parser: str | None = None) -> CandidateTree:
        return CandidateTree(
            parser=parser or self.parser,
            heads=self.heads,
            labels=self.labels,
            raw_score=self.raw_score,
            posterior=posterior,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "parser": self.parser,
            "heads": list(self.heads),
            "labels": list(self.labels),
            "raw_score": self.raw_score,
            "posterior": self.posterior,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateTree:
        return cls(
            parser=str(data["parser"]),
            heads=tuple(int(value) for value in data["heads"]),
            labels=tuple(str(value) for value in data["labels"]),
            raw_score=float(data["raw_score"]),
            posterior=float(data.get("posterior", 0.0)),
        )


@dataclass(frozen=True)
class SentenceForest:
    text_start: int
    text_end: int
    tokens: tuple[TokenSpan, ...]
    candidates: tuple[CandidateTree, ...]

    def __post_init__(self) -> None:
        if self.text_start < 0 or self.text_end <= self.text_start:
            raise ValueError("invalid sentence offsets")
        if not self.tokens or not self.candidates:
            raise ValueError("sentence forest requires tokens and candidates")
        token_count = len(self.tokens)
        if any(len(candidate.heads) != token_count for candidate in self.candidates):
            raise ValueError("candidate token count does not match sentence tokens")
        posterior_sum = sum(candidate.posterior for candidate in self.candidates)
        if posterior_sum > 0 and not math.isclose(posterior_sum, 1.0, abs_tol=1e-6):
            raise ValueError("candidate posterior probabilities must sum to one")

    def to_dict(self) -> dict[str, object]:
        return {
            "text_start": self.text_start,
            "text_end": self.text_end,
            "tokens": [token.to_dict() for token in self.tokens],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SentenceForest:
        return cls(
            text_start=int(data["text_start"]),
            text_end=int(data["text_end"]),
            tokens=tuple(TokenSpan.from_dict(item) for item in data["tokens"]),
            candidates=tuple(CandidateTree.from_dict(item) for item in data["candidates"]),
        )


@dataclass(frozen=True)
class DocumentForest:
    example_id: str
    sentences: tuple[SentenceForest, ...]

    def __post_init__(self) -> None:
        if not self.example_id or not self.sentences:
            raise ValueError("document forest requires an ID and at least one sentence")

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "sentences": [sentence.to_dict() for sentence in self.sentences],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DocumentForest:
        return cls(
            example_id=str(data["example_id"]),
            sentences=tuple(SentenceForest.from_dict(item) for item in data["sentences"]),
        )


@dataclass(frozen=True)
class ForestManifest:
    schema_version: int
    data_sha256: str
    tokenizer_revision: str
    parser_versions: Mapping[str, str] = field(default_factory=dict)
    parser_hashes: Mapping[str, str] = field(default_factory=dict)
    calibration_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("forest schema_version must equal 1")
        for name, digest in {
            "data_sha256": self.data_sha256,
            "calibration_sha256": self.calibration_sha256,
            **{f"parser_hashes.{key}": value for key, value in self.parser_hashes.items()},
        }.items():
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "data_sha256": self.data_sha256,
            "tokenizer_revision": self.tokenizer_revision,
            "parser_versions": dict(self.parser_versions),
            "parser_hashes": dict(self.parser_hashes),
            "calibration_sha256": self.calibration_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ForestManifest:
        return cls(
            schema_version=int(data["schema_version"]),
            data_sha256=str(data["data_sha256"]),
            tokenizer_revision=str(data["tokenizer_revision"]),
            parser_versions={str(k): str(v) for k, v in data["parser_versions"].items()},
            parser_hashes={str(k): str(v) for k, v in data["parser_hashes"].items()},
            calibration_sha256=str(data["calibration_sha256"]),
        )

