from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, cast

from sumoe.forest.io import sha256_file


def _configuration_dict(config: object) -> dict[str, Any]:
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(cast(Any, config))
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("config must be a dataclass or mapping")


def _hash_mapping(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in sorted(paths)}


@dataclass(frozen=True)
class RunManifest:
    configuration: dict[str, Any]
    configuration_sha256: str
    data_sha256: dict[str, str]
    forest_sha256: dict[str, str]
    python_version: str
    package_versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def build_run_manifest(
    config: object,
    data_files: Sequence[Path],
    forest_files: Sequence[Path],
) -> RunManifest:
    configuration = _configuration_dict(config)
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    packages = {}
    for name in ("torch", "transformers", "datasets", "accelerate", "deepspeed"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return RunManifest(
        configuration=configuration,
        configuration_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        data_sha256=_hash_mapping(data_files),
        forest_sha256=_hash_mapping(forest_files),
        python_version=platform.python_version(),
        package_versions=packages,
    )


def verify_resume_manifest(current: RunManifest, stored: RunManifest) -> None:
    if current.configuration_sha256 != stored.configuration_sha256:
        raise ValueError("configuration SHA-256 changed since the stored run")
    if current.data_sha256 != stored.data_sha256:
        raise ValueError("data SHA-256 changed since the stored run")
    if current.forest_sha256 != stored.forest_sha256:
        raise ValueError("forest SHA-256 changed since the stored run")
    if current.package_versions != stored.package_versions:
        raise ValueError("package versions changed since the stored run")
