from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import torch

from sumoe.config import ExperimentConfig
from sumoe.forest.io import verify_forest_file

REQUIRED_VERSIONS = {
    "torch": "2.4.1",
    "transformers": "4.46.3",
    "datasets": "2.21.0",
    "accelerate": "1.1.1",
    "deepspeed": "0.15.4",
    "stanza": "1.9.2",
    "spacy": "3.8.2",
    "scikit-learn": "1.5.2",
    "scipy": "1.14.1",
    "rouge-score": "0.1.2",
    "PyYAML": "6.0.2",
}


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


def enforce_preflight(report: PreflightReport) -> None:
    if report.errors:
        raise RuntimeError("preflight failed:\n- " + "\n- ".join(report.errors))


def run_preflight(config: ExperimentConfig, check_only: bool = False) -> PreflightReport:
    checks: list[str] = []
    errors: list[str] = []
    try:
        config.validate()
        checks.append("configuration: paper invariants satisfied")
    except ValueError as error:
        errors.append(f"configuration: {error}")
    if platform.python_version() != "3.10.14":
        errors.append(f"Python version: expected 3.10.14, found {platform.python_version()}")
    else:
        checks.append("Python version: 3.10.14")
    for package, required in REQUIRED_VERSIONS.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = "not-installed"
        if installed != required:
            errors.append(f"{package} version: expected {required}, found {installed}")
        else:
            checks.append(f"{package} version: {required}")

    gpu_count = torch.cuda.device_count()
    if gpu_count != 8:
        errors.append(f"GPU count: expected 8, found {gpu_count}")
    else:
        checks.append("GPU count: 8")
        for index in range(gpu_count):
            name = torch.cuda.get_device_name(index)
            memory_gib = torch.cuda.get_device_properties(index).total_memory / 2**30
            if "H20" not in name or memory_gib < 79:
                errors.append(
                    f"GPU {index}: expected NVIDIA H20 80GB, found {name} {memory_gib:.1f}GiB"
                )
        if not torch.cuda.is_bf16_supported():
            errors.append("BF16: CUDA devices do not report BF16 support")

    data_root = Path(config.data.data_dir)
    forest_root = Path(config.data.forest_dir)
    for task in config.data.tasks:
        data_path = data_root / task / "train.jsonl"
        forest_path = forest_root / task / "train.jsonl"
        if not data_path.is_file():
            errors.append(f"data file missing: {data_path}")
        else:
            checks.append(f"data file: {data_path}")
        if not forest_path.is_file():
            errors.append(f"forest file missing: {forest_path}")
        else:
            verification = verify_forest_file(forest_path)
            if not verification.valid:
                errors.append(f"forest SHA-256 sidecar mismatch: {forest_path}")
            else:
                checks.append(f"forest SHA-256: {forest_path}")
    report = PreflightReport(tuple(checks), tuple(errors))
    if not check_only:
        enforce_preflight(report)
    return report
