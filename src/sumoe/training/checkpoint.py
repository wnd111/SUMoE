from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer  # type: ignore[attr-defined]

from .manifest import RunManifest

CHECKPOINT_ARCHITECTURE_VERSION = "sumoe-literal-eq4-eq5-v3"
ARCHITECTURE_METADATA_FILE = "architecture.json"


@dataclass(frozen=True)
class CheckpointState:
    optimizer_step: int
    epoch: int


def checkpoint_variant(model: nn.Module) -> str | None:
    settings = getattr(model, "settings", None)
    variant = getattr(settings, "variant", None)
    return str(variant) if variant is not None else None


def write_architecture_metadata(directory: Path, variant: str | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture_version": CHECKPOINT_ARCHITECTURE_VERSION,
        "variant": variant,
    }
    (directory / ARCHITECTURE_METADATA_FILE).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_architecture_metadata(
    directory: Path, expected_variant: str | None = None
) -> dict[str, str | None]:
    path = directory / ARCHITECTURE_METADATA_FILE
    if not path.is_file():
        raise RuntimeError(
            "Cannot use this checkpoint: architecture.json is missing. "
            "Legacy checkpoints are incompatible with the current SUMoE architecture."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("architecture_version")
    if version != CHECKPOINT_ARCHITECTURE_VERSION:
        raise RuntimeError(
            f"Cannot use checkpoint architecture {version!r}; expected "
            f"{CHECKPOINT_ARCHITECTURE_VERSION!r}."
        )
    stored_variant = payload.get("variant")
    if expected_variant is not None and stored_variant not in {None, expected_variant}:
        raise RuntimeError(
            f"Checkpoint variant {stored_variant!r} does not match {expected_variant!r}."
        )
    return {"architecture_version": str(version), "variant": stored_variant}


def resolve_checkpoint_directory(checkpoint: Path) -> Path:
    pointer = checkpoint / "last-checkpoint.txt"
    if pointer.is_file():
        return checkpoint / pointer.read_text(encoding="utf-8").strip()
    return checkpoint


def load_checkpoint_model_state(
    checkpoint: Path, expected_variant: str | None = None
) -> dict[str, torch.Tensor]:
    checkpoint = resolve_checkpoint_directory(checkpoint)
    validate_architecture_metadata(checkpoint, expected_variant)
    training_state = checkpoint / "training-state.pt"
    pytorch_model = checkpoint / "pytorch_model.bin"
    safetensors_model = checkpoint / "model.safetensors"
    if training_state.is_file():
        payload = torch.load(training_state, map_location="cpu", weights_only=False)
        if payload.get("architecture_version") != CHECKPOINT_ARCHITECTURE_VERSION:
            raise RuntimeError("training-state architecture version is missing or incompatible")
        state = payload["model"]
    elif pytorch_model.is_file():
        state = torch.load(pytorch_model, map_location="cpu", weights_only=True)
    elif safetensors_model.is_file():
        from safetensors.torch import load_file

        state = load_file(safetensors_model)
    else:
        raise FileNotFoundError(
            "checkpoint must contain training-state.pt, pytorch_model.bin, or model.safetensors"
        )
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise TypeError("checkpoint model state must map parameter names to tensors")
    return state


def save_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer_step: int,
    epoch: int,
    manifest: RunManifest | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_architecture_metadata(directory, checkpoint_variant(model))
    torch.save(
        {
            "architecture_version": CHECKPOINT_ARCHITECTURE_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "optimizer_step": optimizer_step,
            "epoch": epoch,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        directory / "training-state.pt",
    )
    if manifest is not None:
        manifest.save(directory / "run-manifest.json")


def load_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> CheckpointState:
    validate_architecture_metadata(directory, checkpoint_variant(model))
    payload = torch.load(directory / "training-state.pt", map_location="cpu", weights_only=False)
    if "architecture_version" not in payload:
        raise RuntimeError(
            "Cannot resume this checkpoint: legacy checkpoints use a different "
            "SUMoE architecture. Restart training with the current code."
        )
    if payload["architecture_version"] != CHECKPOINT_ARCHITECTURE_VERSION:
        raise RuntimeError(
            "Cannot resume checkpoint architecture "
            f"{payload['architecture_version']!r}; expected "
            f"{CHECKPOINT_ARCHITECTURE_VERSION!r}."
        )
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return CheckpointState(
        optimizer_step=int(payload["optimizer_step"]), epoch=int(payload["epoch"])
    )
