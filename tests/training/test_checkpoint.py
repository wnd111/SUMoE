from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sumoe.training.checkpoint import (
    load_checkpoint,
    load_checkpoint_model_state,
    save_checkpoint,
)


def test_checkpoint_restores_model_optimizer_scheduler_and_step(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    save_checkpoint(tmp_path / "step-7", model, optimizer, scheduler, 7, 2)
    assert (tmp_path / "step-7" / "architecture.json").is_file()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    state = load_checkpoint(tmp_path / "step-7", model, optimizer, scheduler)
    assert state.optimizer_step == 7
    assert state.epoch == 2
    assert all(torch.equal(original[name], value) for name, value in model.state_dict().items())


def test_legacy_checkpoint_fails_with_architecture_message(tmp_path: Path) -> None:
    directory = tmp_path / "legacy"
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    save_checkpoint(directory, model, optimizer, scheduler, 7, 2)
    path = directory / "training-state.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("architecture_version")
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="legacy checkpoints"):
        load_checkpoint(directory, model, optimizer, scheduler)


def test_prediction_state_rejects_checkpoint_without_architecture_metadata(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "legacy-consolidated"
    directory.mkdir()
    torch.save(torch.nn.Linear(3, 2).state_dict(), directory / "pytorch_model.bin")

    with pytest.raises(RuntimeError, match="architecture.json is missing"):
        load_checkpoint_model_state(directory)


def test_literal_equation_architecture_rejects_v2_forest_attention_state(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "v2-forest-attention"
    directory.mkdir()
    torch.save(torch.nn.Linear(3, 2).state_dict(), directory / "pytorch_model.bin")
    (directory / "architecture.json").write_text(
        '{"architecture_version":"sumoe-method-preserved-v2","variant":"sumoe"}\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="sumoe-literal-eq4-eq5-v3"):
        load_checkpoint_model_state(directory, expected_variant="sumoe")
