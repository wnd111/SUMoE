from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from sumoe.training.trainer import SumoeTrainer, TrainerSettings


class TinyRoutedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = torch.nn.Linear(4, 4)
        self.expert_pool = torch.nn.Linear(4, 4)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> SimpleNamespace:
        hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
        logits = self.expert_pool(torch.tanh(self.router(hidden)))
        loss = torch.nn.functional.mse_loss(logits, labels.float())
        return SimpleNamespace(loss=loss)


class TinyEffectiveBalanceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = torch.nn.Linear(2, 2, bias=False)
        self.seen_effective_batch_sizes: list[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        defer_balance_loss: bool = False,
    ) -> SimpleNamespace:
        assert defer_balance_loss
        semantic = torch.nn.functional.one_hot(input_ids[:, 0], num_classes=2).float()
        prediction = self.router(semantic)
        return SimpleNamespace(
            loss=torch.nn.functional.mse_loss(prediction, labels.float()),
            routing_semantic=semantic.detach(),
            routing_structural=torch.zeros_like(semantic),
        )

    def effective_batch_balance_loss(
        self, semantic: torch.Tensor, structural: torch.Tensor
    ) -> torch.Tensor:
        self.seen_effective_batch_sizes.append(semantic.shape[0])
        return 0.01 * self.router(semantic + structural).square().mean()


def test_tiny_trainer_updates_router_and_expert_parameters(tmp_path: Path) -> None:
    model = TinyRoutedModel()
    rows = [{"input_ids": torch.tensor([0, 1]), "labels": torch.zeros(2, 4)} for _ in range(2)]
    loader = DataLoader(rows, batch_size=1)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    trainer = SumoeTrainer(
        model,
        loader,
        tmp_path,
        TrainerSettings(epochs=1, gradient_accumulation=1, save_steps=1, logging_steps=1),
    )
    result = trainer.train(max_optimizer_steps=1)
    assert result.optimizer_steps == 1
    assert math.isfinite(result.last_loss)
    assert (tmp_path / "last-checkpoint.txt").read_text(encoding="utf-8").strip() == (
        "checkpoint-1"
    )
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())


def test_trainer_computes_balance_once_over_accumulated_documents(tmp_path: Path) -> None:
    model = TinyEffectiveBalanceModel()
    rows = [{"input_ids": torch.tensor([index]), "labels": torch.zeros(2)} for index in (0, 1)]
    trainer = SumoeTrainer(
        model,
        DataLoader(rows, batch_size=1),
        tmp_path,
        TrainerSettings(epochs=1, gradient_accumulation=2, save_steps=10, logging_steps=10),
    )

    result = trainer.train(max_optimizer_steps=1)

    assert result.optimizer_steps == 1
    assert model.seen_effective_batch_sizes == [2]
