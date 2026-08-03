from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW  # type: ignore[attr-defined]

from .checkpoint import (
    checkpoint_variant,
    load_checkpoint,
    save_checkpoint,
    validate_architecture_metadata,
    write_architecture_metadata,
)
from .manifest import RunManifest, verify_resume_manifest


@dataclass(frozen=True)
class TrainerSettings:
    learning_rate: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    epochs: int = 3
    gradient_accumulation: int = 8
    warmup_ratio: float = 0.03
    minimum_learning_rate: float = 3e-6
    max_grad_norm: float = 1.0
    save_steps: int = 500
    logging_steps: int = 100
    distributed: bool = False


@dataclass(frozen=True)
class TrainResult:
    optimizer_steps: int
    last_loss: float
    checkpoint: Path | None


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if hasattr(value, "to") and value.__class__.__name__ == "ForestBatch":
        return value.to(device)
    return value


def _model_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    ignored = {"example_ids", "references", "tasks"}
    return {key: _move(value, device) for key, value in batch.items() if key not in ignored}


class SumoeTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: Iterable[Mapping[str, Any]],
        output_dir: Path,
        settings: TrainerSettings | None = None,
        manifest: RunManifest | None = None,
    ) -> None:
        self.model = model
        self.supports_effective_batch_balance = (
            callable(getattr(model, "effective_batch_balance_loss", None))
            and getattr(model, "router", None) is not None
        )
        self.train_dataloader = train_dataloader
        self.output_dir = output_dir
        settings = settings or TrainerSettings()
        self.settings = settings
        self.manifest = manifest
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if manifest is not None:
            manifest.save(self.output_dir / "run-manifest.json")

        self.accelerator: Any | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            betas=(settings.beta1, settings.beta2),
            eps=settings.epsilon,
            weight_decay=settings.weight_decay,
        )
        try:
            batches_per_epoch = len(train_dataloader)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError("train_dataloader must define __len__") from error
        self.steps_per_epoch = math.ceil(batches_per_epoch / settings.gradient_accumulation)
        self.total_steps = max(1, self.steps_per_epoch * settings.epochs)
        warmup_steps = int(self.total_steps * settings.warmup_ratio)
        minimum_ratio = settings.minimum_learning_rate / settings.learning_rate

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            denominator = max(1, self.total_steps - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        if settings.distributed:
            from accelerate import Accelerator, DeepSpeedPlugin

            plugin = DeepSpeedPlugin(
                zero_stage=3,
                gradient_accumulation_steps=settings.gradient_accumulation,
                gradient_clipping=settings.max_grad_norm,
                zero3_init_flag=True,
                zero3_save_16bit_model=True,
            )
            self.accelerator = Accelerator(
                gradient_accumulation_steps=settings.gradient_accumulation,
                mixed_precision="bf16",
                deepspeed_plugin=plugin,
            )
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.scheduler
            )
            self.device = self.accelerator.device
        else:
            self.model.to(self.device)

    def _effective_batch_balance(
        self,
        semantic_summaries: list[torch.Tensor],
        structural_summaries: list[torch.Tensor],
    ) -> torch.Tensor | None:
        if not semantic_summaries:
            return None
        model = (
            self.accelerator.unwrap_model(self.model)
            if self.accelerator is not None
            else self.model
        )
        method = getattr(model, "effective_batch_balance_loss", None)
        if not callable(method):
            raise RuntimeError("model lost its effective-batch balance interface")
        return method(
            torch.cat(semantic_summaries, dim=0),
            torch.cat(structural_summaries, dim=0),
        )

    @staticmethod
    def _append_routing_summaries(
        output: Any,
        semantic_summaries: list[torch.Tensor],
        structural_summaries: list[torch.Tensor],
    ) -> None:
        semantic = getattr(output, "routing_semantic", None)
        structural = getattr(output, "routing_structural", None)
        if semantic is None or structural is None:
            raise RuntimeError("deferred balance requires routing summaries from every batch")
        semantic_summaries.append(semantic)
        structural_summaries.append(structural)

    def _finite_or_fail(self, loss: torch.Tensor, batch: Mapping[str, Any]) -> None:
        finite = torch.isfinite(loss.detach()).all()
        if self.accelerator is not None:
            finite_value = self.accelerator.reduce(finite.to(dtype=torch.int32), reduction="min")
            finite = finite_value.bool()
        if not bool(finite.item()):
            payload = {
                "error": "non-finite loss",
                "example_ids": list(batch.get("example_ids", [])),
            }
            (self.output_dir / "non-finite-batch.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            raise FloatingPointError("non-finite training loss")

    def _save(self, step: int, epoch: int) -> Path:
        checkpoint = self.output_dir / f"checkpoint-{step}"
        if self.accelerator is not None:
            self.accelerator.save_state(str(checkpoint))
            if self.accelerator.is_main_process:
                base_model = self.accelerator.unwrap_model(self.model)
                write_architecture_metadata(checkpoint, checkpoint_variant(base_model))
                (checkpoint / "trainer-state.json").write_text(
                    json.dumps({"optimizer_step": step, "epoch": epoch}) + "\n",
                    encoding="utf-8",
                )
                if self.manifest is not None:
                    self.manifest.save(checkpoint / "run-manifest.json")
            self.accelerator.wait_for_everyone()
        else:
            save_checkpoint(
                checkpoint,
                self.model,
                self.optimizer,
                self.scheduler,
                step,
                epoch,
                self.manifest,
            )
        if self.accelerator is None or self.accelerator.is_main_process:
            (self.output_dir / "last-checkpoint.txt").write_text(
                checkpoint.name + "\n", encoding="utf-8"
            )
        return checkpoint

    def train(
        self,
        resume_from_checkpoint: Path | None = None,
        max_optimizer_steps: int | None = None,
    ) -> TrainResult:
        optimizer_step = 0
        start_epoch = 0
        if resume_from_checkpoint is not None:
            if self.manifest is not None:
                stored = RunManifest.load(resume_from_checkpoint / "run-manifest.json")
                verify_resume_manifest(self.manifest, stored)
            if self.accelerator is not None:
                base_model = self.accelerator.unwrap_model(self.model)
                validate_architecture_metadata(
                    resume_from_checkpoint, checkpoint_variant(base_model)
                )
                self.accelerator.load_state(str(resume_from_checkpoint))
                state = json.loads(
                    (resume_from_checkpoint / "trainer-state.json").read_text(encoding="utf-8")
                )
                optimizer_step = int(state["optimizer_step"])
                start_epoch = int(state["epoch"])
            else:
                state = load_checkpoint(
                    resume_from_checkpoint, self.model, self.optimizer, self.scheduler
                )
                optimizer_step, start_epoch = state.optimizer_step, state.epoch

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        last_loss = math.nan
        last_checkpoint: Path | None = None
        pending_semantic: list[torch.Tensor] = []
        pending_structural: list[torch.Tensor] = []
        for epoch in range(start_epoch, self.settings.epochs):
            if hasattr(self.train_dataloader, "sampler") and hasattr(
                self.train_dataloader.sampler, "set_epoch"
            ):
                self.train_dataloader.sampler.set_epoch(epoch)
            total_batches = len(self.train_dataloader)  # type: ignore[arg-type]
            for batch_index, batch in enumerate(self.train_dataloader):
                inputs = _model_inputs(batch, self.device)
                if self.supports_effective_batch_balance:
                    inputs["defer_balance_loss"] = True
                if self.accelerator is not None:
                    with self.accelerator.accumulate(self.model):
                        output = self.model(**inputs)
                        if output.loss is None:
                            raise ValueError("model returned no training loss")
                        if self.supports_effective_batch_balance:
                            self._append_routing_summaries(
                                output, pending_semantic, pending_structural
                            )
                        balance_loss = (
                            self._effective_batch_balance(pending_semantic, pending_structural)
                            if self.accelerator.sync_gradients
                            else None
                        )
                        reported_loss = output.loss
                        backward_loss = output.loss
                        if balance_loss is not None:
                            reported_loss = reported_loss + balance_loss
                            backward_loss = backward_loss + (
                                balance_loss * self.settings.gradient_accumulation
                            )
                        self._finite_or_fail(reported_loss, batch)
                        last_loss = float(reported_loss.detach().float().item())
                        self.accelerator.backward(backward_loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(), self.settings.max_grad_norm
                            )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    should_step = self.accelerator.sync_gradients
                else:
                    output = self.model(**inputs)
                    if output.loss is None:
                        raise ValueError("model returned no training loss")
                    if self.supports_effective_batch_balance:
                        self._append_routing_summaries(output, pending_semantic, pending_structural)
                    should_step = (
                        batch_index + 1
                    ) % self.settings.gradient_accumulation == 0 or batch_index + 1 == total_batches
                    balance_loss = (
                        self._effective_batch_balance(pending_semantic, pending_structural)
                        if should_step
                        else None
                    )
                    reported_loss = output.loss
                    backward_loss = output.loss / self.settings.gradient_accumulation
                    if balance_loss is not None:
                        reported_loss = reported_loss + balance_loss
                        backward_loss = backward_loss + balance_loss
                    self._finite_or_fail(reported_loss, batch)
                    last_loss = float(reported_loss.detach().float().item())
                    backward_loss.backward()
                    if should_step:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.settings.max_grad_norm
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                if not should_step:
                    continue
                pending_semantic.clear()
                pending_structural.clear()
                optimizer_step += 1
                if optimizer_step % self.settings.logging_steps == 0 and (
                    self.accelerator is None or self.accelerator.is_main_process
                ):
                    print(
                        json.dumps(
                            {
                                "optimizer_step": optimizer_step,
                                "epoch": epoch,
                                "loss": last_loss,
                                "learning_rate": self.scheduler.get_last_lr()[0],
                            }
                        )
                    )
                if optimizer_step % self.settings.save_steps == 0:
                    last_checkpoint = self._save(optimizer_step, epoch)
                if max_optimizer_steps is not None and optimizer_step >= max_optimizer_steps:
                    if (
                        last_checkpoint is None
                        or last_checkpoint.name != f"checkpoint-{optimizer_step}"
                    ):
                        last_checkpoint = self._save(optimizer_step, epoch)
                    return TrainResult(optimizer_step, last_loss, last_checkpoint)
        if optimizer_step and (
            last_checkpoint is None or last_checkpoint.name != f"checkpoint-{optimizer_step}"
        ):
            last_checkpoint = self._save(optimizer_step, self.settings.epochs)
        return TrainResult(optimizer_step, last_loss, last_checkpoint)
