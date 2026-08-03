from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import normalized_mutual_info_score

from sumoe.data.tasks import TASKS


@dataclass(frozen=True)
class RoutingReport:
    sample_count: int
    expert_loads: tuple[float, ...]
    expert_frequencies: tuple[float, ...]
    entropy_bits: float
    load_cv: float
    task_expert_nmi: float
    task_matrix: dict[str, tuple[float, ...]]
    family_matrix: dict[str, tuple[float, ...]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized_rows(
    labels: Sequence[str], preferred: NDArray[np.int64], num_experts: int
) -> dict[str, tuple[float, ...]]:
    rows: dict[str, tuple[float, ...]] = {}
    for label in sorted(set(labels)):
        mask = np.asarray([item == label for item in labels])
        counts = np.bincount(preferred[mask], minlength=num_experts).astype(float)
        rows[label] = tuple((counts / counts.sum()).tolist())
    return rows


def routing_statistics(
    assignments: torch.Tensor | Sequence[int],
    task_ids: Sequence[str],
    num_experts: int = 8,
) -> RoutingReport:
    tensor = torch.as_tensor(assignments)
    if tensor.ndim == 1:
        if tensor.numel() != len(task_ids):
            raise ValueError("assignment count must match task_ids")
        preferred = tensor.long()
        if preferred.numel() and (
            int(preferred.min()) < 0 or int(preferred.max()) >= num_experts
        ):
            raise ValueError("expert index lies outside [0, num_experts)")
        loads = torch.bincount(preferred, minlength=num_experts).double()
    elif tensor.ndim == 2:
        if tensor.shape != (len(task_ids), num_experts):
            raise ValueError("assignment matrix must have shape [samples, num_experts]")
        if (tensor < 0).any():
            raise ValueError("assignment weights must be non-negative")
        loads = tensor.double().sum(dim=0)
        preferred = tensor.argmax(dim=1)
    else:
        raise ValueError("assignments must be expert indices or a weight matrix")
    if not task_ids or float(loads.sum()) <= 0:
        raise ValueError("routing statistics require non-empty positive assignments")
    if any(task not in TASKS for task in task_ids):
        raise ValueError("task_ids contains a task outside the seven SCROLLS tasks")
    frequencies = loads / loads.sum()
    nonzero = frequencies[frequencies > 0]
    entropy = float(-(nonzero * torch.log2(nonzero)).sum())
    load_mean = float(loads.mean())
    load_cv = float(loads.std(unbiased=False) / load_mean) if load_mean else math.nan
    preferred_numpy: NDArray[np.int64] = preferred.cpu().numpy()
    task_numpy = np.asarray(task_ids)
    nmi = float(normalized_mutual_info_score(task_numpy, preferred_numpy))
    families = [TASKS[task].family for task in task_ids]
    return RoutingReport(
        sample_count=len(task_ids),
        expert_loads=tuple(loads.tolist()),
        expert_frequencies=tuple(frequencies.tolist()),
        entropy_bits=entropy,
        load_cv=load_cv,
        task_expert_nmi=nmi,
        task_matrix=_normalized_rows(task_ids, preferred_numpy, num_experts),
        family_matrix=_normalized_rows(families, preferred_numpy, num_experts),
    )
