from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence

from torch.utils.data import Sampler

from .tasks import TASKS


class BalancedFamilySampler(Sampler[int]):
    """Uniformly balance task families, then cycle uniformly through their tasks."""

    def __init__(
        self,
        task_to_indices: Mapping[str, Sequence[int]],
        seed: int,
        num_replicas: int,
        rank: int,
    ) -> None:
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.task_to_indices = {
            task: tuple(indices) for task, indices in task_to_indices.items()
        }
        if any(task not in TASKS for task in self.task_to_indices):
            raise ValueError("task_to_indices contains an unsupported task")
        if any(not indices for indices in self.task_to_indices.values()):
            raise ValueError("every sampled task must contain at least one index")
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        family_sizes: dict[str, int] = defaultdict(int)
        for task, indices in self.task_to_indices.items():
            family_sizes[TASKS[task].family] += len(indices)
        required = {"summarization", "qa", "reasoning"}
        if set(family_sizes) != required:
            raise ValueError("sampler requires tasks from all three paper families")
        self.samples_per_family = max(family_sizes.values())
        global_size = 3 * self.samples_per_family
        self.total_size = math.ceil(global_size / num_replicas) * num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _global_order(self) -> list[int]:
        rng = random.Random(self.seed + self.epoch)
        family_tasks: dict[str, list[str]] = defaultdict(list)
        shuffled_indices: dict[str, list[int]] = {}
        cursors: dict[str, int] = {}
        for task, indices in self.task_to_indices.items():
            family_tasks[TASKS[task].family].append(task)
            shuffled_indices[task] = list(indices)
            rng.shuffle(shuffled_indices[task])
            cursors[task] = 0
        for tasks in family_tasks.values():
            tasks.sort()
            rng.shuffle(tasks)

        family_orders: dict[str, list[int]] = {}
        for family in ("summarization", "qa", "reasoning"):
            tasks = family_tasks[family]
            family_order: list[int] = []
            for position in range(self.samples_per_family):
                task = tasks[position % len(tasks)]
                choices = shuffled_indices[task]
                cursor = cursors[task]
                if cursor and cursor % len(choices) == 0:
                    rng.shuffle(choices)
                family_order.append(choices[cursor % len(choices)])
                cursors[task] = cursor + 1
            family_orders[family] = family_order

        order: list[int] = []
        for position in range(self.samples_per_family):
            families = ["summarization", "qa", "reasoning"]
            rng.shuffle(families)
            order.extend(family_orders[family][position] for family in families)
        while len(order) < self.total_size:
            order.append(order[len(order) % (3 * self.samples_per_family)])
        return order

    def __iter__(self) -> Iterator[int]:
        return iter(self._global_order()[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.total_size // self.num_replicas
