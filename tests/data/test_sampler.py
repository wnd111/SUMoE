from __future__ import annotations

from collections import Counter

from scripts.train import build_training_sampler
from sumoe.data.sampler import BalancedFamilySampler
from sumoe.data.tasks import TASKS


def index_map() -> dict[str, list[int]]:
    return {
        "gov_report": [0, 1],
        "summ_screen_fd": [2, 3],
        "qmsum": [4],
        "qasper": [5],
        "narrative_qa": [6, 7],
        "quality": [8, 9],
        "contract_nli": [10, 11],
    }


def test_family_sampler_is_balanced_and_seeded() -> None:
    mapping = index_map()
    order_a = list(BalancedFamilySampler(mapping, seed=13, num_replicas=1, rank=0))
    order_b = list(BalancedFamilySampler(mapping, seed=13, num_replicas=1, rank=0))
    assert order_a == order_b
    index_to_family = {
        index: TASKS[task].family for task, indices in mapping.items() for index in indices
    }
    counts = Counter(index_to_family[index] for index in order_a)
    assert counts == {"summarization": 4, "qa": 4, "reasoning": 4}


def test_distributed_sampler_shards_a_shared_global_order() -> None:
    mapping = index_map()
    left = list(BalancedFamilySampler(mapping, 21, 2, 0))
    right = list(BalancedFamilySampler(mapping, 21, 2, 1))
    assert len(left) == len(right) == 6


def test_paper_training_sampler_uses_each_multitask_example_once() -> None:
    examples = list(range(11))
    order = list(build_training_sampler(examples, seed=13))

    assert len(order) == len(examples)
    assert sorted(order) == list(range(len(examples)))
