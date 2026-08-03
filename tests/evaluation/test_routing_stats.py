from __future__ import annotations

import pytest
import torch

from sumoe.evaluation.routing_stats import routing_statistics


def test_uniform_eight_expert_entropy_is_three_bits() -> None:
    assignments = torch.eye(8)
    tasks = [
        "gov_report",
        "summ_screen_fd",
        "qmsum",
        "qasper",
        "narrative_qa",
        "quality",
        "contract_nli",
        "contract_nli",
    ]
    report = routing_statistics(assignments, tasks, num_experts=8)
    assert report.entropy_bits == pytest.approx(3.0)
    assert report.load_cv == pytest.approx(0.0)

