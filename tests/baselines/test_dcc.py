from __future__ import annotations

import torch

from sumoe.baselines.dcc import (
    DCCState,
    fit_dcc_centroids,
    route_dcc,
    route_dcc_training_assignments,
)


def test_dcc_routes_to_two_nearest_centroids() -> None:
    state = DCCState(centroids=torch.tensor([[0.0], [1.0], [3.0]]))
    routing = route_dcc(torch.tensor([[1.2]]), state, top_k=2)
    assert routing.indices.tolist() == [[1, 0]]
    assert torch.allclose(routing.weights.sum(-1), torch.ones(1))


def test_dcc_fit_is_reproducible_for_fixed_seed() -> None:
    embeddings = torch.tensor([[0.0], [0.1], [5.0], [5.1]])
    source_ids = ("a", "b", "c", "d")
    left = fit_dcc_centroids(embeddings, num_experts=2, seed=13, source_ids=source_ids)
    right = fit_dcc_centroids(embeddings, num_experts=2, seed=13, source_ids=source_ids)
    assert torch.equal(left.centroids, right.centroids)
    assert left.source_ids == source_ids
    assert torch.equal(left.training_assignments, right.training_assignments)


def test_dcc_training_uses_saved_fixed_partition() -> None:
    state = DCCState(
        centroids=torch.tensor([[0.0], [10.0], [20.0]]),
        source_ids=("doc-a", "doc-b"),
        training_assignments=torch.tensor([2, 0]),
    )

    routing = route_dcc_training_assignments(("doc-b", "doc-a"), state)

    assert routing.indices.tolist() == [[0], [2]]
    assert routing.dense_assignments.tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
