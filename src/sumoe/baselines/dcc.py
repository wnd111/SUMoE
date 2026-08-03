from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.cluster import MiniBatchKMeans

from sumoe.model.router import RoutingOutput


@dataclass(frozen=True)
class DCCState:
    centroids: torch.Tensor
    source_ids: tuple[str, ...] = ()
    training_assignments: torch.Tensor | None = None
    seed: int = 13
    batch_size: int = 4096
    n_init: int = 10
    max_iter: int = 300

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "centroids": self.centroids.detach().cpu(),
                "source_ids": self.source_ids,
                "training_assignments": (
                    self.training_assignments.detach().cpu()
                    if self.training_assignments is not None
                    else None
                ),
                "seed": self.seed,
                "batch_size": self.batch_size,
                "n_init": self.n_init,
                "max_iter": self.max_iter,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, map_location: str | torch.device = "cpu") -> DCCState:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        return cls(
            centroids=payload["centroids"],
            source_ids=tuple(payload.get("source_ids", ())),
            training_assignments=payload.get("training_assignments"),
            seed=int(payload["seed"]),
            batch_size=int(payload["batch_size"]),
            n_init=int(payload["n_init"]),
            max_iter=int(payload["max_iter"]),
        )


def fit_dcc_centroids(
    embeddings: torch.Tensor,
    num_experts: int = 8,
    seed: int = 13,
    source_ids: tuple[str, ...] | None = None,
) -> DCCState:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [documents, hidden]")
    if embeddings.shape[0] < num_experts:
        raise ValueError("DCC requires at least num_experts documents")
    if source_ids is not None and len(source_ids) != embeddings.shape[0]:
        raise ValueError("source_ids must contain one identifier per embedding")
    estimator = MiniBatchKMeans(
        n_clusters=num_experts,
        batch_size=4096,
        n_init=10,
        max_iter=300,
        random_state=seed,
        reassignment_ratio=0.0,
    )
    estimator.fit(embeddings.detach().float().cpu().numpy())
    centroids = torch.from_numpy(estimator.cluster_centers_).to(dtype=embeddings.dtype)
    assignments = torch.from_numpy(estimator.labels_).to(dtype=torch.long)
    return DCCState(
        centroids=centroids,
        source_ids=source_ids or (),
        training_assignments=assignments if source_ids is not None else None,
        seed=seed,
    )


def route_dcc_training_assignments(
    source_ids: tuple[str, ...],
    state: DCCState,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> RoutingOutput:
    """Route training documents to their immutable precomputed DCC partition."""
    if state.training_assignments is None or not state.source_ids:
        raise ValueError("DCC state does not contain fixed training assignments")
    if len(state.source_ids) != state.training_assignments.shape[0]:
        raise ValueError("DCC source IDs and training assignments have different lengths")
    assignment_by_source = dict(
        zip(state.source_ids, state.training_assignments.tolist(), strict=True)
    )
    try:
        assigned = torch.tensor(
            [assignment_by_source[source_id] for source_id in source_ids],
            dtype=torch.long,
            device=device or state.centroids.device,
        )
    except KeyError as error:
        raise KeyError(f"DCC training source has no fixed partition: {error.args[0]}") from error
    num_experts = state.centroids.shape[0]
    if assigned.numel() and (assigned.min() < 0 or assigned.max() >= num_experts):
        raise ValueError("DCC training assignment lies outside the expert pool")
    dense = torch.zeros(
        (len(source_ids), num_experts),
        dtype=dtype or state.centroids.dtype,
        device=device or state.centroids.device,
    ).scatter(1, assigned[:, None], 1.0)
    hard_mask = dense.to(dtype=torch.bool)
    return RoutingOutput(
        logits=dense,
        probabilities=dense,
        indices=assigned[:, None],
        weights=torch.ones((len(source_ids), 1), dtype=dense.dtype, device=dense.device),
        dense_assignments=dense,
        selection_assignments=dense,
        hard_mask=hard_mask,
    )


def route_dcc(embeddings: torch.Tensor, state: DCCState, top_k: int = 2) -> RoutingOutput:
    if embeddings.ndim != 2 or state.centroids.ndim != 2:
        raise ValueError("embeddings and centroids must both be matrices")
    if embeddings.shape[1] != state.centroids.shape[1]:
        raise ValueError("embedding width does not match DCC centroid width")
    if not 1 <= top_k <= state.centroids.shape[0]:
        raise ValueError("top_k lies outside the available centroids")
    centroids = state.centroids.to(embeddings.device, embeddings.dtype)
    squared_distances = (embeddings[:, None, :] - centroids[None, :, :]).square().sum(-1)
    indices = torch.argsort(squared_distances, dim=-1, stable=True)[:, :top_k]
    selected_distances = squared_distances.gather(1, indices)
    inverse = selected_distances.add(1e-8).reciprocal()
    weights = inverse / inverse.sum(dim=-1, keepdim=True)
    dense = torch.zeros_like(squared_distances).scatter(1, indices, weights)
    hard_mask = torch.zeros_like(squared_distances, dtype=torch.bool).scatter(1, indices, True)
    probabilities = torch.softmax(-squared_distances, dim=-1)
    return RoutingOutput(
        logits=-squared_distances,
        probabilities=probabilities,
        indices=indices,
        weights=weights,
        dense_assignments=dense,
        selection_assignments=hard_mask.to(dtype=dense.dtype),
        hard_mask=hard_mask,
    )
