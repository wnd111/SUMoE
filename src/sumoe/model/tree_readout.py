from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from sumoe.forest.collate import ForestBatch

from .segment_ops import segment_softmax, segment_sum


class SparseGATLayer(nn.Module):
    def __init__(self, node_size: int, num_heads: int = 4) -> None:
        super().__init__()
        if node_size % num_heads:
            raise ValueError("node_size must be divisible by num_heads")
        self.node_size = node_size
        self.num_heads = num_heads
        self.head_size = node_size // num_heads
        self.projection = nn.Linear(node_size, node_size, bias=False)
        self.attention_source = nn.Parameter(torch.empty(num_heads, self.head_size))
        self.attention_target = nn.Parameter(torch.empty(num_heads, self.head_size))
        self.output = nn.Linear(node_size, node_size, bias=False)
        self.norm = nn.LayerNorm(node_size)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)

    def forward(
        self, nodes: torch.Tensor, edge_source: torch.Tensor, edge_target: torch.Tensor
    ) -> torch.Tensor:
        transformed = self.projection(nodes).view(
            -1, self.num_heads, self.head_size
        )
        source = transformed.index_select(0, edge_source)
        target = transformed.index_select(0, edge_target)
        scores = F.leaky_relu(
            (source * self.attention_source).sum(-1)
            + (target * self.attention_target).sum(-1),
            negative_slope=0.2,
        ) / math.sqrt(self.head_size)
        weights = segment_softmax(scores, edge_target, nodes.shape[0])
        messages = source * weights.unsqueeze(-1)
        aggregated = nodes.new_zeros(
            (nodes.shape[0], self.num_heads, self.head_size)
        )
        aggregated.index_add_(0, edge_target, messages)
        update = self.output(aggregated.flatten(1))
        return self.norm(nodes + F.elu(update))


class ForestTreeReadout(nn.Module):
    """Posterior-weighted tree readout used as the router's structural input."""

    def __init__(
        self,
        hidden_size: int,
        node_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("the SUMoE forest readout uses exactly two GAT layers")
        self.node_size = node_size
        self.input_projection = nn.Linear(hidden_size, node_size)
        self.layers = nn.ModuleList(
            [SparseGATLayer(node_size, num_heads) for _ in range(num_layers)]
        )
        self.tree_pool = nn.Linear(2 * node_size, node_size)

    def forward(self, hidden_states: torch.Tensor, forest: ForestBatch) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if (batch_size, sequence_length) != (forest.batch_size, forest.sequence_length):
            raise ValueError("forest dimensions do not match hidden_states")
        device = hidden_states.device
        candidate_count = forest.candidate_posterior.numel()
        tree_dependent = forest.tree_edge_dependent.to(device)
        if tree_dependent.numel() == 0 or candidate_count == 0:
            return hidden_states.new_zeros((batch_size, self.node_size))

        tree_head = forest.tree_edge_head.to(device)
        tree_candidate = forest.tree_edge_candidate.to(device)
        total_tokens = batch_size * sequence_length
        dependent_codes = tree_candidate * total_tokens + tree_dependent
        head_codes = tree_candidate * total_tokens + tree_head
        node_codes = torch.unique(torch.cat((dependent_codes, head_codes)), sorted=True)
        edge_target = torch.searchsorted(node_codes, dependent_codes)
        edge_source = torch.searchsorted(node_codes, head_codes)
        node_tokens = torch.remainder(node_codes, total_tokens)
        node_candidates = torch.div(node_codes, total_tokens, rounding_mode="floor")

        flat_hidden = hidden_states.reshape(-1, hidden_size)
        nodes = self.input_projection(flat_hidden.index_select(0, node_tokens))
        for layer in self.layers:
            nodes = layer(nodes, edge_source, edge_target)

        counts = segment_sum(
            torch.ones((nodes.shape[0], 1), dtype=nodes.dtype, device=device),
            node_candidates,
            candidate_count,
        )
        means = segment_sum(nodes, node_candidates, candidate_count) / counts.clamp_min(1.0)
        root_edge_mask = tree_dependent.eq(tree_head)
        root_nodes = torch.unique(edge_target[root_edge_mask])
        root_candidates = node_candidates.index_select(0, root_nodes)
        roots = segment_sum(
            nodes.index_select(0, root_nodes), root_candidates, candidate_count
        )
        candidate_vectors = self.tree_pool(torch.cat((roots, means), dim=-1))

        posteriors = forest.candidate_posterior.to(device, hidden_states.dtype)
        candidate_sentences = forest.candidate_sentence.to(device)
        candidate_documents = forest.candidate_document.to(device)
        sentence_count = int(candidate_sentences.max().item()) + 1
        posterior_sums = segment_sum(
            posteriors.unsqueeze(-1), candidate_sentences, sentence_count
        )
        sentence_vectors = segment_sum(
            candidate_vectors * posteriors.unsqueeze(-1),
            candidate_sentences,
            sentence_count,
        ) / posterior_sums.clamp_min(1e-12)

        sentence_documents = torch.zeros(sentence_count, dtype=torch.long, device=device)
        sentence_documents.index_copy_(0, candidate_sentences, candidate_documents)
        document_counts = segment_sum(
            torch.ones((sentence_count, 1), dtype=hidden_states.dtype, device=device),
            sentence_documents,
            batch_size,
        )
        return segment_sum(
            sentence_vectors, sentence_documents, batch_size
        ) / document_counts.clamp_min(1.0)
