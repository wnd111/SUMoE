from __future__ import annotations

import torch
from transformers import LlamaConfig

from sumoe.forest.collate import ForestBatch
from sumoe.model.factory import build_tiny_model
from sumoe.model.sumoe_model import SumoeModelSettings


def tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        _attn_implementation="eager",
    )


def settings(variant: str = "sumoe") -> SumoeModelSettings:
    return SumoeModelSettings(
        injection_layers=(1, 2, 3, 4),
        gat_node_size=16,
        gat_heads=4,
        router_size=16,
        num_experts=4,
        top_k=2,
        expert_intermediate_size=32,
        expert_dropout=0.0,
        variant=variant,
    )


def tiny_forest(sequence_length: int = 6) -> ForestBatch:
    return ForestBatch(
        batch_size=1,
        sequence_length=sequence_length,
        edge_dependent=torch.tensor([0, 1, 2, 3]),
        edge_head=torch.tensor([0, 0, 1, 2]),
        edge_marginal=torch.ones(4),
        edge_document=torch.zeros(4, dtype=torch.long),
        edge_sentence=torch.zeros(4, dtype=torch.long),
        tree_edge_dependent=torch.tensor([0, 1, 2, 3]),
        tree_edge_head=torch.tensor([0, 0, 1, 2]),
        tree_edge_candidate=torch.zeros(4, dtype=torch.long),
        tree_edge_sentence=torch.zeros(4, dtype=torch.long),
        candidate_posterior=torch.ones(1),
        candidate_document=torch.zeros(1, dtype=torch.long),
        candidate_sentence=torch.zeros(1, dtype=torch.long),
    )


def test_sumoe_tiny_forward_computes_all_three_losses() -> None:
    torch.manual_seed(4)
    model = build_tiny_model(tiny_llama_config(), settings())
    input_ids = torch.tensor([[1, 5, 6, 7, 8, 2]])
    attention_mask = torch.ones_like(input_ids)
    source_mask = torch.tensor([[True, True, True, True, False, False]])
    labels = torch.tensor([[-100, -100, -100, -100, 8, 2]])
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        source_mask=source_mask,
        labels=labels,
        forest_batch=tiny_forest(),
    )
    assert output.logits.shape == (1, 6, 64)
    assert output.routing is not None
    assert output.routing.indices.shape == (1, 2)
    for loss in (output.loss, output.lm_loss, output.balance_loss, output.forest_loss):
        assert loss is not None
        assert torch.isfinite(loss)
    assert torch.allclose(
        output.loss,
        output.lm_loss + output.balance_loss + output.forest_loss,
    )
    output.loss.backward()


def test_deferred_balance_exposes_detached_effective_batch_summaries() -> None:
    model = build_tiny_model(tiny_llama_config(), settings())
    input_ids = torch.tensor([[1, 5, 6, 7, 8, 2]])
    source_mask = torch.tensor([[True, True, True, True, False, False]])
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        source_mask=source_mask,
        labels=torch.tensor([[-100, -100, -100, -100, 8, 2]]),
        forest_batch=tiny_forest(),
        defer_balance_loss=True,
    )

    assert output.routing_semantic is not None
    assert output.routing_structural is not None
    assert not output.routing_semantic.requires_grad
    assert not output.routing_structural.requires_grad
    assert output.balance_loss is not None
    assert output.balance_loss.item() == 0.0
    assert output.loss is not None
    assert torch.allclose(output.loss, output.task_loss + output.forest_loss)
    effective = model.effective_batch_balance_loss(
        output.routing_semantic, output.routing_structural
    )
    assert torch.isfinite(effective)


def test_classification_task_uses_its_prediction_head() -> None:
    torch.manual_seed(5)
    model = build_tiny_model(tiny_llama_config(), settings())
    input_ids = torch.tensor([[1, 5, 6, 7, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    source_mask = attention_mask.bool()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        source_mask=source_mask,
        labels=torch.full_like(input_ids, -100),
        task_names=("quality",),
        classification_labels=torch.tensor([2]),
        forest_batch=tiny_forest(),
    )

    assert output.task_logits is not None
    assert output.task_logits.shape == (1, 4)
    assert output.classification_loss is not None
    assert output.classification_loss > 0
    assert output.loss is not None
    output.loss.backward()


def test_span_positions_use_span_selection_head() -> None:
    model = build_tiny_model(tiny_llama_config(), settings())
    input_ids = torch.tensor([[1, 5, 6, 7, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        source_mask=attention_mask.bool(),
        span_start_positions=torch.tensor([1]),
        span_end_positions=torch.tensor([3]),
        forest_batch=tiny_forest(),
    )

    assert output.span_logits is not None
    assert output.span_logits.shape == (1, 6, 2)
    assert output.span_loss is not None
    assert output.span_loss > 0
    assert output.loss is not None
