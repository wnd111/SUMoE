from __future__ import annotations

import torch

from sumoe.model.factory import build_tiny_model
from tests.model.test_sumoe_tiny import settings, tiny_forest, tiny_llama_config


def _generate_with_forward_count(
    model: torch.nn.Module, max_new_tokens: int, eos_token_id: int | None = None
) -> tuple[torch.Tensor, int]:
    prefix = torch.tensor([[1, 9, 10, 11]])
    forward_calls = 0

    def count_forward(_module: torch.nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal forward_calls
        forward_calls += 1

    handle = model.register_forward_pre_hook(count_forward)
    try:
        generated = model.generate(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            source_mask=torch.ones_like(prefix, dtype=torch.bool),
            forest_batch=tiny_forest(sequence_length=4),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
    finally:
        handle.remove()
    return generated, forward_calls


def test_cached_step_reuses_document_routing_without_forest() -> None:
    model = build_tiny_model(tiny_llama_config(), settings()).eval()
    prefix = torch.tensor([[1, 9, 10, 11]])
    with torch.no_grad():
        first = model(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            source_mask=torch.ones_like(prefix, dtype=torch.bool),
            forest_batch=tiny_forest(sequence_length=4),
            use_cache=True,
        )
        assert first.past_key_values is not None
        assert first.routing is not None
        second = model(
            input_ids=torch.tensor([[12]]),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            source_mask=torch.ones(1, 1, dtype=torch.bool),
            past_key_values=first.past_key_values,
            routing_cache=first.routing,
            expert_history=first.expert_history,
            use_cache=True,
        )
    assert second.logits.shape == (1, 1, 64)
    assert second.routing is first.routing


def test_cached_expert_attention_matches_full_prefix_forward() -> None:
    torch.manual_seed(31)
    model = build_tiny_model(tiny_llama_config(), settings()).eval()
    prefix = torch.tensor([[1, 9, 10, 11]])
    next_token = torch.tensor([[12]])
    with torch.no_grad():
        first = model(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            source_mask=torch.ones_like(prefix, dtype=torch.bool),
            forest_batch=tiny_forest(sequence_length=4),
            use_cache=True,
        )
        cached = model(
            input_ids=next_token,
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            source_mask=torch.zeros(1, 1, dtype=torch.bool),
            past_key_values=first.past_key_values,
            routing_cache=first.routing,
            expert_history=first.expert_history,
            use_cache=True,
        )
        full = model(
            input_ids=torch.cat((prefix, next_token), dim=1),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            source_mask=torch.tensor([[True, True, True, True, False]]),
            forest_batch=tiny_forest(sequence_length=5),
            use_cache=False,
        )

    torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], atol=1e-5, rtol=1e-5)


def test_generation_stops_without_an_unused_forward_at_limit_or_eos() -> None:
    torch.manual_seed(4)
    model = build_tiny_model(tiny_llama_config(), settings()).eval()

    first_token, first_token_forward_calls = _generate_with_forward_count(model, 1)
    generated, limit_forward_calls = _generate_with_forward_count(model, 3)
    repeated, repeated_forward_calls = _generate_with_forward_count(model, 3)
    eos_token_id = int(first_token[0, -1])
    stopped_at_eos, eos_forward_calls = _generate_with_forward_count(
        model, 4, eos_token_id=eos_token_id
    )

    assert first_token_forward_calls == 1
    assert limit_forward_calls == 3
    assert repeated_forward_calls == 3
    assert eos_forward_calls == 1
    assert torch.equal(repeated, generated)
    assert torch.equal(stopped_at_eos, first_token)
