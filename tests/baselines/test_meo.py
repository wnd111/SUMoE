from __future__ import annotations

import torch

from sumoe.baselines.meo import (
    explicit_merged_expert,
    functional_merged_expert,
    functional_merged_transformer_expert,
    initialize_meo_from_sumoe_state_dict,
    merge_swiglu_parameters,
)
from sumoe.model.experts import SwiGLUExpert, TransformerExpert
from sumoe.model.factory import build_tiny_model
from tests.model.test_sumoe_tiny import settings, tiny_llama_config


def experts() -> list[SwiGLUExpert]:
    torch.manual_seed(8)
    return [SwiGLUExpert(8, 12, dropout=0.0) for _ in range(3)]


def test_meo_functional_output_equals_explicit_parameter_merge() -> None:
    modules = experts()
    weights = torch.tensor([[0.2, 0.3, 0.5]])
    hidden = torch.randn(1, 4, 8)
    functional = functional_merged_expert(hidden, modules, weights)
    explicit = explicit_merged_expert(hidden, merge_swiglu_parameters(modules, weights))
    assert torch.allclose(functional, explicit, atol=1e-6)


def test_meo_does_not_modify_stored_expert_parameters() -> None:
    modules = experts()
    before = [parameter.detach().clone() for module in modules for parameter in module.parameters()]
    functional_merged_expert(torch.randn(2, 3, 8), modules, torch.softmax(torch.randn(2, 3), -1))
    after = [parameter.detach() for module in modules for parameter in module.parameters()]
    assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))


def test_transformer_meo_one_hot_merge_equals_selected_expert() -> None:
    torch.manual_seed(23)
    modules = [TransformerExpert(8, 12, num_heads=2, dropout=0.0) for _ in range(3)]
    hidden = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)

    merged = functional_merged_transformer_expert(
        hidden,
        modules,
        torch.tensor([[0.0, 1.0, 0.0]]),
        mask,
    )

    torch.testing.assert_close(merged, modules[1](hidden, mask))


def test_meo_initialization_imports_trained_sumoe_backbone_router_and_experts() -> None:
    torch.manual_seed(41)
    sumoe = build_tiny_model(tiny_llama_config(), settings("sumoe"))
    meo = build_tiny_model(tiny_llama_config(), settings("meo"))

    initialize_meo_from_sumoe_state_dict(meo, sumoe.state_dict())

    sumoe_state = sumoe.state_dict()
    for key, value in meo.state_dict().items():
        source_key = key
        if key.startswith("backbone.model.layers."):
            parts = key.split(".")
            source_key = ".".join((*parts[:4], "decoder_layer", *parts[4:]))
        assert torch.equal(value, sumoe_state[source_key]), key
