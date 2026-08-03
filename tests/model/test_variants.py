from __future__ import annotations

import pytest
import torch

from sumoe.baselines.dcc import DCCState
from sumoe.model.factory import build_tiny_model

from .test_sumoe_tiny import settings, tiny_forest, tiny_llama_config


@pytest.mark.parametrize(
    ("variant", "uses_forest", "uses_moe"),
    [
        ("llama", False, False),
        ("llama_forest", True, False),
        ("llama_moe", False, True),
        ("sumoe", True, True),
    ],
)
def test_controlled_variants_activate_only_intended_components(
    variant: str, uses_forest: bool, uses_moe: bool
) -> None:
    model = build_tiny_model(tiny_llama_config(), settings(variant))
    ids = torch.tensor([[1, 3, 4, 2]])
    output = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        source_mask=torch.ones_like(ids, dtype=torch.bool),
        forest_batch=tiny_forest(sequence_length=4) if uses_forest else None,
    )
    assert bool(model.injection_wrappers) is uses_forest
    assert (output.routing is not None) is uses_moe


def test_dcc_and_meo_execute_their_controlled_routing_paths() -> None:
    ids = torch.tensor([[1, 3, 4, 2]])
    mask = torch.ones_like(ids)
    dcc = build_tiny_model(tiny_llama_config(), settings("dcc")).eval()
    dcc.set_dcc_state(DCCState(torch.randn(4, 32)))
    dcc_output = dcc(input_ids=ids, attention_mask=mask, source_mask=mask.bool())
    meo = build_tiny_model(tiny_llama_config(), settings("meo"))
    meo_output = meo(input_ids=ids, attention_mask=mask, source_mask=mask.bool())
    assert dcc_output.routing is not None
    assert meo_output.routing is not None
    assert not dcc.injection_wrappers
    assert not meo.injection_wrappers


def test_dcc_training_routes_by_saved_source_partition() -> None:
    ids = torch.tensor([[1, 3, 4, 2]])
    mask = torch.ones_like(ids)
    dcc = build_tiny_model(tiny_llama_config(), settings("dcc"))
    dcc.set_dcc_state(
        DCCState(
            torch.randn(4, 32),
            source_ids=("doc-1",),
            training_assignments=torch.tensor([3]),
        )
    )

    output = dcc(
        input_ids=ids,
        attention_mask=mask,
        source_mask=mask.bool(),
        source_ids=("doc-1",),
    )

    assert output.routing is not None
    assert output.routing.indices.tolist() == [[3]]
    assert output.balance_loss is not None
    assert output.balance_loss.item() == 0.0
