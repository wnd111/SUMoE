from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn


@dataclass(frozen=True)
class ParameterReport:
    backbone: int
    forest_injections: int
    tree_readout: int
    router: int
    experts: int
    task_heads: int
    total: int
    trainable: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def count_parameters(model: nn.Module) -> ParameterReport:
    """Count each unique model parameter once, grouped by its module path."""
    counts = {
        "backbone": 0,
        "forest_injections": 0,
        "tree_readout": 0,
        "router": 0,
        "experts": 0,
        "task_heads": 0,
    }
    trainable = 0
    seen_parameters: set[int] = set()

    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_id = id(parameter)
        if parameter_id in seen_parameters:
            continue
        seen_parameters.add(parameter_id)

        if ".injection." in name:
            category = "forest_injections"
        elif name.startswith("tree_readout."):
            category = "tree_readout"
        elif name.startswith("router."):
            category = "router"
        elif name.startswith("expert_pool."):
            category = "experts"
        elif name.startswith("classification_heads.") or name.startswith("span_head."):
            category = "task_heads"
        else:
            category = "backbone"

        counts[category] += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()

    total = sum(counts.values())
    return ParameterReport(
        backbone=counts["backbone"],
        forest_injections=counts["forest_injections"],
        tree_readout=counts["tree_readout"],
        router=counts["router"],
        experts=counts["experts"],
        task_heads=counts["task_heads"],
        total=total,
        trainable=trainable,
    )
