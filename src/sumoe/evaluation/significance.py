from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import mean, stdev
from typing import Any

from scipy import stats


@dataclass(frozen=True)
class PairedResult:
    mean_difference: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    t_statistic: float
    p_value: float
    degrees_of_freedom: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_three_seed_test(
    baseline: Sequence[float], sumoe: Sequence[float]
) -> PairedResult:
    if len(baseline) != 3 or len(sumoe) != 3:
        raise ValueError("the paper significance test requires exactly three paired seeds")
    differences = [right - left for left, right in zip(baseline, sumoe, strict=True)]
    difference_mean = mean(differences)
    standard_error = stdev(differences) / math.sqrt(3)
    degrees_of_freedom = 2
    critical = float(stats.t.ppf(0.975, degrees_of_freedom))
    if standard_error == 0:
        t_statistic = math.inf if difference_mean != 0 else 0.0
        p_value = 0.0 if difference_mean != 0 else 1.0
    else:
        test = stats.ttest_rel(sumoe, baseline)
        t_statistic = float(test.statistic)
        p_value = float(test.pvalue)
    margin = critical * standard_error
    return PairedResult(
        mean_difference=difference_mean,
        standard_error=standard_error,
        confidence_low=difference_mean - margin,
        confidence_high=difference_mean + margin,
        t_statistic=t_statistic,
        p_value=p_value,
        degrees_of_freedom=degrees_of_freedom,
    )


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must lie in [0, 1]")
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted_sorted: list[float] = []
    previous = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        previous = max(previous, candidate)
        adjusted_sorted.append(previous)
    adjusted = [0.0] * count
    for index, value in zip(order, adjusted_sorted, strict=True):
        adjusted[index] = value
    return tuple(adjusted)
