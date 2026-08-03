from __future__ import annotations

import pytest

from sumoe.evaluation.significance import holm_adjust, paired_three_seed_test


def test_holm_adjustment_is_monotonic_in_sorted_order() -> None:
    assert holm_adjust([0.001, 0.04, 0.02]) == pytest.approx((0.003, 0.04, 0.04))


def test_three_seed_interval_uses_two_degrees_of_freedom() -> None:
    result = paired_three_seed_test([33.2, 33.8, 33.5], [34.8, 35.6, 35.1])
    assert result.mean_difference == pytest.approx(1.6666666667)
    assert result.degrees_of_freedom == 2
    assert result.confidence_low < result.mean_difference < result.confidence_high

