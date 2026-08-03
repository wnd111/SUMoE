from __future__ import annotations

import pytest

from sumoe.training.preflight import PreflightReport, enforce_preflight


def test_enforce_preflight_raises_with_all_failed_checks() -> None:
    report = PreflightReport(
        checks=("configuration: ok",), errors=("GPU count: expected 8, found 0",)
    )
    with pytest.raises(RuntimeError, match="expected 8"):
        enforce_preflight(report)
