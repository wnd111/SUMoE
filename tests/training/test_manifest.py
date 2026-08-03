from __future__ import annotations

from pathlib import Path

import pytest

from sumoe.training.manifest import build_run_manifest, verify_resume_manifest


def test_resume_rejects_changed_forest_hash(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    forest = tmp_path / "forest.jsonl"
    data.write_text("data", encoding="utf-8")
    forest.write_text("forest-a", encoding="utf-8")
    stored = build_run_manifest({"seed": 13}, [data], [forest])
    forest.write_text("forest-b", encoding="utf-8")
    current = build_run_manifest({"seed": 13}, [data], [forest])
    with pytest.raises(ValueError, match="forest SHA-256 changed"):
        verify_resume_manifest(current, stored)

