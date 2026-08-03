from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_identity_is_paper_aligned_without_an_official_claim() -> None:
    identity_files = (
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "src/sumoe/__init__.py",
        ROOT / "scripts/build_release.py",
        ROOT / "scripts/train.py",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in identity_files).lower()
    stale_identity = re.compile(("inde" + "pendent") + r".{0,32}" + ("repro" + "duction"))

    assert "paper-aligned implementation" in combined
    official_claim = re.compile(("offi" + "cial") + r".{0,16}implementation")
    assert not official_claim.search(combined)
    assert not stale_identity.search(combined)
    assert ("独立完整" + "复现") not in combined
    assert ("未读取、未复制、" + "未导入") not in combined


def test_load_balance_stop_gradient_scope_is_explicitly_documented() -> None:
    notes = (ROOT / "docs/IMPLEMENTATION_NOTES.md").read_text(encoding="utf-8")

    assert "does not define a gradient estimator" in notes
    assert "router parameters only" in notes


def test_paper_significance_example_uses_llama_moe() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    command = next(line for line in readme.splitlines() if "collect_seed_scores.py" in line)
    assert "reports/llama_moe/" in command
    assert "reports/llama/" not in command


def test_readme_documents_non_distribution_policy_and_implementation_notes() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Model weights and training checkpoints are not distributed" in readme
    assert "docs/IMPLEMENTATION_NOTES.md" in readme
    assert (ROOT / "docs/IMPLEMENTATION_NOTES.md").is_file()


def test_every_readme_python_command_points_to_an_existing_script() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    scripts = re.findall(r"python\s+(scripts/[A-Za-z0-9_./-]+\.py)", readme)
    assert scripts
    assert all((ROOT / script).is_file() for script in scripts)


def test_readme_covers_every_reproduction_stage_without_placeholders() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required = [
        "prepare_scrolls.py",
        "prepare_ptb_dependencies.py",
        "train_transition_parser.py",
        "calibrate_parsers.py",
        "build_forests.py",
        "preflight.py",
        "train.py",
        "predict.py",
        "evaluate.py",
        "profile.py",
        "routing_statistics.py",
        "significance.py",
    ]
    assert all(item in readme for item in required)
    assert not re.search(r"TODO|TBD|待补充|占位符", readme, flags=re.IGNORECASE)


def test_profile_entrypoint_does_not_shadow_python_standard_library() -> None:
    command = (
        "import sys; "
        "sys.path.insert(0, 'scripts'); "
        "import cProfile; "
        "assert hasattr(cProfile, 'Profile')"
    )
    subprocess.run([sys.executable, "-c", command], cwd=ROOT, check=True)
