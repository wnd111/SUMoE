from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.build_release import build_release, iter_public_files, parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_release_cli_defaults_to_the_finley_archive_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "sys.argv", ["build_release.py", "--output", str(tmp_path / "release.zip")]
    )

    args = parse_args()

    assert args.archive_root == "SUMoE-Finley-paper-aligned-code"


def create_fixture_tree(root: Path) -> None:
    files = {
        "README.md": "public readme",
        "src/sumoe/__init__.py": "",
        "configs/model.yaml": "model: sumoe",
        "docs/guide.md": "public guide",
        "docs/superpowers/plan.md": "internal plan",
        "outputs/run/metrics.json": "{}",
        "nested/OUTPUTS/log.txt": "generated log",
        ".git/config": "private git state",
        "data/example.jsonl": "licensed data",
        "weights/model.pt": "checkpoint",
        "weights/model.safetensors": "checkpoint",
        "artifacts/source.zip": "archive",
    }
    for relative_path, content in files.items():
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


def test_release_inventory_excludes_private_and_generated_paths(tmp_path: Path) -> None:
    """A missing exclusion must not leak internal, generated, or model files."""
    create_fixture_tree(tmp_path)

    paths = {path.as_posix() for path in iter_public_files(tmp_path)}

    assert "README.md" in paths
    assert "src/sumoe/__init__.py" in paths
    assert not any("docs/superpowers" in path for path in paths)
    assert not any("outputs" in path.casefold() for path in paths)
    assert not any(path.endswith((".pt", ".safetensors", ".zip")) for path in paths)


def test_release_inventory_uses_only_declared_public_roots_and_files(
    tmp_path: Path,
) -> None:
    """An unlisted top-level file or directory must never enter a release."""
    required_files = {
        ".gitattributes",
        ".gitignore",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
    }
    required_roots = {".github", "configs", "docs", "scripts", "src", "tests"}
    for relative_path in required_files:
        (tmp_path / relative_path).write_text("public", encoding="utf-8")
    for root in required_roots:
        destination = tmp_path / root / "kept.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("public", encoding="utf-8")
    secret = tmp_path / "deployment-secret.env"
    secret.write_text("TOKEN=private", encoding="utf-8")
    parser = tmp_path / "third_party/stanford-corenlp/parser.jar"
    parser.parent.mkdir(parents=True)
    parser.write_text("downloaded parser", encoding="utf-8")

    paths = {path.as_posix() for path in iter_public_files(tmp_path)}

    assert required_files <= paths
    assert {f"{root}/kept.txt" for root in required_roots} <= paths
    assert "deployment-secret.env" not in paths
    assert not any(path.startswith("third_party/") for path in paths)


def test_release_zip_uses_stable_root_and_sorted_members(tmp_path: Path) -> None:
    """A release archive must have one stable root and lexical member order."""
    repository = tmp_path / "repo"
    create_fixture_tree(repository)
    output = tmp_path / "release.zip"

    build_release(repository, output, "SUMoE-Finley-paper-aligned-code")

    with ZipFile(output) as archive:
        names = archive.namelist()
    assert names == sorted(names)
    assert all(name.startswith("SUMoE-Finley-paper-aligned-code/") for name in names)


def test_release_zip_is_byte_reproducible_with_explicit_metadata(tmp_path: Path) -> None:
    """Release bytes and host metadata must not depend on the build machine."""
    repository = tmp_path / "repo"
    create_fixture_tree(repository)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    build_release(repository, first, "SUMoE-Finley-paper-aligned-code")
    build_release(repository, second, "SUMoE-Finley-paper-aligned-code")

    assert first.read_bytes() == second.read_bytes()
    with ZipFile(first) as archive:
        metadata = archive.infolist()
    assert metadata
    assert all(member.create_system == 3 for member in metadata)


def test_release_builder_refuses_to_overwrite_an_existing_archive(tmp_path: Path) -> None:
    """An existing output must remain unchanged rather than being overwritten."""
    repository = tmp_path / "repo"
    create_fixture_tree(repository)
    output = tmp_path / "release.zip"
    original = b"do not replace"
    output.write_bytes(original)

    with pytest.raises(FileExistsError):
        build_release(repository, output, "SUMoE-Finley-paper-aligned-code")

    assert output.read_bytes() == original


@pytest.mark.parametrize("dangling", [False, True], ids=["valid", "dangling"])
def test_release_builder_refuses_output_symlinks(tmp_path: Path, dangling: bool) -> None:
    """Valid and dangling output symlinks must never redirect archive writes."""
    repository = tmp_path / "repo"
    create_fixture_tree(repository)
    target = tmp_path / "symlink-target.zip"
    if not dangling:
        target.write_bytes(b"existing target")
    output = tmp_path / "release.zip"
    try:
        output.symlink_to(target)
    except OSError as error:
        pytest.skip(f"OS refused symlink creation: {error}")

    with pytest.raises(FileExistsError):
        build_release(repository, output, "SUMoE-Finley-paper-aligned-code")

    assert output.is_symlink()
    if dangling:
        assert not target.exists()
    else:
        assert target.read_bytes() == b"existing target"


def test_release_builder_rejects_a_multicomponent_archive_root(tmp_path: Path) -> None:
    """A path-like archive root would create an ambiguous public archive layout."""
    repository = tmp_path / "repo"
    create_fixture_tree(repository)

    with pytest.raises(ValueError):
        build_release(repository, tmp_path / "release.zip", "public/release")


def test_release_inventory_excludes_casefolded_artifact_directories(tmp_path: Path) -> None:
    """Generated parser and model artifacts must not rely on filename suffixes."""
    files = {
        "src/Forests/forest.txt": "generated forest",
        "src/PARSER_RESOURCES/parser.txt": "downloaded parser resource",
        "src/WEIGHTS/weights.txt": "model weight",
        "src/ARTIFACTS/artifact.txt": "generated artifact",
        "src/sumoe/model/architecture.py": "public source",
    }
    for relative_path, content in files.items():
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    paths = {path.as_posix() for path in iter_public_files(tmp_path)}

    assert "src/sumoe/model/architecture.py" in paths
    assert not any("/forests/" in path.casefold() for path in paths)
    assert not any("/parser_resources/" in path.casefold() for path in paths)
    assert not any("/weights/" in path.casefold() for path in paths)
    assert not any("/artifacts/" in path.casefold() for path in paths)


def test_release_inventory_excludes_model_and_parser_formats_in_public_roots(
    tmp_path: Path,
) -> None:
    """Model and parser binaries must stay private even below a public root."""
    files = {
        "src/sumoe/module.py": "public source",
        "src/vendor/parser.jar": "parser runtime",
        "src/vendor/tokenizer.model": "parser model",
        "src/vendor/weights.gguf": "model weights",
        "src/vendor/network.tflite": "model weights",
        "src/vendor/graph.pb": "model weights",
        "src/vendor/runtime.engine": "model weights",
    }
    for relative_path, content in files.items():
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    paths = {path.as_posix() for path in iter_public_files(tmp_path)}

    assert paths == {"src/sumoe/module.py"}


def test_release_inventory_excludes_a_casefolded_git_metadata_file(tmp_path: Path) -> None:
    """A linked-worktree .git file must not expose private repository metadata."""
    (tmp_path / ".GIT").write_text("gitdir: ../private-worktree", encoding="utf-8")

    paths = {path.as_posix() for path in iter_public_files(tmp_path)}

    assert ".GIT" not in paths


def test_release_inventory_excludes_nested_casefolded_docs_superpowers(tmp_path: Path) -> None:
    """Internal planning must stay excluded below any nested docs directory."""
    destination = tmp_path / "src/nested/DOCS/SUPERPOWERS/plan.txt"
    destination.parent.mkdir(parents=True)
    destination.write_text("internal plan", encoding="utf-8")

    paths = {path.as_posix() for path in iter_public_files(tmp_path)}

    assert "src/nested/DOCS/SUPERPOWERS/plan.txt" not in paths


def test_release_inventory_excludes_a_casefolded_dot_superpowers_component(
    tmp_path: Path,
) -> None:
    """Internal task artifacts must stay excluded at any directory depth."""
    destination = tmp_path / "scripts/nested/.SuPeRpOwErS/sdd/task-report.md"
    destination.parent.mkdir(parents=True)
    destination.write_text("internal release report", encoding="utf-8")

    paths = {path.as_posix() for path in iter_public_files(tmp_path)}

    assert "scripts/nested/.SuPeRpOwErS/sdd/task-report.md" not in paths


def test_release_inventory_scopes_runtime_data_exclusion_to_the_root(tmp_path: Path) -> None:
    """Runtime data must stay private without hiding public data-package source."""
    files = {
        "DaTa/runtime.jsonl": "runtime data",
        "src/sumoe/data/loader.py": "public Python source",
        "configs/data/scrolls.yaml": "public configuration",
    }
    for relative_path, content in files.items():
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    paths = {path.as_posix() for path in iter_public_files(tmp_path)}

    assert "DaTa/runtime.jsonl" not in paths
    assert "src/sumoe/data/loader.py" in paths
    assert "configs/data/scrolls.yaml" in paths


def test_real_release_inventory_includes_tracked_public_data_source_and_config() -> None:
    """The repository's public data package and configuration must ship."""
    expected = {
        "configs/data/scrolls.yaml",
        "src/sumoe/data/__init__.py",
        "src/sumoe/data/collator.py",
        "src/sumoe/data/sampler.py",
        "src/sumoe/data/scrolls.py",
        "src/sumoe/data/tasks.py",
        "src/sumoe/data/templates.py",
    }

    paths = {path.as_posix() for path in iter_public_files(ROOT)}

    assert expected <= paths
