from __future__ import annotations

import argparse
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

PUBLIC_TOP_LEVEL_FILES = {
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
}

PUBLIC_DIRECTORY_ROOTS = {
    ".github",
    "configs",
    "docs",
    "scripts",
    "src",
    "tests",
}

EXCLUDED_DIRS = {
    ".git",
    ".superpowers",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "checkpoints",
    "outputs",
    "predictions",
    "reports",
    "cache",
    "wandb",
    ".egg-info",
    "forests",
    "parser_resources",
    "parser-resources",
    "parser_models",
    "parser-models",
    "models",
    "weights",
    "model_weights",
    "model-weights",
    "artifacts",
    "model_artifacts",
    "model-artifacts",
}

EXCLUDED_TOP_LEVEL_DIRS = {"data"}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".zip",
    ".bin",
    ".onnx",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".jar",
    ".model",
    ".gguf",
    ".tflite",
    ".pb",
    ".engine",
}

_FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_EXCLUDED_DIRS_CASEFOLDED = {directory.casefold() for directory in EXCLUDED_DIRS}
_EXCLUDED_TOP_LEVEL_DIRS_CASEFOLDED = {
    directory.casefold() for directory in EXCLUDED_TOP_LEVEL_DIRS
}
_EXCLUDED_SUFFIXES_CASEFOLDED = {suffix.casefold() for suffix in EXCLUDED_SUFFIXES}


def _is_excluded(relative_path: Path) -> bool:
    parts = tuple(part.casefold() for part in relative_path.parts)
    if parts and parts[0] in _EXCLUDED_TOP_LEVEL_DIRS_CASEFOLDED:
        return True
    if any(part in _EXCLUDED_DIRS_CASEFOLDED or part.endswith(".egg-info") for part in parts):
        return True
    if any(
        first == "docs" and second == "superpowers"
        for first, second in zip(parts, parts[1:], strict=False)
    ):
        return True
    return relative_path.suffix.casefold() in _EXCLUDED_SUFFIXES_CASEFOLDED


def iter_public_files(root: Path) -> tuple[Path, ...]:
    """Return sorted, public file paths relative to a repository root."""
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Release root is not a directory: {root}")

    public_files = []
    for public_file in PUBLIC_TOP_LEVEL_FILES:
        relative_path = Path(public_file)
        candidate = root / relative_path
        if candidate.is_file() and not candidate.is_symlink() and not _is_excluded(relative_path):
            public_files.append(relative_path)
    for directory_root in PUBLIC_DIRECTORY_ROOTS:
        public_root = root / directory_root
        if public_root.is_symlink() or not public_root.is_dir():
            continue
        for candidate in public_root.rglob("*"):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            relative_path = candidate.relative_to(root)
            if not _is_excluded(relative_path):
                public_files.append(relative_path)
    return tuple(sorted(public_files, key=lambda path: path.as_posix()))


def _validate_archive_root(archive_root: str) -> str:
    if not archive_root or archive_root in {".", ".."}:
        raise ValueError("archive_root must be a non-empty directory name")
    if "/" in archive_root or "\\" in archive_root:
        raise ValueError("archive_root must be a single directory name")
    return archive_root


def build_release(root: Path, output: Path, archive_root: str) -> None:
    """Build a deterministic public-source ZIP without replacing an existing file."""
    archive_root = _validate_archive_root(archive_root)
    root = root.resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite existing release archive: {output}")
    output = output.parent.resolve() / output.name
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite existing release archive: {output}")

    public_files = iter_public_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for relative_path in public_files:
            member = ZipInfo(f"{archive_root}/{relative_path.as_posix()}")
            member.date_time = _FIXED_ZIP_TIMESTAMP
            member.create_system = 3
            member.compress_type = ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, (root / relative_path).read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a filtered public SUMoE release archive.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for a new ZIP archive; an existing path is never overwritten.",
    )
    parser.add_argument(
        "--archive-root",
        default="SUMoE-Finley-paper-aligned-code",
        help="Single top-level directory name inside the archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_release(Path.cwd(), args.output, args.archive_root)


if __name__ == "__main__":
    main()
