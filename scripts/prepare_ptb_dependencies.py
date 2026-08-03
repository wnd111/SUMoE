from __future__ import annotations

import argparse
from pathlib import Path


def section_number(path: Path) -> int:
    stem = path.stem.lower()
    for value in range(25):
        if f"{value:02d}" in stem:
            return value
    raise ValueError(f"cannot determine PTB section from {path}")


def combine(files: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for path in sorted(files):
            text = path.read_text(encoding="utf-8").rstrip()
            if text:
                target.write(text + "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split CoreNLP-converted PTB CoNLL-U files into sections 02-21, 22, and 23"
    )
    parser.add_argument("--section-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    files = list(args.section_dir.rglob("*.conllu"))
    if not files:
        raise FileNotFoundError(f"no CoNLL-U section files found in {args.section_dir}")
    grouped = {"train": [], "dev": [], "test": []}
    for path in files:
        section = section_number(path)
        if 2 <= section <= 21:
            grouped["train"].append(path)
        elif section == 22:
            grouped["dev"].append(path)
        elif section == 23:
            grouped["test"].append(path)
    for split, split_files in grouped.items():
        if not split_files:
            raise ValueError(f"PTB {split} section files are missing")
        combine(split_files, args.output_dir / f"ptb-{split}.conllu")


if __name__ == "__main__":
    main()

