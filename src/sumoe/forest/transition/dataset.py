from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

PAD = "<pad>"
UNK = "<unk>"


@dataclass(frozen=True)
class DependencySentence:
    words: tuple[str, ...]
    upos: tuple[str, ...]
    heads: tuple[int, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        length = len(self.words)
        fields = (self.upos, self.heads, self.labels)
        if length < 1 or not all(len(values) == length for values in fields):
            raise ValueError("dependency sentence fields must have equal nonzero length")


class ParserVocabulary:
    def __init__(
        self,
        words: Sequence[str],
        upos: Sequence[str],
        characters: Sequence[str],
        labels: Sequence[str],
    ) -> None:
        self.words = tuple(words)
        self.upos = tuple(upos)
        self.characters = tuple(characters)
        self.labels = tuple(labels)
        self._word = {value: index for index, value in enumerate(self.words)}
        self._upos = {value: index for index, value in enumerate(self.upos)}
        self._char = {value: index for index, value in enumerate(self.characters)}

    @classmethod
    def from_sequences(
        cls,
        words: Iterable[Sequence[str]],
        upos: Iterable[Sequence[str]],
        labels: Iterable[Sequence[str]],
    ) -> ParserVocabulary:
        word_sequences = [tuple(sequence) for sequence in words]
        upos_sequences = [tuple(sequence) for sequence in upos]
        label_sequences = [tuple(sequence) for sequence in labels]
        word_values = (PAD, UNK, *sorted({word.lower() for seq in word_sequences for word in seq}))
        upos_values = (PAD, UNK, *sorted({tag for seq in upos_sequences for tag in seq}))
        character_values = {
            char for sequence in word_sequences for word in sequence for char in word
        }
        characters = (PAD, UNK, *sorted(character_values))
        unique_labels = {label for sequence in label_sequences for label in sequence}
        label_values = tuple(
            sorted(unique_labels, key=lambda value: (value != "root", value))
        )
        return cls(word_values, upos_values, characters, label_values)

    def word_id(self, value: str) -> int:
        return self._word.get(value.lower(), self._word[UNK])

    def upos_id(self, value: str) -> int:
        return self._upos.get(value, self._upos[UNK])

    def char_id(self, value: str) -> int:
        return self._char.get(value, self._char[UNK])

    def to_dict(self) -> dict[str, object]:
        return {
            "words": list(self.words),
            "upos": list(self.upos),
            "characters": list(self.characters),
            "labels": list(self.labels),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ParserVocabulary:
        return cls(
            data["words"], data["upos"], data["characters"], data["labels"]  # type: ignore[arg-type]
        )


def load_conllu(path: Path) -> list[DependencySentence]:
    sentences: list[DependencySentence] = []
    rows: list[list[str]] = []

    def flush() -> None:
        if not rows:
            return
        words = tuple(row[1] for row in rows)
        upos = tuple(row[3] for row in rows)
        heads = tuple(-1 if int(row[6]) == 0 else int(row[6]) - 1 for row in rows)
        labels = tuple(row[7] for row in rows)
        sentences.append(DependencySentence(words, upos, heads, labels))
        rows.clear()

    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            if not line:
                flush()
                continue
            if line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) != 10:
                raise ValueError(f"invalid CoNLL-U row in {path}: {line}")
            if "-" in columns[0] or "." in columns[0]:
                continue
            rows.append(columns)
    flush()
    return sentences
