from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .types import DocumentForest, ForestManifest


@dataclass(frozen=True)
class ForestVerification:
    valid: bool
    sha256: str
    expected_sha256: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def write_forests(
    path: Path,
    forests: Iterable[DocumentForest],
    manifest: ForestManifest,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps({"type": "manifest", "manifest": manifest.to_dict()}, sort_keys=True)
            + "\n"
        )
        for forest in forests:
            stream.write(
                json.dumps({"type": "forest", "forest": forest.to_dict()}, sort_keys=True)
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    digest = sha256_file(path)
    _sidecar(path).write_text(digest + "\n", encoding="ascii")


def read_forests(path: Path) -> tuple[ForestManifest, Iterator[DocumentForest]]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or rows[0].get("type") != "manifest":
        raise ValueError("forest file must begin with a manifest row")
    manifest = ForestManifest.from_dict(rows[0]["manifest"])
    forests: list[DocumentForest] = []
    for row in rows[1:]:
        if row.get("type") != "forest":
            raise ValueError("forest data row has an invalid type")
        forests.append(DocumentForest.from_dict(row["forest"]))
    return manifest, iter(forests)


def verify_forest_file(path: Path) -> ForestVerification:
    actual = sha256_file(path)
    sidecar = _sidecar(path)
    expected = sidecar.read_text(encoding="ascii").strip() if sidecar.exists() else None
    return ForestVerification(valid=expected == actual, sha256=actual, expected_sha256=expected)

