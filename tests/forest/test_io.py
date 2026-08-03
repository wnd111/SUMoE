from __future__ import annotations

from pathlib import Path

from sumoe.forest.io import read_forests, verify_forest_file, write_forests
from sumoe.forest.types import (
    CandidateTree,
    DocumentForest,
    ForestManifest,
    SentenceForest,
    TokenSpan,
)


def sample_manifest() -> ForestManifest:
    return ForestManifest(
        schema_version=1,
        data_sha256="a" * 64,
        tokenizer_revision="main",
        parser_versions={"stanza": "1.9.2", "spacy": "3.8.2", "transition": "1"},
        parser_hashes={"transition": "b" * 64},
        calibration_sha256="c" * 64,
    )


def sample_document_forest() -> DocumentForest:
    return DocumentForest(
        example_id="qasper::train::1",
        sentences=(
            SentenceForest(
                text_start=0,
                text_end=12,
                tokens=(TokenSpan("Good", 0, 4), TokenSpan("result", 5, 11)),
                candidates=(
                    CandidateTree(
                        parser="stanza",
                        heads=(-1, 0),
                        labels=("root", "obj"),
                        raw_score=-0.2,
                        posterior=1.0,
                    ),
                ),
            ),
        ),
    )


def test_forest_jsonl_round_trip_preserves_hash(tmp_path: Path) -> None:
    output = tmp_path / "forest.jsonl"
    write_forests(output, [sample_document_forest()], sample_manifest())
    manifest, loaded = read_forests(output)
    assert manifest == sample_manifest()
    assert list(loaded) == [sample_document_forest()]
    verification = verify_forest_file(output)
    assert verification.valid
    assert len(verification.sha256) == 64


def test_modified_forest_fails_hash_verification(tmp_path: Path) -> None:
    output = tmp_path / "forest.jsonl"
    write_forests(output, [sample_document_forest()], sample_manifest())
    output.write_text(output.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert not verify_forest_file(output).valid

