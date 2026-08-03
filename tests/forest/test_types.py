from __future__ import annotations

import pytest

from sumoe.forest.types import CandidateTree, TokenSpan


def test_candidate_tree_requires_exactly_one_root() -> None:
    with pytest.raises(ValueError, match="exactly one root"):
        CandidateTree(
            parser="stanza",
            heads=(1, 0),
            labels=("dep", "dep"),
            raw_score=-0.2,
        )


def test_candidate_tree_rejects_invalid_head_index() -> None:
    with pytest.raises(ValueError, match="head index 3 is invalid"):
        CandidateTree(
            parser="stanza",
            heads=(-1, 3),
            labels=("root", "obj"),
            raw_score=-0.2,
        )


def test_token_span_requires_nonempty_ordered_offsets() -> None:
    with pytest.raises(ValueError, match="invalid token span"):
        TokenSpan(text="word", start=5, end=5)

