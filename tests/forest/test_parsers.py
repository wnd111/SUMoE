from __future__ import annotations

from pathlib import Path

import yaml

from sumoe.forest.parsers.transition_parser import TransitionParserAdapter
from sumoe.forest.transition.training import (
    load_transition_checkpoint,
    train_transition_parser,
)


def test_transition_adapter_returns_requested_n_best(tmp_path: Path) -> None:
    config = yaml.safe_load(
        Path("tests/fixtures/transition_tiny.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(config, dict)
    config["output"] = str(tmp_path)
    checkpoint = train_transition_parser(config, smoke_test=True).checkpoint
    model = load_transition_checkpoint(checkpoint)
    adapter = TransitionParserAdapter(model=model, n_best=3, beam_size=8)
    sentence = adapter.make_sentence(("A", "works"), ("NOUN", "VERB"), 0)
    candidates = adapter.parse_sentence(sentence)
    assert 1 <= len(candidates) <= 3
    assert all(candidate.parser == "transition" for candidate in candidates)

