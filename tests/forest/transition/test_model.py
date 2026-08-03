from __future__ import annotations

import torch

from sumoe.forest.transition.actions import ActionVocabulary
from sumoe.forest.transition.dataset import ParserVocabulary
from sumoe.forest.transition.model import ParserFeatureBatch, StackTransformerParser


def test_stack_transformer_returns_one_logit_per_action() -> None:
    vocabulary = ParserVocabulary.from_sequences(
        words=[("a", "works")],
        upos=[("NOUN", "VERB")],
        labels=[("nsubj", "root")],
    )
    actions = ActionVocabulary(vocabulary.labels)
    model = StackTransformerParser(
        vocabulary=vocabulary,
        action_vocabulary=actions,
        model_dim=32,
        word_dim=16,
        upos_dim=8,
        char_dim=8,
        char_channels=8,
        num_heads=4,
        ffn_dim=64,
        num_layers=2,
        dropout=0.0,
    )
    batch = ParserFeatureBatch(
        word_ids=torch.tensor([[vocabulary.word_id("a"), vocabulary.word_id("works")]]),
        upos_ids=torch.tensor([[vocabulary.upos_id("NOUN"), vocabulary.upos_id("VERB")]]),
        char_ids=torch.tensor([[[2, 0], [3, 4]]]),
        stack_indices=torch.tensor([[-1, 0]]),
        buffer_indices=torch.tensor([[1]]),
        action_history=torch.tensor([[actions.shift_id]]),
    )
    logits = model(batch)
    assert logits.shape == (1, len(actions))
    assert torch.isfinite(logits).all()

