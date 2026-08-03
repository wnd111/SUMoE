from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .actions import Action, ActionVocabulary, ParserState
from .dataset import DependencySentence, ParserVocabulary


@dataclass(frozen=True)
class ParserFeatureBatch:
    word_ids: torch.Tensor
    upos_ids: torch.Tensor
    char_ids: torch.Tensor
    stack_indices: torch.Tensor
    buffer_indices: torch.Tensor
    action_history: torch.Tensor

    def to(self, device: torch.device | str) -> ParserFeatureBatch:
        return ParserFeatureBatch(
            word_ids=self.word_ids.to(device),
            upos_ids=self.upos_ids.to(device),
            char_ids=self.char_ids.to(device),
            stack_indices=self.stack_indices.to(device),
            buffer_indices=self.buffer_indices.to(device),
            action_history=self.action_history.to(device),
        )


class StackTransformerParser(nn.Module):
    def __init__(
        self,
        vocabulary: ParserVocabulary,
        action_vocabulary: ActionVocabulary,
        model_dim: int = 256,
        word_dim: int = 256,
        upos_dim: int = 64,
        char_dim: int = 32,
        char_channels: int = 96,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.action_vocabulary = action_vocabulary
        self.model_dim = model_dim
        self.word_embedding = nn.Embedding(len(vocabulary.words), word_dim, padding_idx=0)
        self.upos_embedding = nn.Embedding(len(vocabulary.upos), upos_dim, padding_idx=0)
        self.char_embedding = nn.Embedding(len(vocabulary.characters), char_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(char_dim, char_channels, kernel_size=3, padding=1)
        self.token_projection = nn.Linear(word_dim + upos_dim + char_channels, model_dim)
        self.root_embedding = nn.Parameter(torch.empty(model_dim))
        self.stack_cls = nn.Parameter(torch.empty(model_dim))
        self.buffer_cls = nn.Parameter(torch.empty(model_dim))
        self.action_cls = nn.Parameter(torch.empty(model_dim))
        self.action_embedding = nn.Embedding(len(action_vocabulary), model_dim)
        self.position_embedding = nn.Embedding(1024, model_dim)

        def encoder() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

        self.stack_encoder = encoder()
        self.buffer_encoder = encoder()
        self.action_encoder = encoder()
        self.classifier = nn.Sequential(
            nn.Linear(model_dim * 3, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, len(action_vocabulary)),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.root_embedding, std=0.02)
        nn.init.normal_(self.stack_cls, std=0.02)
        nn.init.normal_(self.buffer_cls, std=0.02)
        nn.init.normal_(self.action_cls, std=0.02)

    @property
    def labels(self) -> tuple[str, ...]:
        return self.action_vocabulary.labels

    def _token_states(self, batch: ParserFeatureBatch) -> torch.Tensor:
        word = self.word_embedding(batch.word_ids)
        upos = self.upos_embedding(batch.upos_ids)
        batch_size, length, characters = batch.char_ids.shape
        char = self.char_embedding(batch.char_ids).reshape(
            batch_size * length, characters, -1
        ).transpose(1, 2)
        char = torch.relu(self.char_cnn(char)).amax(dim=-1).reshape(batch_size, length, -1)
        return self.dropout(self.token_projection(torch.cat((word, upos, char), dim=-1)))

    def _encode_indices(
        self,
        token_states: torch.Tensor,
        indices: torch.Tensor,
        cls: torch.Tensor,
        encoder: nn.TransformerEncoder,
    ) -> torch.Tensor:
        batch_size = token_states.shape[0]
        root = self.root_embedding.view(1, 1, -1).expand(batch_size, 1, -1)
        states = torch.cat((root, token_states), dim=1)
        padding = indices == -2
        gather = (indices + 1).clamp_min(0).unsqueeze(-1).expand(-1, -1, self.model_dim)
        sequence = states.gather(1, gather)
        prefix = cls.view(1, 1, -1).expand(batch_size, 1, -1)
        sequence = torch.cat((prefix, sequence), dim=1)
        mask = torch.cat(
            (torch.zeros(batch_size, 1, dtype=torch.bool, device=indices.device), padding), dim=1
        )
        positions = self.position_embedding(
            torch.arange(sequence.shape[1], device=sequence.device)
        ).unsqueeze(0)
        return encoder(sequence + positions, src_key_padding_mask=mask)[:, 0]

    def _encode_actions(self, history: torch.Tensor) -> torch.Tensor:
        batch_size = history.shape[0]
        padding = history < 0
        sequence = self.action_embedding(history.clamp_min(0))
        prefix = self.action_cls.view(1, 1, -1).expand(batch_size, 1, -1)
        sequence = torch.cat((prefix, sequence), dim=1)
        mask = torch.cat(
            (torch.zeros(batch_size, 1, dtype=torch.bool, device=history.device), padding), dim=1
        )
        positions = self.position_embedding(
            torch.arange(sequence.shape[1], device=sequence.device)
        ).unsqueeze(0)
        return self.action_encoder(sequence + positions, src_key_padding_mask=mask)[:, 0]

    def forward(self, batch: ParserFeatureBatch) -> torch.Tensor:
        token_states = self._token_states(batch)
        stack = self._encode_indices(
            token_states, batch.stack_indices, self.stack_cls, self.stack_encoder
        )
        buffer = self._encode_indices(
            token_states, batch.buffer_indices, self.buffer_cls, self.buffer_encoder
        )
        actions = self._encode_actions(batch.action_history)
        return self.classifier(torch.cat((stack, buffer, actions), dim=-1))

    def feature_batch(self, sentence: DependencySentence, state: ParserState) -> ParserFeatureBatch:
        max_chars = max(len(word) for word in sentence.words)
        char_rows = [
            [self.vocabulary.char_id(char) for char in word] + [0] * (max_chars - len(word))
            for word in sentence.words
        ]
        history = [self.action_vocabulary.id(action) for action in state.actions]
        if not history:
            history = [-1]
        return ParserFeatureBatch(
            word_ids=torch.tensor([[self.vocabulary.word_id(word) for word in sentence.words]]),
            upos_ids=torch.tensor([[self.vocabulary.upos_id(tag) for tag in sentence.upos]]),
            char_ids=torch.tensor([char_rows]),
            stack_indices=torch.tensor([state.stack]),
            buffer_indices=torch.tensor([state.buffer or (-2,)]),
            action_history=torch.tensor([history]),
        )

    @torch.no_grad()
    def score_state(self, sentence: DependencySentence, state: ParserState) -> dict[Action, float]:
        device = next(self.parameters()).device
        logits = self(self.feature_batch(sentence, state).to(device))[0].cpu()
        return {
            action: float(logits[index].item())
            for index, action in enumerate(self.action_vocabulary.actions)
        }
