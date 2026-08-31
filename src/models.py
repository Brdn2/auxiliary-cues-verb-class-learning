from __future__ import annotations

import torch
from torch import nn


class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int = 128,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        outputs, _ = self.lstm(embedded)
        return self.output(self.dropout(outputs))


class TransformerMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_length: int = 96,
        embedding_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_length={self.max_length}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        padding_mask = input_ids.eq(self.pad_id)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.output(self.norm(encoded))
