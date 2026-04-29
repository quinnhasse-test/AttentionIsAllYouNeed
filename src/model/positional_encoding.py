"""Sinusoidal positional encoding from Vaswani et al. (2017).

PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

The encoding is fixed (not learned) and added to the token embedding
before the first encoder or decoder layer.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Add sinusoidal position signals to token embeddings.

    The encoding matrix is pre-computed once and stored as a non-parameter
    buffer, so it travels with the model on device moves and checkpoint saves.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build (max_len, d_model) encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term shape: (d_model // 2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        # Store as (1, max_len, d_model) for broadcasting over the batch dim
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to x and apply dropout.

        Args:
            x: Token embeddings of shape (batch, seq_len, d_model).

        Returns:
            Tensor of same shape with positional information added.
        """
        # pe is (1, max_len, d_model); slice to seq_len
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
