"""Transformer encoder: N identical layers of self-attention + FFN.

Each layer:
  1. Multi-head self-attention (with padding mask)
  2. Add & LayerNorm
  3. Position-wise feed-forward
  4. Add & LayerNorm

The original paper uses post-norm (layer norm after the residual add).
"""

import copy
import torch
import torch.nn as nn
from src.model.attention import MultiHeadAttention
from src.model.feed_forward import PositionWiseFeedForward
from typing import Optional


class EncoderLayer(nn.Module):
    """Single encoder layer: self-attention -> add+norm -> FFN -> add+norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for one encoder layer.

        Args:
            x:        (batch, src_len, d_model)
            src_mask: (batch, 1, 1, src_len) — padding mask; True = pad

        Returns:
            (batch, src_len, d_model)
        """
        # Self-attention sublayer (post-norm)
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward sublayer (post-norm)
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class Encoder(nn.Module):
    """Stack of N encoder layers followed by a final layer norm."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run input through all encoder layers.

        Args:
            x:        (batch, src_len, d_model)
            src_mask: Optional padding mask.

        Returns:
            (batch, src_len, d_model) — encoder memory.
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
