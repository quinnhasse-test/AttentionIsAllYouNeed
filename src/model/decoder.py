"""Transformer decoder: N identical layers with masked self-attention and cross-attention.

Each layer:
  1. Masked multi-head self-attention (prevents looking at future tokens)
  2. Add & LayerNorm
  3. Multi-head cross-attention over encoder memory
  4. Add & LayerNorm
  5. Position-wise feed-forward
  6. Add & LayerNorm
"""

import copy
import torch
import torch.nn as nn
from src.model.attention import MultiHeadAttention
from src.model.feed_forward import PositionWiseFeedForward
from typing import Optional


class DecoderLayer(nn.Module):
    """Single decoder layer: masked self-attn -> cross-attn -> FFN, each with add+norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for one decoder layer.

        Args:
            x:        (batch, tgt_len, d_model) — target sequence so far
            memory:   (batch, src_len, d_model) — encoder output
            src_mask: Padding mask for encoder memory (batch, 1, 1, src_len)
            tgt_mask: Causal + padding mask for target (batch, 1, tgt_len, tgt_len)

        Returns:
            (batch, tgt_len, d_model)
        """
        # Masked self-attention (post-norm)
        sa_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(sa_out))

        # Cross-attention over encoder memory (post-norm)
        ca_out = self.cross_attn(x, memory, memory, mask=src_mask)
        x = self.norm2(x + self.dropout(ca_out))

        # Feed-forward (post-norm)
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


class Decoder(nn.Module):
    """Stack of N decoder layers followed by a final layer norm."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run target through all decoder layers.

        Args:
            x:        (batch, tgt_len, d_model)
            memory:   (batch, src_len, d_model) — encoder output
            src_mask: Encoder padding mask.
            tgt_mask: Decoder causal mask.

        Returns:
            (batch, tgt_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
