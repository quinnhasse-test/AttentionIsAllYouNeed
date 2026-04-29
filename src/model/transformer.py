"""Full transformer model for sequence-to-sequence translation.

Combines:
  - Source token embedding + positional encoding
  - Encoder stack
  - Target token embedding + positional encoding
  - Decoder stack
  - Linear projection to target vocabulary logits

Mask generation lives here: padding masks for encoder input and
combined causal+padding mask for decoder input.
"""

import math
import torch
import torch.nn as nn
from src.model.attention import MultiHeadAttention
from src.model.positional_encoding import PositionalEncoding
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from typing import Optional


def make_src_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Build a padding mask for the source sequence.

    Args:
        src:     (batch, src_len) integer token ids.
        pad_idx: Token id used for padding.

    Returns:
        (batch, 1, 1, src_len) bool tensor; True where src == pad_idx.
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Build a combined causal + padding mask for the target sequence.

    The causal mask prevents position i from attending to positions > i.
    Combined with the padding mask so that pad tokens are also ignored.

    Args:
        tgt:     (batch, tgt_len) integer token ids.
        pad_idx: Token id used for padding.

    Returns:
        (batch, 1, tgt_len, tgt_len) bool tensor.
    """
    tgt_len = tgt.size(1)
    # Causal mask: upper triangle (excluding diagonal) is True (masked)
    causal = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1,
    )  # (tgt_len, tgt_len)
    # Padding mask: (batch, 1, 1, tgt_len)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    # Broadcast and combine: True means "mask this position"
    return causal.unsqueeze(0).unsqueeze(0) | pad_mask


class Transformer(nn.Module):
    """Encoder-decoder transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        d_model:        Model dimensionality (default 512).
        n_heads:        Number of attention heads (default 8).
        n_layers:       Number of encoder and decoder layers each (default 6).
        d_ff:           Feed-forward inner dimensionality (default 2048).
        max_len:        Maximum sequence length for positional encoding.
        dropout:        Dropout rate (default 0.1).
        pad_idx:        Padding token index (default 0).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.src_pe = PositionalEncoding(d_model, max_len, dropout)
        self.tgt_pe = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Tie decoder embedding weights to output projection (optional but common)
        # Only when src and tgt share a vocabulary; kept separate here for generality.

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialization for all linear and embedding layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode source tokens to memory.

        Args:
            src:      (batch, src_len) integer token ids.
            src_mask: Optional (batch, 1, 1, src_len) padding mask.

        Returns:
            (batch, src_len, d_model) encoder memory.
        """
        embed = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(embed, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target tokens conditioned on encoder memory.

        Args:
            tgt:      (batch, tgt_len) integer token ids.
            memory:   (batch, src_len, d_model) encoder output.
            src_mask: Encoder padding mask.
            tgt_mask: Decoder causal+padding mask.

        Returns:
            (batch, tgt_len, d_model)
        """
        embed = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        return self.decoder(embed, memory, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass: src -> encoder -> decoder -> logits.

        Masks are built automatically from src and tgt using self.pad_idx.

        Args:
            src: (batch, src_len) integer token ids.
            tgt: (batch, tgt_len) integer token ids.
                 Should be the target shifted right (teacher forcing).

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size) unnormalized log-probabilities.
        """
        src_mask = make_src_mask(src, self.pad_idx)
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)

        memory = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.output_proj(dec_out)
