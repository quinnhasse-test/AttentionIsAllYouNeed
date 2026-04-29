"""Attention mechanisms from Vaswani et al. (2017).

Scaled dot-product attention:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Multi-head attention:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
    head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    Args:
        q: Query tensor of shape (batch, heads, seq_q, d_k).
        k: Key tensor of shape (batch, heads, seq_k, d_k).
        v: Value tensor of shape (batch, heads, seq_k, d_v).
        mask: Optional boolean mask of shape broadcastable to
            (batch, heads, seq_q, seq_k).  Positions where mask is
            True (or 1) are *excluded* from attention (set to -inf).
        dropout: Optional dropout applied to attention weights.

    Returns:
        output: Tensor of shape (batch, heads, seq_q, d_v).
        weights: Attention weight tensor of shape (batch, heads, seq_q, seq_k).
    """
    d_k = q.size(-1)
    # (batch, heads, seq_q, seq_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    # Replace NaN that arises when an entire row is masked (all -inf → NaN)
    weights = weights.nan_to_num(0.0)

    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, v)
    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate Q, K, V linear projections.

    Splits d_model into h heads of dimension d_k = d_model // h each.
    After computing attention in each head independently, concatenates
    the results and projects back to d_model.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None  # saved for visualization

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, seq, d_model) -> (batch, heads, seq, d_k)."""
        batch, seq, _ = x.size()
        x = x.view(batch, seq, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, heads, seq, d_k) -> (batch, seq, d_model)."""
        batch, _, seq, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask:  Optional mask broadcastable to (batch, heads, seq_q, seq_k).
                   True positions are masked out.

        Returns:
            output: (batch, seq_q, d_model)
        """
        q = self._split_heads(self.w_q(query))  # (B, h, T_q, d_k)
        k = self._split_heads(self.w_k(key))    # (B, h, T_k, d_k)
        v = self._split_heads(self.w_v(value))  # (B, h, T_k, d_k)

        context, self.attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )
        context = self._merge_heads(context)    # (B, T_q, d_model)
        return self.w_o(context)
