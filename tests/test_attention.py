"""Tests for scaled dot-product attention and MultiHeadAttention.

Covers:
  - Output shape matches expected (batch, heads, seq_q, d_k)
  - Attention weights sum to 1.0 along key dimension
  - Causal mask prevents attending to future positions
  - Padding mask forces zero weight on masked positions
  - Multi-head attention output shape is (batch, seq_q, d_model)
  - Gradient flows through MultiHeadAttention
"""

import math
import pytest
import torch
import torch.nn as nn

from src.model.attention import scaled_dot_product_attention, MultiHeadAttention


class TestScaledDotProductAttention:
    """Tests for the bare attention function."""

    def test_output_shape(self):
        batch, heads, seq_q, seq_k, d_k = 2, 4, 5, 7, 16
        q = torch.randn(batch, heads, seq_q, d_k)
        k = torch.randn(batch, heads, seq_k, d_k)
        v = torch.randn(batch, heads, seq_k, d_k)

        out, weights = scaled_dot_product_attention(q, k, v)

        assert out.shape == (batch, heads, seq_q, d_k)
        assert weights.shape == (batch, heads, seq_q, seq_k)

    def test_weights_sum_to_one(self):
        """Attention weights must be a probability distribution over keys."""
        q = torch.randn(2, 4, 6, 8)
        k = torch.randn(2, 4, 10, 8)
        v = torch.randn(2, 4, 10, 8)

        _, weights = scaled_dot_product_attention(q, k, v)

        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_causal_mask_blocks_future(self):
        """With a strict lower-triangular causal mask, upper weights must be zero."""
        seq_len = 6
        q = k = v = torch.randn(1, 1, seq_len, 32)

        # Upper-triangular mask (True = masked)
        mask = torch.triu(torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool), diagonal=1)
        _, weights = scaled_dot_product_attention(q, k, v, mask=mask)

        # All positions in the upper triangle must be exactly 0
        upper = torch.triu(weights[0, 0], diagonal=1)
        assert upper.abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_padding_mask_zeros_out_padded_keys(self):
        """Padded key positions should receive zero attention weight."""
        batch, seq_q, seq_k, d_k = 1, 4, 6, 16
        q = torch.randn(batch, 1, seq_q, d_k)
        k = torch.randn(batch, 1, seq_k, d_k)
        v = torch.randn(batch, 1, seq_k, d_k)

        # Mask positions 4 and 5 (last two keys)
        mask = torch.zeros(batch, 1, seq_q, seq_k, dtype=torch.bool)
        mask[:, :, :, 4:] = True

        _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        assert weights[:, :, :, 4:].abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_no_nan_on_all_masked_row(self):
        """A fully masked row must produce zero output, not NaN."""
        q = torch.randn(1, 1, 3, 8)
        k = torch.randn(1, 1, 3, 8)
        v = torch.randn(1, 1, 3, 8)

        mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)  # mask everything
        out, weights = scaled_dot_product_attention(q, k, v, mask=mask)

        assert not torch.isnan(out).any()
        assert not torch.isnan(weights).any()


class TestMultiHeadAttention:
    """Tests for the MultiHeadAttention module."""

    @pytest.fixture
    def mha(self):
        return MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)

    def test_output_shape_self_attention(self, mha):
        batch, seq, d = 2, 10, 64
        x = torch.randn(batch, seq, d)
        out = mha(x, x, x)
        assert out.shape == (batch, seq, d)

    def test_output_shape_cross_attention(self, mha):
        batch = 2
        q = torch.randn(batch, 8, 64)
        kv = torch.randn(batch, 12, 64)
        out = mha(q, kv, kv)
        assert out.shape == (batch, 8, 64)

    def test_attn_weights_stored(self, mha):
        x = torch.randn(1, 5, 64)
        mha.eval()
        with torch.no_grad():
            mha(x, x, x)
        assert mha.attn_weights is not None
        assert mha.attn_weights.shape == (1, 4, 5, 5)

    def test_gradient_flows(self, mha):
        x = torch.randn(2, 6, 64, requires_grad=True)
        out = mha(x, x, x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_d_model_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, n_heads=4)

    def test_different_query_key_length(self, mha):
        """Cross-attention: query length can differ from key/value length."""
        q = torch.randn(3, 5, 64)
        kv = torch.randn(3, 20, 64)
        out = mha(q, kv, kv)
        assert out.shape == (3, 5, 64)
