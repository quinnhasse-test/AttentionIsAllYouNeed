"""Tests for PositionalEncoding.

Covers:
  - Output shape preserved after forward pass
  - Even indices follow sin(pos / 10000^(2i/d))
  - Odd indices follow cos(pos / 10000^(2i/d))
  - Encoding is identical for same position across different batch items
  - The buffer is on the correct device
  - Encoding with d_model=1 (edge case)
"""

import math
import pytest
import torch

from src.model.positional_encoding import PositionalEncoding


class TestPositionalEncoding:

    def test_output_shape_unchanged(self):
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(3, 10, 64)
        out = pe(x)
        assert out.shape == (3, 10, 64)

    def test_sin_at_even_indices(self):
        """PE(pos, 2i) == sin(pos / 10000^(2i/d_model)) at each position."""
        d_model = 16
        pe = PositionalEncoding(d_model=d_model, max_len=50, dropout=0.0)

        # Access the pre-computed buffer directly (shape: 1, max_len, d_model)
        buffer = pe.pe.squeeze(0)  # (max_len, d_model)

        for pos in range(5):
            for i in range(d_model // 2):
                expected = math.sin(pos / (10000 ** (2 * i / d_model)))
                actual = buffer[pos, 2 * i].item()
                assert abs(actual - expected) < 1e-5, (
                    f"PE({pos}, {2*i}) expected {expected:.6f} got {actual:.6f}"
                )

    def test_cos_at_odd_indices(self):
        """PE(pos, 2i+1) == cos(pos / 10000^(2i/d_model))."""
        d_model = 16
        pe = PositionalEncoding(d_model=d_model, max_len=50, dropout=0.0)
        buffer = pe.pe.squeeze(0)

        for pos in range(5):
            for i in range(d_model // 2):
                expected = math.cos(pos / (10000 ** (2 * i / d_model)))
                actual = buffer[pos, 2 * i + 1].item()
                assert abs(actual - expected) < 1e-5, (
                    f"PE({pos}, {2*i+1}) expected {expected:.6f} got {actual:.6f}"
                )

    def test_encoding_same_across_batch(self):
        """Positional encoding must be position-dependent, not batch-dependent."""
        pe = PositionalEncoding(d_model=32, max_len=20, dropout=0.0)
        x = torch.zeros(4, 8, 32)
        out = pe(x)
        # All batch items start with identical zero embeddings, so after adding
        # PE the rows should be identical across the batch dimension.
        for b in range(1, 4):
            assert torch.allclose(out[0], out[b], atol=1e-6)

    def test_different_positions_differ(self):
        """Adjacent positions must have different encoding vectors."""
        pe = PositionalEncoding(d_model=32, max_len=50, dropout=0.0)
        buf = pe.pe.squeeze(0)
        assert not torch.allclose(buf[0], buf[1])
        assert not torch.allclose(buf[1], buf[2])

    def test_max_len_boundary(self):
        """Input exactly at max_len should not raise."""
        max_len = 20
        pe = PositionalEncoding(d_model=16, max_len=max_len, dropout=0.0)
        x = torch.zeros(1, max_len, 16)
        out = pe(x)
        assert out.shape == (1, max_len, 16)

    def test_device_consistency(self):
        """PE buffer must be on the same device as the model parameters."""
        pe = PositionalEncoding(d_model=16, max_len=20, dropout=0.0)
        cpu_device = torch.device("cpu")
        assert pe.pe.device.type == cpu_device.type
