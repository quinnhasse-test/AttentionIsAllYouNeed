"""Tests for Encoder, Decoder, and their sublayers.

Covers:
  - EncoderLayer output shape
  - Encoder (N-layer stack) output shape
  - DecoderLayer output shape with src and tgt masks
  - Decoder (N-layer stack) output shape
  - Mask shapes are handled correctly
  - Gradient flow through encoder and decoder
"""

import pytest
import torch

from src.model.encoder import Encoder, EncoderLayer
from src.model.decoder import Decoder, DecoderLayer
from src.model.transformer import make_src_mask, make_tgt_mask


@pytest.fixture
def dims():
    return {"d_model": 64, "n_heads": 4, "d_ff": 128, "dropout": 0.0}


class TestEncoderLayer:

    def test_output_shape(self, dims):
        layer = EncoderLayer(**dims)
        batch, seq = 2, 10
        x = torch.randn(batch, seq, dims["d_model"])
        out = layer(x)
        assert out.shape == (batch, seq, dims["d_model"])

    def test_with_src_mask(self, dims):
        layer = EncoderLayer(**dims)
        batch, seq = 2, 8
        x = torch.randn(batch, seq, dims["d_model"])
        # Mask last 2 positions for all items in batch
        mask = torch.zeros(batch, 1, 1, seq, dtype=torch.bool)
        mask[:, :, :, -2:] = True
        out = layer(x, src_mask=mask)
        assert out.shape == (batch, seq, dims["d_model"])

    def test_gradient_flows(self, dims):
        layer = EncoderLayer(**dims)
        x = torch.randn(2, 5, dims["d_model"], requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None


class TestEncoder:

    def test_output_shape(self, dims):
        encoder = Encoder(n_layers=3, **dims)
        batch, seq = 2, 15
        x = torch.randn(batch, seq, dims["d_model"])
        out = encoder(x)
        assert out.shape == (batch, seq, dims["d_model"])

    def test_with_pad_mask(self, dims):
        """Encoder output should have same shape when padding mask is supplied."""
        encoder = Encoder(n_layers=2, **dims)
        src = torch.randint(1, 100, (3, 12))
        src[:, -3:] = 0  # last 3 positions are padding
        mask = make_src_mask(src, pad_idx=0)

        embed = torch.randn(3, 12, dims["d_model"])
        out = encoder(embed, mask)
        assert out.shape == (3, 12, dims["d_model"])


class TestDecoderLayer:

    def test_output_shape(self, dims):
        layer = DecoderLayer(**dims)
        batch, tgt_len, src_len = 2, 7, 10
        x = torch.randn(batch, tgt_len, dims["d_model"])
        memory = torch.randn(batch, src_len, dims["d_model"])
        out = layer(x, memory)
        assert out.shape == (batch, tgt_len, dims["d_model"])

    def test_with_masks(self, dims):
        layer = DecoderLayer(**dims)
        batch, tgt_len, src_len = 2, 6, 9

        src = torch.randint(1, 50, (batch, src_len))
        tgt = torch.randint(1, 50, (batch, tgt_len))
        src_mask = make_src_mask(src, pad_idx=0)
        tgt_mask = make_tgt_mask(tgt, pad_idx=0)

        x = torch.randn(batch, tgt_len, dims["d_model"])
        memory = torch.randn(batch, src_len, dims["d_model"])
        out = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        assert out.shape == (batch, tgt_len, dims["d_model"])

    def test_gradient_flows(self, dims):
        layer = DecoderLayer(**dims)
        x = torch.randn(1, 4, dims["d_model"], requires_grad=True)
        memory = torch.randn(1, 6, dims["d_model"])
        out = layer(x, memory)
        out.sum().backward()
        assert x.grad is not None


class TestDecoder:

    def test_output_shape(self, dims):
        decoder = Decoder(n_layers=3, **dims)
        batch, tgt_len, src_len = 2, 8, 12
        x = torch.randn(batch, tgt_len, dims["d_model"])
        memory = torch.randn(batch, src_len, dims["d_model"])
        out = decoder(x, memory)
        assert out.shape == (batch, tgt_len, dims["d_model"])
