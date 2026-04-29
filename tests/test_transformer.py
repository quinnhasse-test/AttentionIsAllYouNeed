"""Tests for the full Transformer model.

Covers:
  - Forward pass output shape (batch, tgt_len, vocab_size)
  - make_src_mask: True at pad positions
  - make_tgt_mask: True at pad positions and above diagonal
  - Encode-only path output shape
  - Gradient flows end-to-end
  - Parameter initialization: no all-zero linear layers
"""

import pytest
import torch

from src.model.transformer import Transformer, make_src_mask, make_tgt_mask


@pytest.fixture
def small_model():
    return Transformer(
        src_vocab_size=100,
        tgt_vocab_size=120,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_len=50,
        dropout=0.0,
        pad_idx=0,
    )


class TestMaskGeneration:

    def test_src_mask_shape(self):
        src = torch.randint(0, 50, (3, 10))
        mask = make_src_mask(src, pad_idx=0)
        assert mask.shape == (3, 1, 1, 10)

    def test_src_mask_true_at_pad(self):
        src = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
        mask = make_src_mask(src, pad_idx=0)
        assert mask[0, 0, 0, 2].item() is True
        assert mask[0, 0, 0, 3].item() is True
        assert mask[0, 0, 0, 0].item() is False

    def test_tgt_mask_shape(self):
        tgt = torch.randint(0, 50, (2, 8))
        mask = make_tgt_mask(tgt, pad_idx=0)
        assert mask.shape == (2, 1, 8, 8)

    def test_tgt_mask_upper_triangular(self):
        """Above the diagonal must be True (causal masking)."""
        tgt = torch.ones(1, 6, dtype=torch.long)  # no padding
        mask = make_tgt_mask(tgt, pad_idx=0).squeeze()  # (6, 6)
        # Upper triangle (excluding diagonal) should all be True
        upper = torch.triu(mask, diagonal=1)
        lower_diag = torch.tril(mask, diagonal=0)
        assert upper.all()
        assert not lower_diag.any()

    def test_tgt_mask_pad_position_true(self):
        tgt = torch.tensor([[2, 5, 0, 0]])  # positions 2,3 are pad
        mask = make_tgt_mask(tgt, pad_idx=0)
        # Pad columns must be True
        assert mask[0, 0, :, 2].all()
        assert mask[0, 0, :, 3].all()


class TestTransformerForward:

    def test_output_shape(self, small_model):
        src = torch.randint(1, 100, (2, 10))
        tgt = torch.randint(1, 120, (2, 8))
        logits = small_model(src, tgt)
        assert logits.shape == (2, 8, 120)

    def test_encode_shape(self, small_model):
        src = torch.randint(1, 100, (2, 12))
        memory = small_model.encode(src)
        assert memory.shape == (2, 12, 64)

    def test_deterministic_in_eval(self, small_model):
        small_model.eval()
        src = torch.randint(1, 100, (1, 6))
        tgt = torch.randint(1, 120, (1, 4))
        with torch.no_grad():
            out1 = small_model(src, tgt)
            out2 = small_model(src, tgt)
        assert torch.allclose(out1, out2)

    def test_gradient_flows_end_to_end(self, small_model):
        src = torch.randint(1, 100, (2, 8))
        tgt = torch.randint(1, 120, (2, 6))
        logits = small_model(src, tgt)
        logits.sum().backward()
        # Check at least one parameter received a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in small_model.parameters()
        )
        assert has_grad

    def test_no_nan_in_output(self, small_model):
        src = torch.randint(1, 100, (2, 10))
        tgt = torch.randint(1, 120, (2, 8))
        logits = small_model(src, tgt)
        assert not torch.isnan(logits).any()

    def test_param_init_not_zero(self, small_model):
        for name, p in small_model.named_parameters():
            if p.dim() > 1:
                assert p.abs().sum().item() > 0, f"Parameter {name} is all zeros"
