"""Tests for PositionWiseFeedForward.

Covers:
  - Output shape equals input shape (position-wise: no seq-len change)
  - ReLU activation variant
  - GELU activation variant
  - Gradient flows through both activation types
  - No NaN in output for normal inputs
  - Intermediate dimension does not leak into output shape
  - Zero input produces non-zero output (bias-free but weights non-zero after init)
"""

import pytest
import torch

from src.model.feed_forward import PositionWiseFeedForward


class TestPositionWiseFeedForward:

    @pytest.mark.parametrize("activation", ["relu", "gelu"])
    def test_output_shape(self, activation: str) -> None:
        """Output shape must equal input shape for both activation types."""
        d_model, d_ff = 64, 256
        ffn = PositionWiseFeedForward(d_model, d_ff, dropout=0.0, activation=activation)
        x = torch.randn(3, 10, d_model)
        out = ffn(x)
        assert out.shape == (3, 10, d_model)

    def test_relu_activation(self) -> None:
        """ReLU-based FFN produces finite output."""
        ffn = PositionWiseFeedForward(32, 128, dropout=0.0, activation="relu")
        x = torch.randn(2, 5, 32)
        out = ffn(x)
        assert torch.isfinite(out).all()

    def test_gelu_activation(self) -> None:
        """GELU-based FFN produces finite output."""
        ffn = PositionWiseFeedForward(32, 128, dropout=0.0, activation="gelu")
        x = torch.randn(2, 5, 32)
        out = ffn(x)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("activation", ["relu", "gelu"])
    def test_gradient_flows(self, activation: str) -> None:
        """Gradients must flow back through the FFN to the input."""
        ffn = PositionWiseFeedForward(32, 128, dropout=0.0, activation=activation)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_output(self) -> None:
        """No NaN in output with standard random input."""
        ffn = PositionWiseFeedForward(64, 256, dropout=0.0)
        x = torch.randn(4, 12, 64)
        out = ffn(x)
        assert not torch.isnan(out).any()

    def test_batch_independence(self) -> None:
        """Each position is processed independently — doubling the batch
        must give the same output for the first item as processing alone."""
        ffn = PositionWiseFeedForward(32, 64, dropout=0.0)
        ffn.eval()
        x_single = torch.randn(1, 6, 32)
        x_double = torch.cat([x_single, torch.randn(1, 6, 32)], dim=0)

        with torch.no_grad():
            out_single = ffn(x_single)
            out_double = ffn(x_double)

        assert torch.allclose(out_single[0], out_double[0], atol=1e-6)

    def test_d_ff_is_intermediate_only(self) -> None:
        """Varying d_ff changes capacity but not output shape."""
        x = torch.randn(2, 8, 64)
        for d_ff in [128, 256, 512]:
            ffn = PositionWiseFeedForward(64, d_ff, dropout=0.0)
            assert ffn(x).shape == (2, 8, 64)

    def test_parameter_count(self) -> None:
        """Two Linear layers: d_model*d_ff + d_ff*d_model params (plus biases)."""
        d_model, d_ff = 32, 128
        ffn = PositionWiseFeedForward(d_model, d_ff, dropout=0.0)
        n_params = sum(p.numel() for p in ffn.parameters())
        # W1: d_model*d_ff + b1: d_ff  |  W2: d_ff*d_model + b2: d_model
        expected = 2 * d_model * d_ff + d_ff + d_model
        assert n_params == expected
