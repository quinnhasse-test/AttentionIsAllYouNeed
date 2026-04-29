"""Position-wise feed-forward sublayer from Vaswani et al. (2017).

FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

The same two-layer network is applied independently to each position.
In the base model: d_model=512, d_ff=2048.
"""

import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    """Two-layer point-wise feed-forward network with ReLU activation.

    Applied identically at each sequence position; no interaction across
    positions here (that is the role of attention).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network at every position.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        return self.net(x)
