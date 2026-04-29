"""Position-wise feed-forward sublayer from Vaswani et al. (2017).

FFN(x) = activation(x W_1 + b_1) W_2 + b_2

The original paper uses ReLU.  GELU is commonly used in more recent
transformer variants (BERT, GPT) and tends to train slightly better.
"""

import torch
import torch.nn as nn
from typing import Literal


class PositionWiseFeedForward(nn.Module):
    """Two-layer point-wise feed-forward network.

    Applied identically at each sequence position.

    Args:
        d_model:    Input and output dimensionality.
        d_ff:       Hidden dimensionality (typically 4 * d_model).
        dropout:    Dropout applied after the activation.
        activation: "relu" (default) or "gelu".
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "relu",
    ) -> None:
        super().__init__()
        act: nn.Module = nn.ReLU() if activation == "relu" else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
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
