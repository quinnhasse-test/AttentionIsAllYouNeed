"""Label smoothing cross-entropy loss.

Replaces the hard one-hot target distribution with a soft one:
  p_smooth(y | x) = (1 - eps) * delta(y, y*) + eps / (V - 1)

where V is the vocabulary size, y* the ground-truth token, and eps the
smoothing coefficient (0.1 in the original paper).

Padding positions (pad_idx) are excluded from the loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing and pad token masking.

    Args:
        vocab_size: Total vocabulary size (number of output classes).
        pad_idx:    Token id to ignore in the loss computation.
        smoothing:  Smoothing coefficient epsilon (default 0.1).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute smoothed cross-entropy loss.

        Args:
            logits:  (batch * tgt_len, vocab_size) unnormalized logits.
            targets: (batch * tgt_len,) integer token ids.

        Returns:
            Scalar mean loss over non-pad positions.
        """
        # Build soft target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(
                logits, self.smoothing / (self.vocab_size - 2)
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
            smooth_targets[:, self.pad_idx] = 0.0

        # KL divergence: sum_y p(y) * (log p(y) - log q(y))
        # Since p is fixed, minimizing -sum_y p(y) log q(y) is equivalent
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Mask out padding positions
        pad_mask = targets.eq(self.pad_idx)
        loss = loss.masked_fill(pad_mask, 0.0)

        n_tokens = (~pad_mask).sum().float().clamp(min=1)
        return loss.sum() / n_tokens
