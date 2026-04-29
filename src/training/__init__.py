"""Noam learning rate scheduler from Vaswani et al. (2017).

lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

The rate increases linearly for the first *warmup_steps* steps, then
decays proportionally to the inverse square root of the step number.

Usage:
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)
    # inside training loop:
    scheduler.step()
"""

import torch.optim as optim


class NoamScheduler:
    """Step-based LR scheduler that implements the Noam warmup schedule."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0

    def step(self) -> float:
        """Advance one training step and update optimizer LR.

        Returns:
            The new learning rate.
        """
        self._step += 1
        lr = self._compute_lr(self._step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _compute_lr(self, step: int) -> float:
        return self.factor * (
            self.d_model ** (-0.5)
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

    @property
    def last_lr(self) -> float:
        """Most recently set learning rate."""
        return self._compute_lr(max(self._step, 1))
