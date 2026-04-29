"""Compare learning rate schedules: Noam warmup, cosine decay, and constant.

Run this script to generate a plot of LR vs step for each schedule.
Useful for understanding the effect of warmup on early training stability.

Usage:
    python -m src.training.warmup_experiment
"""

from __future__ import annotations

import math
import argparse
from typing import List


def noam_lr(step: int, d_model: int, warmup: int) -> float:
    """Noam schedule: linear warmup then inverse-sqrt decay."""
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))


def cosine_lr(step: int, max_steps: int, base_lr: float = 1e-3, min_lr: float = 1e-5) -> float:
    """Cosine annealing without restarts."""
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * step / max_steps))


def constant_lr(step: int, lr: float = 1e-3) -> float:  # noqa: ARG001
    return lr


def plot_schedules(n_steps: int = 20000, d_model: int = 256, warmup: int = 4000) -> None:
    steps = list(range(1, n_steps + 1))
    noam = [noam_lr(s, d_model, warmup) for s in steps]
    cosine = [cosine_lr(s, n_steps) for s in steps]
    const = [constant_lr(s) for s in steps]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, noam, label=f"Noam (d={d_model}, warmup={warmup})", linewidth=1.5)
        ax.plot(steps, cosine, label="Cosine decay", linewidth=1.5)
        ax.plot(steps, const, label="Constant 1e-3", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning rate")
        ax.set_title("LR schedule comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/lr_schedules.png", dpi=150)
        print("Saved figures/lr_schedules.png")
    except ImportError:
        # Print peak values instead
        print(f"Noam peak: {max(noam):.6f} at step {noam.index(max(noam))+1}")
        print(f"Cosine start: {cosine[0]:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=4000)
    args = parser.parse_args()
    plot_schedules(args.steps, args.d_model, args.warmup)
