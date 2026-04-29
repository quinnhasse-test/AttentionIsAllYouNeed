"""Attention heatmap visualization.

Plots a 2D attention weight matrix as an annotated heatmap.
Useful for inspecting what source positions a decoder token attends to.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns  # type: ignore
from typing import List, Optional


def plot_attention(
    attention: np.ndarray,
    src_tokens: List[str],
    tgt_tokens: List[str],
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> matplotlib.figure.Figure:
    """Render a single attention head as a heatmap.

    Args:
        attention:  (tgt_len, src_len) float array of attention weights.
        src_tokens: Token strings for the x-axis (source/keys).
        tgt_tokens: Token strings for the y-axis (target/queries).
        title:      Optional figure title.
        figsize:    Matplotlib figure size tuple.

    Returns:
        Matplotlib Figure object (caller is responsible for closing).
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        attention,
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        ax=ax,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.0,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Source tokens (keys)")
    ax.set_ylabel("Target tokens (queries)")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_all_heads(
    attention_heads: np.ndarray,
    src_tokens: List[str],
    tgt_tokens: List[str],
    title: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Render all attention heads for one layer in a grid.

    Args:
        attention_heads: (n_heads, tgt_len, src_len) float array.
        src_tokens:      Token strings for the x-axis.
        tgt_tokens:      Token strings for the y-axis.
        title:           Optional suptitle.

    Returns:
        Matplotlib Figure object.
    """
    n_heads = attention_heads.shape[0]
    ncols = min(4, n_heads)
    nrows = math.ceil(n_heads / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).reshape(-1)

    for h in range(n_heads):
        sns.heatmap(
            attention_heads[h],
            xticklabels=src_tokens if h >= (nrows - 1) * ncols else [],
            yticklabels=tgt_tokens if h % ncols == 0 else [],
            ax=axes[h],
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            cbar=False,
        )
        axes[h].set_title(f"Head {h}", fontsize=9)

    # Hide unused axes
    for h in range(n_heads, len(axes)):
        axes[h].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


import math  # noqa: E402 — placed here to avoid circular-import confusion
