"""Training and validation loop for the transformer.

Handles:
  - Teacher-forced training pass
  - Gradient clipping (max norm 1.0)
  - LR warmup via NoamScheduler
  - Per-epoch train/val loss tracking
  - Optional W&B logging (skipped if wandb is not configured)
  - Checkpoint saving on validation loss improvement
"""

from __future__ import annotations

import os
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.transformer import Transformer
from src.training import NoamScheduler
from src.training.loss import LabelSmoothingLoss
from src.data import PAD_IDX, BOS_IDX, EOS_IDX


class Trainer:
    """Manages the full training lifecycle for the transformer.

    Args:
        model:         The Transformer model.
        optimizer:     Optimizer (Adam recommended).
        scheduler:     NoamScheduler instance.
        loss_fn:       LabelSmoothingLoss instance.
        device:        torch.device to run on.
        checkpoint_dir: Directory to save model checkpoints.
        use_wandb:     If True, log metrics to W&B (requires prior wandb.init()).
    """

    def __init__(
        self,
        model: Transformer,
        optimizer: torch.optim.Optimizer,
        scheduler: NoamScheduler,
        loss_fn: LabelSmoothingLoss,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_wandb = use_wandb
        self.best_val_loss = float("inf")
        self._global_step = 0

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch, return {"loss": ..., "ppl": ...}."""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0

        for src, tgt in tqdm(loader, desc="  train", leave=False):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Teacher forcing: feed tgt[:-1] as input, predict tgt[1:]
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = self.model(src, tgt_in)
            # Flatten for loss: (batch * tgt_len, vocab_size)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = tgt_out.reshape(-1)

            loss = self.loss_fn(logits_flat, targets_flat)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            lr = self.scheduler.step()
            self._global_step += 1

            n_tokens = (tgt_out != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            if self.use_wandb:
                self._log_wandb({"train/step_loss": loss.item(), "train/lr": lr})

        avg_loss = total_loss / max(total_tokens, 1)
        return {"loss": avg_loss, "ppl": math.exp(min(avg_loss, 100))}

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one validation epoch, return {"loss": ..., "ppl": ...}."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for src, tgt in tqdm(loader, desc="  val  ", leave=False):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = self.model(src, tgt_in)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = tgt_out.reshape(-1)

            loss = self.loss_fn(logits_flat, targets_flat)

            n_tokens = (tgt_out != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        return {"loss": avg_loss, "ppl": math.exp(min(avg_loss, 100))}

    def _save_checkpoint(self, epoch: int, val_loss: float, label: str = "best") -> None:
        path = self.checkpoint_dir / f"transformer_{label}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "global_step": self._global_step,
            },
            path,
        )

    def _log_wandb(self, metrics: dict, step: Optional[int] = None) -> None:
        try:
            import wandb  # type: ignore
            wandb.log(metrics, step=step or self._global_step)
        except Exception:
            pass

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        log_attention_every: int = 5,
        src_vocab=None,
        tgt_vocab=None,
    ) -> None:
        """Train for *n_epochs*, saving the best checkpoint by validation loss.

        Args:
            train_loader:        DataLoader for training split.
            val_loader:          DataLoader for validation split.
            n_epochs:            Number of full passes over the training data.
            log_attention_every: Log attention heatmaps to W&B every N epochs.
            src_vocab:           Source vocabulary (used for W&B attention logging).
            tgt_vocab:           Target vocabulary (used for W&B attention logging).
        """
        print(f"Training for {n_epochs} epochs on {self.device}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._val_epoch(val_loader)
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"train loss {train_metrics['loss']:.4f} ppl {train_metrics['ppl']:.1f} | "
                f"val loss {val_metrics['loss']:.4f} ppl {val_metrics['ppl']:.1f} | "
                f"{elapsed:.0f}s"
            )

            if self.use_wandb:
                self._log_wandb(
                    {
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"],
                        "train/ppl": train_metrics["ppl"],
                        "val/loss": val_metrics["loss"],
                        "val/ppl": val_metrics["ppl"],
                    }
                )

                if epoch % log_attention_every == 0 and src_vocab and tgt_vocab:
                    self._log_attention_heatmap(val_loader, src_vocab, tgt_vocab)

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, val_metrics["loss"], label="best")
                print(f"  -> new best val loss {self.best_val_loss:.4f}, checkpoint saved")

        self._save_checkpoint(n_epochs, val_metrics["loss"], label="final")

    @torch.no_grad()
    def _log_attention_heatmap(self, loader: DataLoader, src_vocab, tgt_vocab) -> None:
        """Log one attention heatmap from the first batch to W&B."""
        from src.evaluation.visualize import plot_attention

        try:
            import wandb  # type: ignore

            self.model.eval()
            src, tgt = next(iter(loader))
            src = src[:1].to(self.device)
            tgt = tgt[:1].to(self.device)

            tgt_in = tgt[:, :-1]
            self.model(src, tgt_in)

            # Grab last encoder layer attention weights
            attn_weights = self.model.encoder.layers[-1].self_attn.attn_weights
            if attn_weights is not None:
                src_tokens = [src_vocab.id_to_token.get(i.item(), "?") for i in src[0]]
                fig = plot_attention(
                    attn_weights[0, 0].cpu().numpy(),
                    src_tokens,
                    src_tokens,
                    title="Encoder self-attention (last layer, head 0)",
                )
                wandb.log({"attention/encoder_head0": wandb.Image(fig)})
                import matplotlib.pyplot as plt
                plt.close(fig)
        except Exception:
            pass
