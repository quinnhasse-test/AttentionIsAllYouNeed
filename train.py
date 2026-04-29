#!/usr/bin/env python3
"""Training entry point for the Multi30k EN-DE transformer.

Usage:
    python train.py --config configs/base_small.yaml
    python train.py --config configs/base.yaml --no-wandb

The script loads the dataset, builds vocabularies, constructs the model,
and runs the training loop.  If W&B is installed and WANDB_API_KEY is set,
metrics and attention heatmaps are logged automatically.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer on Multi30k EN-DE")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_small.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging even if available",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    from src.data.dataset import build_dataloaders

    print("Loading Multi30k dataset and building vocabularies...")
    loaders, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=cfg["training"]["batch_size"],
        max_len=cfg["training"]["max_src_len"],
        min_freq=cfg["data"].get("min_freq", cfg["training"]["min_freq"]),
    )
    print(f"  src vocab: {len(src_vocab)} tokens")
    print(f"  tgt vocab: {len(tgt_vocab)} tokens")
    print(f"  train batches: {len(loaders['train'])}")

    # --- Model ---
    from src.model.transformer import Transformer

    model_cfg = cfg["model"]
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        max_len=model_cfg["max_len"],
        dropout=model_cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}")

    # --- Optimizer and scheduler ---
    from src.training import NoamScheduler
    from src.training.loss import LabelSmoothingLoss

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=model_cfg["d_model"],
        warmup_steps=cfg["training"]["warmup_steps"],
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        smoothing=cfg["training"]["label_smoothing"],
    )

    # --- W&B ---
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb  # type: ignore

            wandb.init(
                project=cfg["wandb"]["project"],
                name=cfg["wandb"]["run_name"],
                config=cfg,
            )
            wandb.watch(model, log="gradients", log_freq=200)
        except Exception as e:
            print(f"W&B init failed ({e}); continuing without logging")
            use_wandb = False

    # --- Trainer ---
    from src.training.trainer import Trainer

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=cfg["training"]["checkpoint_dir"],
        use_wandb=use_wandb,
    )

    trainer.fit(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        n_epochs=cfg["training"]["n_epochs"],
        log_attention_every=cfg["wandb"]["log_attention_every"],
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
    )

    if use_wandb:
        import wandb  # type: ignore
        wandb.finish()

    print("Training complete. Best checkpoint saved to:", cfg["training"]["checkpoint_dir"])


if __name__ == "__main__":
    main()
