#!/usr/bin/env python3
"""Evaluation script: load a trained checkpoint and compute BLEU on Multi30k test.

Usage:
    python evaluate.py --checkpoint checkpoints/base_small/transformer_best.pt \\
                       --config configs/base_small.yaml

Outputs:
    - Corpus BLEU on the test split
    - Ten sample translations (source | hypothesis | reference)
    - Saved attention heatmap PNGs in figures/
"""

import argparse
import os
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate transformer on Multi30k test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_small.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam width for decoding (default 4)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Number of sample translations to print",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save attention heatmaps to figures/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    from src.data.dataset import build_dataloaders

    print("Loading dataset...")
    loaders, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=64,
        max_len=cfg["training"]["max_src_len"],
        min_freq=cfg["training"]["min_freq"],
    )

    # Build model and load weights
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
        dropout=0.0,  # no dropout at eval time
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val loss {ckpt['val_loss']:.4f})")

    # Decode test set
    from src.evaluation.beam_search import beam_search_decode
    from src.evaluation import compute_corpus_bleu

    hypotheses = []
    references = []

    test_loader = loaders["test"]
    print(f"Decoding {len(test_loader.dataset)} test sentences (beam={args.beam_size})...")

    for src_batch, tgt_batch in test_loader:
        for i in range(src_batch.size(0)):
            src_single = src_batch[i : i + 1].to(device)
            hyp = beam_search_decode(
                model,
                src_single,
                tgt_vocab,
                beam_size=args.beam_size,
                max_len=cfg["model"]["max_len"],
                device=device,
            )[0]
            ref_ids = tgt_batch[i].tolist()
            ref = tgt_vocab.decode(ref_ids, strip_special=True)
            hypotheses.append(hyp)
            references.append(ref)

    bleu = compute_corpus_bleu(hypotheses, references)
    print(f"\nCorpus BLEU: {bleu:.2f}")

    # Sample translations
    print("\nSample translations (source | hypothesis | reference):")
    print("-" * 80)
    for idx in range(min(args.max_examples, len(hypotheses))):
        src_ids = test_loader.dataset.src[idx].tolist()
        src_str = src_vocab.decode(src_ids, strip_special=True)
        print(f"SRC: {src_str}")
        print(f"HYP: {hypotheses[idx]}")
        print(f"REF: {references[idx]}")
        print()

    # Attention heatmaps
    if args.save_figures:
        from src.evaluation.visualize import plot_all_heads
        import matplotlib.pyplot as plt

        Path("figures").mkdir(exist_ok=True)
        src_single = test_loader.dataset.src[0].unsqueeze(0).to(device)
        tgt_single = test_loader.dataset.tgt[0].unsqueeze(0).to(device)

        with torch.no_grad():
            model(src_single, tgt_single[:, :-1])

        src_tokens = [src_vocab.id_to_token.get(i.item(), "?") for i in src_single[0]]
        enc_attn = model.encoder.layers[-1].self_attn.attn_weights
        if enc_attn is not None:
            fig = plot_all_heads(
                enc_attn[0].cpu().numpy(),
                src_tokens,
                src_tokens,
                title="Encoder last layer — all heads",
            )
            fig.savefig("figures/encoder_attention.png", dpi=150)
            plt.close(fig)
            print("Saved figures/encoder_attention.png")


if __name__ == "__main__":
    main()
