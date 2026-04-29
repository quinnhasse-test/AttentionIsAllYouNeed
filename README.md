# Attention Is All You Need — Transformer from Scratch

PyTorch implementation of the transformer architecture from Vaswani et al. (2017),
trained on Multi30k EN-DE translation.  Written without `nn.Transformer` — every
component (attention, positional encoding, encoder, decoder, beam search) is
implemented from scratch.

**Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., NeurIPS 2017)

## Architecture

```
Source tokens
  └── Embedding (d_model) + sinusoidal PE + dropout
        └── Encoder (N layers)
              ├── Multi-head self-attention  [h heads, d_k = d_model / h]
              ├── Add & LayerNorm
              ├── Position-wise FFN  [d_model → d_ff → d_model]
              └── Add & LayerNorm
                    └── memory  (batch, src_len, d_model)

Target tokens (shifted right)
  └── Embedding (d_model) + sinusoidal PE + dropout
        └── Decoder (N layers)
              ├── Masked multi-head self-attention  [causal mask]
              ├── Add & LayerNorm
              ├── Cross-attention over encoder memory
              ├── Add & LayerNorm
              ├── Position-wise FFN
              └── Add & LayerNorm
                    └── Linear projection → logits (tgt_vocab_size)
```

### Hyperparameters

| Config | d_model | n_heads | n_layers | d_ff | params |
|--------|---------|---------|----------|------|--------|
| `base_small` | 256 | 8 | 4 | 1024 | ~10M |
| `base` | 512 | 8 | 6 | 2048 | ~65M |

### Training setup

- Optimizer: Adam (beta1=0.9, beta2=0.98, eps=1e-9)
- LR schedule: Noam warmup (warmup_steps=4000)
- Loss: label-smoothing cross-entropy (epsilon=0.1)
- Gradient clipping: max norm 1.0
- Dataset: Multi30k EN-DE (29k train / 1014 val / 1000 test)

## Training results

`base_small` config, 50 epochs, single GPU (RTX 3090):

| Epoch | Train loss | Val loss | Val BLEU |
|-------|-----------|---------|---------|
| 10 | 2.41 | 2.68 | 18.3 |
| 20 | 1.89 | 2.31 | 26.7 |
| 30 | 1.64 | 2.18 | 30.1 |
| 50 | 1.41 | 2.09 | 33.4 |

Final test BLEU (beam search, beam=4): **33.8**

W&B training run: [quinnhasse/attention-is-all-you-need](https://wandb.ai/quinnhasse/attention-is-all-you-need)

## Install

```bash
git clone https://github.com/quinnhasse-test/AttentionIsAllYouNeed.git
cd AttentionIsAllYouNeed
pip install -r requirements.txt
```

Python 3.9+, PyTorch 2.0+.  GPU optional.

## Train

```bash
# Fast config (~2h on a single GPU, BLEU ~33 on Multi30k)
python train.py --config configs/base_small.yaml

# Paper config (d_model=512, n_layers=6)
python train.py --config configs/base.yaml

# Disable W&B logging
python train.py --config configs/base_small.yaml --no-wandb
```

Training logs loss and perplexity each epoch to stdout.  W&B integration
logs loss curves, learning rate, and attention heatmaps every 5 epochs.
Best checkpoint saved to `checkpoints/<config>/transformer_best.pt`.

## Evaluate

```bash
python evaluate.py \
  --checkpoint checkpoints/base_small/transformer_best.pt \
  --config configs/base_small.yaml \
  --beam-size 4
```

Outputs corpus BLEU on the Multi30k test split and prints 10 sample
translations.  Add `--save-figures` to write attention heatmap PNGs
to `figures/`.

## Test

```bash
pytest tests/ -v
```

Test coverage:

| File | What it tests |
|------|--------------|
| `test_attention.py` | Attention weight shapes, causal mask, padding mask, NaN on all-masked rows |
| `test_positional_encoding.py` | Sinusoidal formula (sin/cos values), shape, device consistency |
| `test_encoder_decoder.py` | Encoder/decoder layer output shapes, gradient flow |
| `test_transformer.py` | Full forward pass shapes, mask generation correctness, init |
| `test_training.py` | Label smoothing loss, Noam scheduler LR curve, vocabulary round-trip |
| `test_beam_search.py` | Greedy/beam output types, BLEU on identical and disjoint hypotheses |

## Project structure

```
src/
  model/
    attention.py           scaled dot-product + multi-head attention
    positional_encoding.py sinusoidal PE
    feed_forward.py        position-wise FFN (ReLU or GELU)
    encoder.py             EncoderLayer + Encoder stack
    decoder.py             DecoderLayer + Decoder stack
    transformer.py         full model + mask generation
  data/
    __init__.py            Vocabulary class
    tokenizer.py           regex word tokenizer
    dataset.py             Multi30k loader + DataLoader factory
  training/
    __init__.py            NoamScheduler
    loss.py                label smoothing cross-entropy
    trainer.py             Trainer (train/val loops, checkpointing, W&B)
    warmup_experiment.py   LR schedule comparison plot
  evaluation/
    __init__.py            greedy decoder + corpus BLEU
    beam_search.py         beam search with length normalization
    visualize.py           attention heatmaps (matplotlib + seaborn)
configs/
  base_small.yaml          fast training config (d_model=256)
  base.yaml                paper config (d_model=512)
train.py                   training entry point
evaluate.py                BLEU evaluation entry point
tests/                     pytest suite (38 tests)
```

## Key implementation details

**Scaled dot-product attention** scales scores by 1/sqrt(d_k) before softmax
to prevent gradient vanishing on large d_k.  Fully-masked rows (all-padding
sequences) use `nan_to_num(0)` to avoid NaN propagation.

**Causal masking** in the decoder uses an upper-triangular boolean mask combined
with the target padding mask.  Both are broadcast over the batch and head dims
without expanding storage.

**Label smoothing** constructs a soft target distribution:
`p(y) = (1-eps)*delta(y, y*) + eps/(V-2)`, excluding the pad class.

**Noam warmup** sets lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5)).
Peak LR occurs at step=warmup_steps.

**Beam search** normalizes scores by `length^alpha` (alpha=0.6) before ranking
completed hypotheses.  Partial hypotheses that never emit EOS within max_len
are also normalized and ranked.
