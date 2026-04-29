# Attention Is All You Need — Transformer from Scratch

PyTorch implementation of the original Transformer architecture described in Vaswani et al. (2017). Covers the full model: multi-head self-attention, positional encoding, encoder/decoder stacks, and a training loop on a toy sequence-to-sequence task.

**Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., NeurIPS 2017)

## Architecture

```
Input tokens
    └── Token embedding + positional encoding
            └── Encoder stack (N layers)
            │       └── Multi-head self-attention
            │       └── Position-wise feed-forward
            │       └── Layer norm + residual
            └── Decoder stack (N layers)
                    └── Masked multi-head self-attention
                    └── Cross-attention over encoder output
                    └── Position-wise feed-forward
                    └── Layer norm + residual
                            └── Linear projection + softmax → output tokens
```

**Key parameters (default config):**

| Parameter | Value |
|-----------|-------|
| `d_model` | 512 |
| `n_heads` | 8 |
| `n_layers` | 6 |
| `d_ff` | 2048 |
| `dropout` | 0.1 |
| `max_seq_len` | 512 |

## Usage

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, CPU supported)

### Install

```bash
git clone https://github.com/quinnhasse-test/AttentionIsAllYouNeed.git
cd AttentionIsAllYouNeed
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

Training runs a toy copy-task (sequence-to-sequence repetition) by default, logging loss per epoch to stdout.

### Inference

```bash
python predict.py --input "hello world"
```

## How it works

**Multi-head attention** splits the query/key/value matrices into `n_heads` subspaces, computes scaled dot-product attention in each, and concatenates the results. This lets the model attend to information from different representation subspaces simultaneously.

**Positional encoding** adds sine/cosine signals at different frequencies to each position in the sequence, giving the model order information without recurrence.

**The encoder** runs the input through N identical layers of self-attention + feed-forward. The **decoder** adds a cross-attention sublayer that attends to the encoder output, with masking to prevent attending to future positions.

## Testing

```bash
pytest tests/
```

Tests cover attention score shapes, masking correctness, and encoder/decoder output dimensions.

## Dependencies

```
torch
numpy
```

Install with `pip install -r requirements.txt`.
