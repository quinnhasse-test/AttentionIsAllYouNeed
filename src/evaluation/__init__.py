"""Evaluation utilities: greedy decoding and corpus BLEU.

Greedy decoding is used during training for quick sanity checks.
Beam search (in beam_search.py) is used for final evaluation.

BLEU is computed with sacrebleu's corpus_bleu for standard tokenization.
"""

from __future__ import annotations

import torch
from typing import List

from src.model.transformer import Transformer, make_src_mask
from src.data import Vocabulary, BOS_IDX, EOS_IDX, PAD_IDX


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    max_len: int = 100,
    device: torch.device = torch.device("cpu"),
) -> List[str]:
    """Decode a batch of source sequences using greedy argmax decoding.

    At each step, the most probable next token is selected and fed back
    into the decoder.  Decoding stops when EOS is produced or max_len
    is reached.

    Args:
        model:      Trained Transformer model in eval mode.
        src:        (batch, src_len) integer token ids.
        src_vocab:  Source Vocabulary instance.
        tgt_vocab:  Target Vocabulary instance.
        max_len:    Maximum number of tokens to generate.
        device:     Device to run decoding on.

    Returns:
        List of decoded sentence strings, one per batch item.
    """
    model.eval()
    src = src.to(device)
    batch_size = src.size(0)

    src_mask = make_src_mask(src, model.pad_idx)
    memory = model.encode(src, src_mask)

    # Start all sequences with BOS
    ys = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        from src.model.transformer import make_tgt_mask
        tgt_mask = make_tgt_mask(ys, model.pad_idx)
        dec_out = model.decode(ys, memory, src_mask, tgt_mask)
        logits = model.output_proj(dec_out[:, -1, :])  # (batch, vocab)
        next_token = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)

        # Replace next_token with PAD for already-finished sequences
        next_token = next_token.masked_fill(finished.unsqueeze(1), PAD_IDX)
        ys = torch.cat([ys, next_token], dim=1)

        # Mark sequences that just produced EOS
        finished |= (next_token.squeeze(1) == EOS_IDX)
        if finished.all():
            break

    outputs = []
    for i in range(batch_size):
        ids = ys[i].tolist()
        # Strip BOS; stop at EOS
        try:
            eos_pos = ids.index(EOS_IDX)
            ids = ids[1:eos_pos]
        except ValueError:
            ids = ids[1:]
        outputs.append(tgt_vocab.decode(ids, strip_special=True))

    return outputs


def compute_corpus_bleu(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Compute corpus-level BLEU using sacrebleu.

    Args:
        hypotheses: List of model-generated translation strings.
        references: List of reference translation strings.

    Returns:
        BLEU score as a float (0–100).
    """
    import sacrebleu  # type: ignore

    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result.score
