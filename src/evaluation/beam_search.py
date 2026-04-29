"""Beam search decoder for the transformer.

Standard left-to-right beam search: at each step, keep the top-k
partial hypotheses by cumulative log-probability.

Length normalization:
    score = log_prob / (length ^ alpha)

where alpha=0.6 is the value used in Wu et al. (2016) and roughly
reproduces the behavior of the original "Attention Is All You Need"
paper's evaluation setup.
"""

from __future__ import annotations

import math
import torch
from typing import List, Tuple

from src.model.transformer import Transformer, make_src_mask, make_tgt_mask
from src.data import Vocabulary, BOS_IDX, EOS_IDX, PAD_IDX


def beam_search_decode(
    model: Transformer,
    src: torch.Tensor,
    tgt_vocab: Vocabulary,
    beam_size: int = 4,
    max_len: int = 100,
    alpha: float = 0.6,
    device: torch.device = torch.device("cpu"),
) -> List[str]:
    """Decode one source sentence using beam search.

    Processes one sequence at a time (batch_size=1 expected).

    Args:
        model:     Trained Transformer model in eval mode.
        src:       (1, src_len) integer token ids.
        tgt_vocab: Target Vocabulary instance.
        beam_size: Number of beams (default 4).
        max_len:   Maximum number of tokens to decode.
        alpha:     Length normalization exponent.
        device:    Device for computation.

    Returns:
        List of decoded strings, sorted by score (best first).
        Typically you want the first element.
    """
    model.eval()
    src = src.to(device)
    assert src.size(0) == 1, "beam_search_decode processes one sentence at a time"

    src_mask = make_src_mask(src, model.pad_idx)
    memory = model.encode(src, src_mask)  # (1, src_len, d_model)

    # Each beam: (sequence tensor, cumulative log-prob)
    beams: List[Tuple[torch.Tensor, float]] = [
        (torch.tensor([BOS_IDX], dtype=torch.long, device=device), 0.0)
    ]
    completed: List[Tuple[torch.Tensor, float]] = []

    for _ in range(max_len):
        if len(beams) == 0:
            break

        # Expand all beams in a single batch call
        seqs = torch.stack([b[0] for b in beams])  # (n_beams, cur_len)
        n_beams = seqs.size(0)

        # Repeat memory for each beam
        mem_expanded = memory.expand(n_beams, -1, -1)
        src_mask_expanded = src_mask.expand(n_beams, -1, -1, -1)

        tgt_mask = make_tgt_mask(seqs, model.pad_idx)
        dec_out = model.decode(seqs, mem_expanded, src_mask_expanded, tgt_mask)
        logits = model.output_proj(dec_out[:, -1, :])  # (n_beams, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)  # (n_beams, vocab)

        all_candidates: List[Tuple[torch.Tensor, float]] = []

        for i, (seq, score) in enumerate(beams):
            top_log_probs, top_tokens = log_probs[i].topk(beam_size)
            for lp, token in zip(top_log_probs.tolist(), top_tokens.tolist()):
                new_seq = torch.cat([seq, torch.tensor([token], device=device)])
                new_score = score + lp
                if token == EOS_IDX:
                    length = new_seq.size(0) - 1  # exclude BOS
                    norm_score = new_score / (length ** alpha)
                    completed.append((new_seq, norm_score))
                else:
                    all_candidates.append((new_seq, new_score))

        # Keep top beam_size candidates by raw score (length norm at the end)
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_size]

        if len(completed) >= beam_size:
            break

    # If no beam completed with EOS, take the best partial hypothesis
    if not completed:
        completed = [
            (seq, score / (seq.size(0) ** alpha)) for seq, score in beams
        ]

    completed.sort(key=lambda x: x[1], reverse=True)

    results = []
    for seq, _ in completed:
        ids = seq.tolist()
        try:
            eos_pos = ids.index(EOS_IDX)
            ids = ids[1:eos_pos]
        except ValueError:
            ids = ids[1:]
        results.append(tgt_vocab.decode(ids, strip_special=True))

    return results
