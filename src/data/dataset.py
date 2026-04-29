"""Multi30k EN-DE dataset loader.

Uses the HuggingFace `datasets` library to fetch bentrevett/multi30k,
builds source (EN) and target (DE) vocabularies from the training split,
and returns PyTorch DataLoaders for train / validation / test.

Splits:
    train: 29,000 sentence pairs
    validation: 1,014 pairs
    test: 1,000 pairs

Usage:
    from src.data.dataset import build_dataloaders
    loaders, src_vocab, tgt_vocab = build_dataloaders(batch_size=64)
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

from src.data import Vocabulary, PAD_IDX
from src.data.tokenizer import tokenize


class TranslationDataset(Dataset):
    """Holds encoded (src_ids, tgt_ids) pairs as integer tensors."""

    def __init__(
        self,
        src_encoded: List[List[int]],
        tgt_encoded: List[List[int]],
    ) -> None:
        assert len(src_encoded) == len(tgt_encoded)
        self.src = [torch.tensor(ids, dtype=torch.long) for ids in src_encoded]
        self.tgt = [torch.tensor(ids, dtype=torch.long) for ids in tgt_encoded]

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.src[idx], self.tgt[idx]


def _collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of (src, tgt) tensor pairs to equal length within the batch.

    Sequences are right-padded with PAD_IDX=0.

    Returns:
        src: (batch, max_src_len)
        tgt: (batch, max_tgt_len)
    """
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def build_dataloaders(
    batch_size: int = 64,
    max_len: int = 100,
    min_freq: int = 2,
    num_workers: int = 0,
) -> Tuple[dict, Vocabulary, Vocabulary]:
    """Download Multi30k, build vocabularies, return DataLoaders.

    Args:
        batch_size:  Number of sentence pairs per batch.
        max_len:     Discard sentence pairs where either side exceeds this
                     many tokens (after tokenization, before BOS/EOS).
        min_freq:    Minimum token frequency to include in vocabulary.
        num_workers: DataLoader worker processes (0 = main process).

    Returns:
        loaders:   Dict with keys "train", "val", "test" -> DataLoader.
        src_vocab: English vocabulary (Vocabulary instance).
        tgt_vocab: German vocabulary (Vocabulary instance).
    """
    from datasets import load_dataset  # type: ignore

    raw = load_dataset("bentrevett/multi30k")

    def tok_pair(example):
        return tokenize(example["en"]), tokenize(example["de"])

    # Build vocabularies from training split
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    train_pairs = [tok_pair(ex) for ex in raw["train"]]
    src_vocab.build((p[0] for p in train_pairs), min_freq=min_freq)
    tgt_vocab.build((p[1] for p in train_pairs), min_freq=min_freq)

    def encode_split(examples, filter_len: bool = True):
        src_enc, tgt_enc = [], []
        for en, de in examples:
            if filter_len and (len(en) > max_len or len(de) > max_len):
                continue
            src_enc.append(src_vocab.encode(en))
            tgt_enc.append(tgt_vocab.encode(de))
        return src_enc, tgt_enc

    val_pairs = [tok_pair(ex) for ex in raw["validation"]]
    test_pairs = [tok_pair(ex) for ex in raw["test"]]

    train_src, train_tgt = encode_split(train_pairs)
    val_src, val_tgt = encode_split(val_pairs, filter_len=False)
    test_src, test_tgt = encode_split(test_pairs, filter_len=False)

    loaders = {
        "train": DataLoader(
            TranslationDataset(train_src, train_tgt),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            TranslationDataset(val_src, val_tgt),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            TranslationDataset(test_src, test_tgt),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
            num_workers=num_workers,
        ),
    }
    return loaders, src_vocab, tgt_vocab
