"""Vocabulary: maps tokens to integer ids and back.

Special tokens:
    <pad> = 0  — used to right-pad sequences to equal length
    <unk> = 1  — replaces tokens not seen at training time
    <bos> = 2  — beginning-of-sequence marker
    <eos> = 3  — end-of-sequence marker
"""

from __future__ import annotations
from collections import Counter
from typing import Dict, Iterable, List, Optional


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

SPECIALS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class Vocabulary:
    """Word-to-id and id-to-word mappings built from a token stream."""

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._init_specials()

    def _init_specials(self) -> None:
        for idx, tok in enumerate(SPECIALS):
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

    def build(self, token_sequences: Iterable[List[str]], min_freq: int = 2) -> None:
        """Populate vocabulary from an iterable of tokenized sentences.

        Tokens appearing fewer than *min_freq* times across all sentences
        are discarded and map to <unk> at encode time.

        Args:
            token_sequences: Iterable of lists of string tokens.
            min_freq:        Minimum token frequency to include.
        """
        counts: Counter = Counter()
        for tokens in token_sequences:
            counts.update(tokens)

        for token, freq in sorted(counts.items()):
            if freq >= min_freq and token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def encode(self, tokens: List[str], add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Convert a token list to a list of integer ids.

        Args:
            tokens:  Pre-tokenized list of strings.
            add_bos: Prepend the BOS id.
            add_eos: Append the EOS id.

        Returns:
            List of integer token ids.
        """
        ids = [self.token_to_id.get(t, UNK_IDX) for t in tokens]
        if add_bos:
            ids = [BOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        """Convert a list of integer ids back to a string.

        Args:
            ids:           List of integer token ids.
            strip_special: If True, omit BOS/EOS/PAD tokens from output.

        Returns:
            Reconstructed sentence string.
        """
        specials = {PAD_IDX, BOS_IDX, EOS_IDX}
        tokens = []
        for i in ids:
            if strip_special and i in specials:
                continue
            tokens.append(self.id_to_token.get(i, UNK_TOKEN))
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id
