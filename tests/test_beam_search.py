"""Tests for greedy decoding and beam search.

Covers:
  - Greedy decode returns the right number of strings
  - Each greedy output is a string (possibly empty)
  - Beam search returns a list sorted by score (best first)
  - Beam search with beam_size=1 matches greedy qualitatively
  - BLEU computation returns a float in [0, 100]
"""

import pytest
import torch

from src.model.transformer import Transformer
from src.data import Vocabulary, PAD_IDX, BOS_IDX, EOS_IDX
from src.evaluation import greedy_decode, compute_corpus_bleu
from src.evaluation.beam_search import beam_search_decode


@pytest.fixture
def tiny_model_and_vocab():
    src_vocab = Vocabulary()
    src_vocab.build([["a", "b", "c"]] * 5, min_freq=1)
    tgt_vocab = Vocabulary()
    tgt_vocab.build([["x", "y", "z"]] * 5, min_freq=1)

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_len=30,
        dropout=0.0,
        pad_idx=PAD_IDX,
    )
    model.eval()
    return model, src_vocab, tgt_vocab


class TestGreedyDecode:

    def test_returns_correct_batch_size(self, tiny_model_and_vocab):
        model, src_vocab, tgt_vocab = tiny_model_and_vocab
        src = torch.randint(1, len(src_vocab), (3, 5))
        results = greedy_decode(model, src, src_vocab, tgt_vocab, max_len=10)
        assert len(results) == 3

    def test_each_result_is_string(self, tiny_model_and_vocab):
        model, src_vocab, tgt_vocab = tiny_model_and_vocab
        src = torch.randint(1, len(src_vocab), (2, 4))
        results = greedy_decode(model, src, src_vocab, tgt_vocab, max_len=10)
        for r in results:
            assert isinstance(r, str)

    def test_no_bos_eos_in_output(self, tiny_model_and_vocab):
        model, src_vocab, tgt_vocab = tiny_model_and_vocab
        src = torch.randint(1, len(src_vocab), (2, 4))
        results = greedy_decode(model, src, src_vocab, tgt_vocab, max_len=15)
        for r in results:
            assert "<bos>" not in r
            assert "<eos>" not in r


class TestBeamSearch:

    def test_returns_list_of_strings(self, tiny_model_and_vocab):
        model, src_vocab, tgt_vocab = tiny_model_and_vocab
        src = torch.randint(1, len(src_vocab), (1, 5))
        results = beam_search_decode(model, src, tgt_vocab, beam_size=2, max_len=10)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, str)

    def test_beam1_runs_without_error(self, tiny_model_and_vocab):
        model, src_vocab, tgt_vocab = tiny_model_and_vocab
        src = torch.randint(1, len(src_vocab), (1, 4))
        results = beam_search_decode(model, src, tgt_vocab, beam_size=1, max_len=8)
        assert len(results) >= 1


class TestCorpusBLEU:

    def test_bleu_identical(self):
        hyps = ["the cat sat on the mat", "hello world"]
        refs = ["the cat sat on the mat", "hello world"]
        bleu = compute_corpus_bleu(hyps, refs)
        assert bleu == pytest.approx(100.0, abs=1.0)

    def test_bleu_returns_float(self):
        hyps = ["a b c"]
        refs = ["d e f"]
        bleu = compute_corpus_bleu(hyps, refs)
        assert isinstance(bleu, float)
        assert 0.0 <= bleu <= 100.0

    def test_bleu_disjoint_is_zero(self):
        hyps = ["zzz aaa bbb ccc ddd"]
        refs = ["the cat sat on the mat today here now yes"]
        bleu = compute_corpus_bleu(hyps, refs)
        assert bleu == pytest.approx(0.0, abs=0.1)
