"""Tests for training utilities and vocabulary.

Covers:
  - LabelSmoothingLoss output is scalar and >= 0
  - LabelSmoothingLoss ignores pad positions
  - NoamScheduler LR increases during warmup then decreases
  - NoamScheduler peak LR is at warmup_steps
  - Vocabulary build, encode, decode round-trip
  - Vocabulary min_freq filtering
  - Vocabulary OOV tokens map to UNK
"""

import pytest
import torch

from src.training.loss import LabelSmoothingLoss
from src.training import NoamScheduler
from src.data import Vocabulary, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX


class TestLabelSmoothingLoss:

    def test_output_is_scalar(self):
        loss_fn = LabelSmoothingLoss(vocab_size=100, pad_idx=0, smoothing=0.1)
        logits = torch.randn(16, 100)
        targets = torch.randint(1, 100, (16,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_loss_non_negative(self):
        loss_fn = LabelSmoothingLoss(vocab_size=50, pad_idx=0, smoothing=0.1)
        logits = torch.randn(10, 50)
        targets = torch.randint(1, 50, (10,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_pad_positions_excluded(self):
        """A batch of only pad tokens should give zero loss."""
        loss_fn = LabelSmoothingLoss(vocab_size=50, pad_idx=0, smoothing=0.1)
        logits = torch.randn(8, 50)
        targets = torch.zeros(8, dtype=torch.long)  # all PAD
        loss = loss_fn(logits, targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows(self):
        loss_fn = LabelSmoothingLoss(vocab_size=80, smoothing=0.1)
        logits = torch.randn(12, 80, requires_grad=True)
        targets = torch.randint(1, 80, (12,))
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_lower_smoothing_lower_loss_on_correct(self):
        """With epsilon=0, loss on the correct token class should be <= smoothed loss."""
        logits = torch.zeros(1, 10)
        logits[0, 3] = 10.0  # high confidence on token 3
        targets = torch.tensor([3])
        loss_hard = LabelSmoothingLoss(10, smoothing=0.0)(logits, targets)
        loss_smooth = LabelSmoothingLoss(10, smoothing=0.1)(logits, targets)
        assert loss_hard.item() <= loss_smooth.item()


class TestNoamScheduler:

    @pytest.fixture
    def scheduler(self):
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        return NoamScheduler(opt, d_model=512, warmup_steps=4000)

    def test_lr_increases_during_warmup(self, scheduler):
        lrs = []
        for _ in range(100):
            lrs.append(scheduler.step())
        # LR should be monotonically increasing in early steps
        assert all(lrs[i] < lrs[i + 1] for i in range(50))

    def test_lr_decreases_after_warmup(self, scheduler):
        for _ in range(4001):
            scheduler.step()
        lr_at_peak = scheduler.last_lr
        for _ in range(1000):
            scheduler.step()
        lr_later = scheduler.last_lr
        assert lr_later < lr_at_peak

    def test_peak_at_warmup_steps(self, scheduler):
        warmup = 4000
        lrs = [scheduler.step() for _ in range(warmup + 200)]
        peak_step = lrs.index(max(lrs))
        # Peak should be at or near warmup_steps
        assert abs(peak_step - warmup) <= 5


class TestVocabulary:

    @pytest.fixture
    def vocab(self):
        v = Vocabulary()
        v.build([["hello", "world", "foo"], ["hello", "foo", "bar"]], min_freq=2)
        return v

    def test_special_tokens_present(self, vocab):
        assert PAD_IDX == vocab.token_to_id["<pad>"]
        assert UNK_IDX == vocab.token_to_id["<unk>"]
        assert BOS_IDX == vocab.token_to_id["<bos>"]
        assert EOS_IDX == vocab.token_to_id["<eos>"]

    def test_min_freq_filtering(self, vocab):
        # "bar" appears once -> should be excluded (min_freq=2)
        assert "bar" not in vocab
        # "hello" and "foo" appear twice -> should be included
        assert "hello" in vocab
        assert "foo" in vocab

    def test_encode_adds_bos_eos(self, vocab):
        ids = vocab.encode(["hello", "world"])
        assert ids[0] == BOS_IDX
        assert ids[-1] == EOS_IDX

    def test_oov_maps_to_unk(self, vocab):
        ids = vocab.encode(["unknown_word_xyz"], add_bos=False, add_eos=False)
        assert ids[0] == UNK_IDX

    def test_round_trip(self, vocab):
        tokens = ["hello", "foo"]
        ids = vocab.encode(tokens, add_bos=False, add_eos=False)
        decoded = vocab.decode(ids)
        assert decoded == "hello foo"

    def test_decode_strips_specials(self, vocab):
        ids = [BOS_IDX] + vocab.encode(["hello"], add_bos=False, add_eos=False) + [EOS_IDX]
        decoded = vocab.decode(ids, strip_special=True)
        assert "<bos>" not in decoded
        assert "<eos>" not in decoded
