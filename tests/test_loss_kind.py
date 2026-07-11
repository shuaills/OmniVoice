"""loss_kind contract tests (contract v2 §2).

Covers: IGNORE<->-100 invariant, exactly one cb0 EOS per document,
padding kind is IGNORE after collate, RNG-free derivation, and
decouple vs legacy branch kinds.
"""
import random

import torch

from omnivoice.blockdiff_dual import (
    KIND_ACOUSTIC,
    KIND_EOS,
    KIND_IGNORE,
    KIND_VOID,
    SILENCE_FRAME_TOKENS,
    build_loss_kind,
    block_eos_id,
)


def _mk_noisy_labels(C=8, T=70, bs=32, decouple=True, void_window=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_blocks = T // bs + 1
    canvas = n_blocks * bs
    labels = torch.full((C, canvas), -100, dtype=torch.long)
    content_mask = torch.rand((C, T), generator=g) < 0.5
    vals = torch.randint(0, 1024, (C, T), generator=g)
    labels[:, :T][content_mask] = vals[content_mask]
    eos = block_eos_id(1024)
    v_hi = min(T + 1 + void_window, canvas) if decouple else T + 1
    if decouple:
        labels[0, T] = eos
        if v_hi > T + 1:
            labels[:, T + 1:v_hi] = SILENCE_FRAME_TOKENS[:C].unsqueeze(1)
    else:
        labels[0, T:min(T + 4, canvas)] = eos
    return labels, T, canvas, v_hi


def test_invariant_ignore_iff_unlabeled():
    for decouple in (True, False):
        labels, T, canvas, v_hi = _mk_noisy_labels(decouple=decouple)
        kind = build_loss_kind(labels, T, canvas, decouple, v_hi)
        assert torch.equal(kind == KIND_IGNORE, labels == -100)


def test_exactly_one_eos_on_cb0_decouple():
    labels, T, canvas, v_hi = _mk_noisy_labels(decouple=True)
    kind = build_loss_kind(labels, T, canvas, True, v_hi)
    assert (kind[0] == KIND_EOS).sum().item() == 1
    assert (kind[1:] == KIND_EOS).sum().item() == 0
    assert kind[0, T].item() == KIND_EOS


def test_kind_regions():
    labels, T, canvas, v_hi = _mk_noisy_labels(decouple=True)
    kind = build_loss_kind(labels, T, canvas, True, v_hi)
    # acoustic only inside content region
    assert (kind[:, :T][kind[:, :T] != KIND_IGNORE] == KIND_ACOUSTIC).all()
    # void spans all codebooks in (T, v_hi)
    assert (kind[:, T + 1:v_hi] == KIND_VOID).all()
    # nothing supervised beyond v_hi
    assert (kind[:, v_hi:] == KIND_IGNORE).all()


def test_rng_free():
    labels, T, canvas, v_hi = _mk_noisy_labels()
    random.seed(1234)
    torch_state = torch.get_rng_state().clone()
    py_probe_before = None
    build_loss_kind(labels, T, canvas, True, v_hi)
    assert torch.equal(torch.get_rng_state(), torch_state)
    state_after = random.getstate()
    random.seed(1234)
    assert random.getstate() == state_after  # python RNG untouched


def test_collate_padding_is_ignore():
    from omnivoice.data.collator import PackingDataCollator

    samples = []
    for seed in (0, 1):
        labels, T, canvas, v_hi = _mk_noisy_labels(seed=seed)
        kind = build_loss_kind(labels, T, canvas, True, v_hi)
        C, L = labels.shape
        samples.append(
            {
                "input_ids": torch.zeros((C, L), dtype=torch.long),
                "labels": labels,
                "audio_mask": torch.ones(L, dtype=torch.bool),
                "length": L,
                "position_ids": torch.arange(L),
                "copy_tag": torch.zeros(L, dtype=torch.int32),
                "block_idx": torch.zeros(L, dtype=torch.int32),
                "loss_kind": kind,
            }
        )
    total = sum(s["length"] for s in samples)
    import types
    stub = types.SimpleNamespace(text_tokenizer=types.SimpleNamespace(pad_token_id=0))
    collator = PackingDataCollator(processor=stub, batch_tokens=total + 17)
    out = collator(samples)
    lk = out["loss_kind"][0]
    assert lk.shape[1] == total + 17
    assert (lk[:, total:] == KIND_IGNORE).all()
    lab = out["labels"][0]
    assert torch.equal(lk == KIND_IGNORE, lab == -100)


def test_collate_mixed_presence_raises():
    import types

    from omnivoice.data.collator import PackingDataCollator

    labels, T, canvas, v_hi = _mk_noisy_labels(seed=0)
    kind = build_loss_kind(labels, T, canvas, True, v_hi)
    C, L = labels.shape
    base = {
        "input_ids": torch.zeros((C, L), dtype=torch.long),
        "labels": labels,
        "audio_mask": torch.ones(L, dtype=torch.bool),
        "length": L,
        "position_ids": torch.arange(L),
        "copy_tag": torch.zeros(L, dtype=torch.int32),
        "block_idx": torch.zeros(L, dtype=torch.int32),
    }
    with_kind = dict(base, loss_kind=kind)
    stub = types.SimpleNamespace(text_tokenizer=types.SimpleNamespace(pad_token_id=0))
    collator = PackingDataCollator(processor=stub, batch_tokens=2 * L + 5)
    try:
        collator([with_kind, dict(base)])
    except AssertionError as e:
        assert "mixed loss_kind" in str(e)
    else:
        raise AssertionError("mixed presence did not raise")
