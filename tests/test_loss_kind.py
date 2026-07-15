"""loss_kind and one-sided EOS-band contract tests.

Covers: IGNORE<->-100 invariant, cb0-only EOS bands, void displacement,
canvas-edge truncation, atomic terminal masking, k=1 bit identity, padding
IGNORE after collate, RNG-free derivation, and decouple vs legacy kinds.
"""
import random
import types

import torch

from omnivoice.blockdiff_dual import (
    KIND_ACOUSTIC,
    KIND_EOS,
    KIND_IGNORE,
    KIND_VOID,
    SILENCE_FRAME_TOKENS,
    OmniVoiceBlockDualSampleProcessor,
    build_loss_kind,
    block_eos_id,
)


def _mk_noisy_labels(
    C=8,
    T=70,
    bs=32,
    decouple=True,
    void_window=32,
    eos_band_k=1,
    seed=0,
):
    g = torch.Generator().manual_seed(seed)
    n_blocks = T // bs + 1
    canvas = n_blocks * bs
    labels = torch.full((C, canvas), -100, dtype=torch.long)
    content_mask = torch.rand((C, T), generator=g) < 0.5
    vals = torch.randint(0, 1024, (C, T), generator=g)
    labels[:, :T][content_mask] = vals[content_mask]
    eos = block_eos_id(1024)
    eos_hi = min(T + eos_band_k, canvas)
    v_hi = min(eos_hi + void_window, canvas) if decouple else T + 1
    if decouple:
        labels[0, T:eos_hi] = eos
        if v_hi > eos_hi:
            labels[:, eos_hi:v_hi] = SILENCE_FRAME_TOKENS[:C].unsqueeze(1)
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


def test_band_labeling_void_displacement_and_canvas_edge():
    C, T, k = 8, 70, 4
    labels, _, canvas, v_hi = _mk_noisy_labels(T=T, eos_band_k=k)
    kind = build_loss_kind(
        labels, T, canvas, True, v_hi, eos_band_k=k
    )
    eos = block_eos_id(1024)
    assert (labels[0, T:T + k] == eos).all()
    assert (labels[1:, T:T + k] == -100).all()
    assert (kind[0, T:T + k] == KIND_EOS).all()
    assert (kind[1:, T:T + k] == KIND_IGNORE).all()
    assert (kind[:, T + k:v_hi] == KIND_VOID).all()
    point_void_columns = canvas - (T + 1)
    band_void_columns = canvas - (T + k)
    assert (point_void_columns - band_void_columns) * C == 24

    # Only two terminal columns fit: the requested k=4 band is clipped to k_s=2.
    edge_T = 94
    edge_labels, _, edge_canvas, edge_v_hi = _mk_noisy_labels(
        T=edge_T, eos_band_k=k
    )
    edge_kind = build_loss_kind(
        edge_labels,
        edge_T,
        edge_canvas,
        True,
        edge_v_hi,
        eos_band_k=k,
    )
    assert edge_canvas == 96
    assert (edge_kind[0, edge_T:edge_canvas] == KIND_EOS).all()
    assert (edge_kind[1:, edge_T:edge_canvas] == KIND_IGNORE).all()
    assert not (edge_kind == KIND_VOID).any()


def test_rng_free():
    labels, T, canvas, v_hi = _mk_noisy_labels()
    random.seed(1234)
    torch_state = torch.get_rng_state().clone()
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


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, _text, return_tensors="pt"):
        return types.SimpleNamespace(
            input_ids=torch.tensor([[11, 12, 13]], dtype=torch.long)
        )


def _dual_processor(eos_band_k=None):
    kwargs = dict(
        text_tokenizer=_FakeTokenizer(),
        num_channels=8,
        audio_mask_id=1024,
        prompt_ratio_range=(0.3, 0.3),
        mask_ratio_range=(0.0, 1.0),
        drop_cond_ratio=0.0,
        language_ratio=0.0,
        use_pinyin_ratio=0.0,
        instruct_ratio=0.0,
        only_instruct_ratio=0.0,
        block_size=32,
        eos_decouple_silence=True,
        silence_void_window=32,
    )
    if eos_band_k is not None:
        kwargs["eos_band_k"] = eos_band_k
    return OmniVoiceBlockDualSampleProcessor(**kwargs)


def _fixed_sample(T=70):
    values = torch.arange(8 * T, dtype=torch.long).reshape(8, T) % 1024
    return {"audio_tokens": values, "label": {"text": "fixed fixture"}}


def _run_fixed(processor, seed, T=70):
    random.seed(seed)
    torch.manual_seed(seed)
    output = processor(_fixed_sample(T))
    return output, random.getstate(), torch.get_rng_state().clone()


def test_eos_band_mask_is_atomic_across_sampled_patterns():
    T, k = 70, 4
    processor = _dual_processor(k)
    canvas = (T // 32 + 1) * 32
    for seed in range(64):
        output, _, _ = _run_fixed(processor, seed, T)
        noisy_input = output["input_ids"][:, -canvas:]
        noisy_labels = output["labels"][:, -canvas:]
        noisy_kind = output["loss_kind"][:, -canvas:]
        band_mask_state = noisy_input[:, T:T + k] == 1024
        # Terminal supervision is deterministically hard-masked in the current
        # pipeline.  This is a stronger atomicity guarantee than a shared draw:
        # no sampled pattern may reveal one EOS sibling to another.
        assert band_mask_state.all()
        assert torch.equal(
            band_mask_state,
            band_mask_state[:, :1].expand_as(band_mask_state),
        )
        assert (noisy_labels[0, T:T + k] == block_eos_id(1024)).all()
        assert (noisy_labels[1:, T:T + k] == -100).all()
        assert (noisy_kind[0, T:T + k] == KIND_EOS).all()
        assert (noisy_kind[:, T + k:] == KIND_VOID).all()

    # Processor-level canvas-edge gate: k=4 clips to the two available cells.
    edge_T = 94
    edge_canvas = (edge_T // 32 + 1) * 32
    edge_output, _, _ = _run_fixed(processor, 7, edge_T)
    edge_labels = edge_output["labels"][:, -edge_canvas:]
    edge_kind = edge_output["loss_kind"][:, -edge_canvas:]
    assert (edge_labels[0, edge_T:] == block_eos_id(1024)).all()
    assert (edge_labels[1:, edge_T:] == -100).all()
    assert (edge_kind[0, edge_T:] == KIND_EOS).all()
    assert not (edge_kind == KIND_VOID).any()


def test_eos_band_k1_is_bitwise_identical_to_default_fixed_fixture():
    default, default_py, default_torch = _run_fixed(_dual_processor(), 20260716)
    explicit, explicit_py, explicit_torch = _run_fixed(
        _dual_processor(1), 20260716
    )
    assert default.keys() == explicit.keys()
    for key in default:
        left, right = default[key], explicit[key]
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), key
        else:
            assert left == right, key
    assert default_py == explicit_py
    assert torch.equal(default_torch, explicit_torch)
