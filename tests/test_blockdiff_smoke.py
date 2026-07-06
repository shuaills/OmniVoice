#!/usr/bin/env python3
"""CPU smoke tests for the block-diffusion conversion (no GPU, no checkpoint).

Gates covered (numbers land in block-conversion/RESULTS.md):
  gate 1 -- processor invariants (supervision confined to the current block,
            EOS fill shape, prompt guard, block alignment)
  gate 2 -- degenerate anchor: block_size >= canvas + eos disabled consumes
            the official RNG stream and yields byte-identical tensors
  gate 4 -- generate_blockwise mechanics: legal ids, EOS truncation,
            max_blocks safety net

Run:  python tests/test_blockdiff_smoke.py
"""

import random

import torch

import omnivoice.blockdiff as BD
from omnivoice.blockdiff import (
    OmniVoiceBlockSampleProcessor,
    _blockwise_decode,
    block_eos_id,
)
from omnivoice.data.processor import OmniVoiceSampleProcessor

C, MASK = 8, 1024
EOS = block_eos_id(MASK)  # 1025


class FakeTok:
    pad_token_id = 0

    def __call__(self, text, return_tensors=None):
        class R:
            input_ids = torch.randint(1, 50, (1, max(1, len(text) // 4)))

        return R()


def _proc(cls, **kw):
    base = dict(
        text_tokenizer=FakeTok(),
        num_channels=C,
        audio_mask_id=MASK,
        prompt_ratio_range=(0.0, 0.3),
        mask_ratio_range=(0.0, 1.0),
        drop_cond_ratio=0.1,
        language_ratio=0.5,
        use_pinyin_ratio=0.0,
        instruct_ratio=0.0,
        only_instruct_ratio=0.0,
    )
    base.update(kw)
    return cls(**base)


def _sample(T, **label_extra):
    label = {"text": "hello block diffusion world", "language_id": "en"}
    label.update(label_extra)
    return {"audio_tokens": torch.randint(0, 1024, (C, T)), "label": label}


def test_gate1_processor_invariants():
    bs = 32
    proc = _proc(OmniVoiceBlockSampleProcessor, block_size=bs)
    for trial in range(300):
        random.seed(1000 + trial)
        torch.manual_seed(1000 + trial)
        T = random.choice([5, 31, 32, 33, 64, 70, 129])
        out = proc(_sample(T))
        ids, labs, amask = out["input_ids"], out["labels"], out["audio_mask"]
        a0 = int((~amask).sum())  # text prefix length (0 when drop_cond)
        a_ids, a_labs = ids[:, a0:], labs[:, a0:]
        La = a_ids.size(1)
        # canvas block-aligned (eos fill guarantees it)
        assert La % bs == 0, (T, La)
        # supervision confined to the current (= last) block
        sup = (a_labs != -100).any(dim=0)
        assert not sup[: La - bs].any(), "loss outside the current block"
        # masked inputs confined to the current block
        m = (a_ids == MASK).any(dim=0)
        assert not m[: La - bs].any(), "mask outside the current block"
        # no eos in inputs; labels: eos only on cb0
        assert (a_ids != EOS).all()
        assert (a_labs[1:] != EOS).all()
        # fill cells: input mask on all codebooks, cb0 label eos, others -100
        fill = a_labs[0] == EOS
        if fill.any():
            assert (a_ids[:, fill] == MASK).all()
            assert (a_labs[1:, fill] == -100).all()
        # content cells in the current block: masked -> truth label,
        # unmasked -> -100
        cur = torch.zeros(La, dtype=torch.bool)
        cur[La - bs :] = True
        content_cur = cur & ~fill
        cc_ids, cc_labs = a_ids[:, content_cur], a_labs[:, content_cur]
        assert (cc_labs[cc_ids != MASK] == -100).all()
        masked_cells = cc_ids == MASK
        lab_vals = cc_labs[masked_cells]
        assert ((lab_vals >= 0) & (lab_vals < 1024)).all()
    print("test_gate1_processor_invariants OK")


def test_gate1_prompt_guard():
    bs = 32
    proc = _proc(OmniVoiceBlockSampleProcessor, block_size=bs)
    for trial in range(100):
        random.seed(5000 + trial)
        torch.manual_seed(5000 + trial)
        out = proc(_sample(70, clean_start_token_idx=40))
        ids, labs, amask = out["input_ids"], out["labels"], out["audio_mask"]
        a0 = int((~amask).sum())
        a_labs = labs[:, a0:]
        sup_cols = (a_labs != -100).any(dim=0).nonzero(as_tuple=True)[0]
        assert (sup_cols >= 40).all(), "supervision inside the prompt"
    print("test_gate1_prompt_guard OK")


def test_gate2_degenerate_byte_identity():
    """block_size >= canvas + eos off == official, byte for byte."""
    official = _proc(OmniVoiceSampleProcessor)
    block = _proc(
        OmniVoiceBlockSampleProcessor, block_size=4096, eos_enabled=False
    )
    for trial in range(200):
        T = 17 + (trial % 60)
        random.seed(9000 + trial)
        torch.manual_seed(9000 + trial)
        s = _sample(T)
        random.seed(31 + trial)
        torch.manual_seed(31 + trial)
        a = official(s)
        random.seed(31 + trial)
        torch.manual_seed(31 + trial)
        b = block(s)
        assert a["length"] == b["length"]
        assert torch.equal(a["input_ids"], b["input_ids"])
        assert torch.equal(a["labels"], b["labels"])
        assert torch.equal(a["audio_mask"], b["audio_mask"])
    print("test_gate2_degenerate_byte_identity OK (200 seeds)")


def _tiny_model(vocab=EOS + 1):
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig

    llm_cfg = dict(
        model_type="qwen3",
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=2048,  # must exceed audio ids: text embed sees them pre-where
        head_dim=16,
        max_position_embeddings=2048,
    )
    cfg = OmniVoiceConfig(
        audio_vocab_size=vocab,
        audio_mask_id=MASK,
        num_audio_codebook=C,
        llm_config=llm_cfg,
    )
    torch.manual_seed(20260706)
    return OmniVoice(cfg).eval()


def _gen_config():
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    gc = OmniVoiceGenerationConfig()
    gc.guidance_scale = 2.0
    gc.class_temperature = 0.0
    gc.position_temperature = 0.0
    return gc


def test_gate4_blockwise_mechanics():
    model = _tiny_model()
    gc = _gen_config()
    prefix = torch.randint(1, 100, (C, 5))
    empty = torch.empty((C, 0), dtype=torch.long)
    real_predict = BD._predict_tokens_blockwise

    # (a) eos banned -> runs to the max_blocks safety net, ids all legal
    def ban_eos(model_, c, u, g):
        pred, conf = real_predict(model_, c, u, g)
        bad = pred == EOS
        pred[bad] = 7
        return pred, conf

    BD._predict_tokens_blockwise = ban_eos
    try:
        out, stats = _blockwise_decode(
            model, prefix, empty, gc, block_size=8, max_blocks=3,
            num_step_per_block=4,
        )
    finally:
        BD._predict_tokens_blockwise = real_predict
    assert out.shape == (C, 24), out.shape
    assert stats["n_blocks"] == 3 and not stats["stopped_by_eos"]
    assert (out != MASK).all() and (out != EOS).all()
    assert (out >= 0).all() and (out < EOS + 1).all()

    # (b) eos forced at chunk col 3 -> truncated to 3 cols, stops after 1 block
    def force_eos(model_, c, u, g):
        pred, conf = real_predict(model_, c, u, g)
        pred[0, 0, 3] = EOS
        conf[0, 0, 3] = 1e9
        return pred, conf

    BD._predict_tokens_blockwise = force_eos
    try:
        out, stats = _blockwise_decode(
            model, prefix, empty, gc, block_size=8, max_blocks=3,
            num_step_per_block=4,
        )
    finally:
        BD._predict_tokens_blockwise = real_predict
    assert stats["stopped_by_eos"] and stats["n_blocks"] == 1
    assert out.shape == (C, 3), out.shape
    assert (out != EOS).all()

    # (c) un-migrated checkpoint (vocab 1025) is refused
    small = _tiny_model(vocab=MASK + 1)
    try:
        _blockwise_decode(small, prefix, empty, gc, block_size=8, max_blocks=1)
        raise RuntimeError("should have refused a checkpoint without [eos]")
    except AssertionError:
        pass
    print("test_gate4_blockwise_mechanics OK")


def test_migration_adds_single_class():
    from omnivoice.elastic import migrate_state_dict

    V, H, NB = 17, 8, 1  # toy vocab
    sd = {
        "audio_embeddings.weight": torch.randn(C * V, H),
        "audio_heads.weight": torch.randn(C * V, H),
        "codebook_layer_offsets": torch.arange(C) * V,
    }
    new = migrate_state_dict(
        {k: v.clone() for k, v in sd.items()}, C, V, num_new_classes=NB
    )
    NV = V + NB
    assert new["audio_embeddings.weight"].shape[0] == C * NV
    assert torch.equal(new["codebook_layer_offsets"], torch.arange(C) * NV)
    h = torch.randn(3, H)
    old_logits = h @ sd["audio_heads.weight"].T
    new_logits = h @ new["audio_heads.weight"].T
    for c in range(C):
        assert torch.allclose(
            old_logits[:, c * V : (c + 1) * V],
            new_logits[:, c * NV : c * NV + V],
        )
    print("test_migration_adds_single_class OK")


def test_collator_integration():
    from omnivoice.data.collator import PaddingDataCollator

    proc = _proc(OmniVoiceBlockSampleProcessor, block_size=16)
    random.seed(3)
    torch.manual_seed(3)
    samples = [proc(_sample(20 + 11 * i)) for i in range(4)]
    batch = PaddingDataCollator(proc, 4096)(samples)
    assert batch["input_ids"].dim() == 3
    assert batch["labels"].shape == batch["input_ids"].shape
    print("test_collator_integration OK")


if __name__ == "__main__":
    test_gate1_processor_invariants()
    test_gate1_prompt_guard()
    test_gate2_degenerate_byte_identity()
    test_gate4_blockwise_mechanics()
    test_migration_adds_single_class()
    test_collator_integration()
    print("ALL BLOCKDIFF CPU SMOKE TESTS PASSED")
