#!/usr/bin/env python3
"""CPU gates for B2 (block-causal dual-copy). No GPU, no checkpoint.

  gate 1  -- attention rule: brute-force spec reference vs dense builder vs
             flex mask_mod, exact equality on packed multi-doc + pad layouts
  gate 1b -- dual processor invariants (copies, labels, positions, tags)
  gate 2  -- train-inference visibility consistency: the mask rows the
             decoder builds for the current block == the training rule rows
             restricted to inference-present columns; training-only columns
             (future clean, other noisy) contribute nothing
  gate 3c -- KV-cache equivalence on a tiny fp64 model: cache path and
             recompute path produce identical tokens, max |dlogit| reported

Run:  python tests/test_block_dual_cpu.py
"""

import random

import torch

from omnivoice.blockdiff import block_eos_id
from omnivoice.blockdiff_dual import (
    TAG_CLEAN,
    TAG_NOISY,
    TAG_PREFIX,
    OmniVoiceBlockDualSampleProcessor,
    _decode_block_causal,
    _mask_mod_block_causal,
    build_block_causal_attn_mask,
)

C, MASK = 8, 1024
EOS = block_eos_id(MASK)


class FakeTok:
    pad_token_id = 0

    def __call__(self, text, return_tensors=None):
        class R:
            input_ids = torch.randint(1, 50, (1, max(1, len(text) // 4)))

        return R()


def _proc(**kw):
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
    return OmniVoiceBlockDualSampleProcessor(**base)


def _sample(T):
    return {
        "audio_tokens": torch.randint(0, 1024, (C, T)),
        "label": {"text": "hello block causal world", "language_id": "en"},
    }


def _spec_visible(qd, qt, qb, kd, kt, kb):
    """Independent literal reimplementation of the three-clause spec."""
    if qd < 0:  # padding row: keep same-doc behaviour
        return kd == qd
    if kd != qd:
        return False
    if qt == TAG_PREFIX:
        return kt == TAG_PREFIX
    if qt == TAG_CLEAN:
        return kt == TAG_PREFIX or (kt == TAG_CLEAN and kb <= qb)
    return (
        kt == TAG_PREFIX
        or (kt == TAG_CLEAN and kb < qb)
        or (kt == TAG_NOISY and kb == qb)
    )


def _layout(docs):
    """docs: list of (P, n_blocks, bs). Returns packed d/t/blk tensors."""
    d, t, blk = [], [], []
    for i, (P, nb, bs) in enumerate(docs):
        cl, cv = (nb - 1) * bs, nb * bs
        d += [i] * (P + cl + cv)
        t += [TAG_PREFIX] * P + [TAG_CLEAN] * cl + [TAG_NOISY] * cv
        blk += [-1] * P + [j // bs for j in range(cl)] + [j // bs for j in range(cv)]
    return (
        torch.tensor(d, dtype=torch.int32),
        torch.tensor(t, dtype=torch.int32),
        torch.tensor(blk, dtype=torch.int32),
    )


def test_gate1_mask_rule():
    docs = [(5, 3, 4), (0, 2, 4), (3, 1, 4)]  # incl. drop_cond doc (P=0)
    d, t, blk = _layout(docs)
    pad = 6
    d = torch.cat([d, torch.full((pad,), -1, dtype=torch.int32)])
    t = torch.cat([t, torch.full((pad,), -1, dtype=torch.int32)])
    blk = torch.cat([blk, torch.full((pad,), -1, dtype=torch.int32)])
    L = d.numel()

    ref = torch.zeros(L, L, dtype=torch.bool)
    for q in range(L):
        for kv in range(L):
            ref[q, kv] = _spec_visible(
                int(d[q]), int(t[q]), int(blk[q]),
                int(d[kv]), int(t[kv]), int(blk[kv]),
            )

    dense = build_block_causal_attn_mask(d, t, blk)[0, 0]
    assert torch.equal(dense, ref), "dense builder != spec"

    qi = torch.arange(L).view(-1, 1).expand(L, L)
    ki = torch.arange(L).view(1, -1).expand(L, L)
    flexed = _mask_mod_block_causal(d, t, blk, 0, 0, qi, ki)
    assert torch.equal(flexed, ref), "flex mask_mod != spec"

    # every real row has a nonempty visible set (softmax safety)
    assert dense[d >= 0].any(dim=-1).all(), "empty visible set for a real row"
    print(f"gate1 OK: {L}x{L} truth table, dense==mask_mod==spec, no empty rows")


def test_gate1b_processor_invariants():
    bs = 32
    proc = _proc(block_size=bs)
    checked = 0
    for trial in range(300):
        random.seed(2000 + trial)
        torch.manual_seed(2000 + trial)
        T = random.choice([5, 31, 32, 33, 64, 70, 129])
        src = _sample(T)
        truth = src["audio_tokens"].clone()
        out = proc(src)
        ids, labs = out["input_ids"], out["labels"]
        tag, blk, pos = out["copy_tag"], out["block_idx"], out["position_ids"]
        nb = T // bs + 1
        cl, cv = (nb - 1) * bs, nb * bs
        P = ids.size(1) - cl - cv
        assert P >= 0 and out["length"] == P + cl + cv
        assert (tag[:P] == TAG_PREFIX).all()
        assert (tag[P : P + cl] == TAG_CLEAN).all()
        assert (tag[P + cl :] == TAG_NOISY).all()
        # clean copy is the untouched truth
        assert torch.equal(ids[:, P : P + cl], truth[:, :cl])
        assert (labs[:, : P + cl] == -100).all(), "loss outside noisy copy"
        noisy_in = ids[:, P + cl :]
        noisy_lab = labs[:, P + cl :]
        # masked content cells: input==MASK, label==truth; unmasked: label==-100
        content = noisy_in[:, :T]
        content_lab = noisy_lab[:, :T]
        masked = content == MASK
        assert torch.equal(content_lab[masked], truth[:, :T][masked])
        assert (content_lab[~masked] == -100).all()
        assert torch.equal(content[~masked], truth[:, :T][~masked])
        # fill: input mask, cb0 label eos, others unsupervised
        assert (noisy_in[:, T:] == MASK).all()
        assert (noisy_lab[0, T:] == EOS).all()
        assert (noisy_lab[1:, T:] == -100).all()
        # shared RoPE positions between the copies
        assert torch.equal(pos[P : P + cl], P + torch.arange(cl))
        assert torch.equal(pos[P + cl :], P + torch.arange(cv))
        assert torch.equal(blk[P : P + cl], torch.arange(cl, dtype=torch.int32) // bs)
        assert torch.equal(blk[P + cl :], torch.arange(cv, dtype=torch.int32) // bs)
        checked += 1
    print(f"gate1b OK: {checked}/300 samples, all invariants hold")


def test_gate2_train_infer_consistency():
    bs, P, nb = 8, 5, 4
    d, t, blk = _layout([(P, nb, bs)])
    train_mask = build_block_causal_attn_mask(d, t, blk)[0, 0]
    cl = (nb - 1) * bs

    for b in range(nb):
        # inference layout when generating block b: [prefix | clean <b | cur]
        Li = P + b * bs + bs
        di = torch.zeros(Li, dtype=torch.int32)
        ti = torch.cat(
            [
                torch.full((P,), TAG_PREFIX, dtype=torch.int32),
                torch.full((b * bs,), TAG_CLEAN, dtype=torch.int32),
                torch.full((bs,), TAG_NOISY, dtype=torch.int32),
            ]
        )
        bi = torch.cat(
            [
                torch.full((P,), -1, dtype=torch.int32),
                torch.arange(b * bs, dtype=torch.int32) // bs,
                torch.full((bs,), b, dtype=torch.int32),
            ]
        )
        infer_mask = build_block_causal_attn_mask(di, ti, bi)[0, 0]
        infer_rows = infer_mask[-bs:]  # current-block queries

        # training rows for noisy block b, at inference-present columns:
        # prefix cols, clean cols < b*bs, noisy block b cols
        q0 = P + cl + b * bs
        train_rows = train_mask[q0 : q0 + bs]
        present = torch.cat(
            [
                torch.arange(P),
                P + torch.arange(b * bs),
                P + cl + b * bs + torch.arange(bs),
            ]
        )
        assert torch.equal(train_rows[:, present], infer_rows), f"block {b}: mismatch"
        # columns absent at inference must be invisible in training too
        absent = torch.ones(train_mask.size(1), dtype=torch.bool)
        absent[present] = False
        assert not train_rows[:, absent].any(), f"block {b}: sees absent cols"
        # cache path: visible set == all cached (P + b*bs keys) + itself
        assert infer_rows.all(), f"block {b}: cache all-visible assumption broken"
    print(f"gate2 OK: {nb} blocks, train rows == infer rows, absent cols dark, "
          "cache all-visible confirmed")


def _tiny_model():
    from transformers import Qwen3Config

    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig

    llm_cfg = Qwen3Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=2048,
        max_position_embeddings=2048,
    )
    cfg = OmniVoiceConfig(
        audio_vocab_size=MASK + 2, audio_mask_id=MASK, num_audio_codebook=C,
        llm_config=llm_cfg,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(20260706)
    model = OmniVoice(cfg).double().eval()
    return model


def test_gate3c_cache_equivalence_tiny():
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    model = _tiny_model()
    gen = OmniVoiceGenerationConfig()
    gen.class_temperature = 0.0
    gen.position_temperature = 0.0

    prefix = torch.randint(1, 200, (C, 6))
    results = {}
    for cfg_scale, tag in ((0.0, "cond"), (2.0, "cfg")):
        gen.guidance_scale = cfg_scale
        toks = {}
        traces = {}
        for use_cache in (True, False):
            trace = []
            out, stats = _decode_block_causal(
                model, prefix, gen, block_size=4, max_blocks=3,
                num_step_per_block=3, use_kv_cache=use_cache, logit_trace=trace,
            )
            toks[use_cache] = out
            traces[use_cache] = trace
        assert torch.equal(toks[True], toks[False]), f"{tag}: token mismatch"
        dmax = max(
            (a - b).abs().max().item()
            for a, b in zip(traces[True], traces[False])
        )
        results[tag] = (toks[True].shape, dmax)
    for tag, (shape, dmax) in results.items():
        print(f"gate3c OK [{tag}]: cache==recompute tokens {tuple(shape)}, "
              f"max |dlogit| = {dmax:.3e}")


if __name__ == "__main__":
    test_gate1_mask_rule()
    test_gate1b_processor_invariants()
    test_gate2_train_infer_consistency()
    test_gate3c_cache_equivalence_tiny()
    print("ALL B2 CPU GATES PASSED")
