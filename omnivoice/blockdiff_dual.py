#!/usr/bin/env python3
# Copyright    2026  (block-diffusion B2: true block-causal attention,
#                     design/block-b2-causal-20260706)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""B2: block-causal attention via the BD3-LM two-copy layout.

Training sequence per sample::

    [ text prefix P | clean acoustic copy | noisy acoustic copy ]

Both acoustic copies share RoPE positions (clean col j and noisy col j are
both at position P + j).  Attention rule (within a packed document):

  - prefix query        -> prefix keys only
  - clean block b query -> prefix + clean blocks <= b
  - noisy block b query -> prefix + clean blocks <  b + noisy block b

Loss is computed on the noisy copy only (masked content cells + EOS fill).
Per-block mask ratio ~ U(0, 1), drawn independently per block.

The point of this geometry: a committed block's representation depends only
on the prefix and earlier committed blocks, never on anything to its right,
so block-level KV caching at inference is *exact* (gate 3 asserts this
against full recompute).  The quality cost relative to B1's bidirectional
right-truncation is what the B1-vs-B2 comparison measures.

The [eos] machinery (vocab 1026, cb0-only semantics, mandatory CFG bypass)
is inherited unchanged from omnivoice.blockdiff.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from omnivoice.blockdiff import block_eos_id
from omnivoice.data.processor import OmniVoiceSampleProcessor

TAG_PREFIX, TAG_CLEAN, TAG_NOISY, TAG_PAD = 0, 1, 2, -1


# ---------------------------------------------------------------------------
# Attention rule (single source of truth, tensor-friendly)
# ---------------------------------------------------------------------------


def _rule(qt, qb, kt, kb):
    """Block-causal visibility.  All args broadcastable int tensors."""
    prefix_kv = kt == TAG_PREFIX
    q_prefix = qt == TAG_PREFIX
    q_clean = qt == TAG_CLEAN
    clean_ok = prefix_kv | ((kt == TAG_CLEAN) & (kb <= qb))
    noisy_ok = (
        prefix_kv
        | ((kt == TAG_CLEAN) & (kb < qb))
        | ((kt == TAG_NOISY) & (kb == qb))
    )
    return torch.where(q_prefix, prefix_kv, torch.where(q_clean, clean_ok, noisy_ok))


def _mask_mod_block_causal(document_ids, copy_tags, block_ids, b, h, q_idx, kv_idx):
    same_doc = document_ids[q_idx] == document_ids[kv_idx]
    ok = _rule(
        copy_tags[q_idx], block_ids[q_idx], copy_tags[kv_idx], block_ids[kv_idx]
    )
    # Padding rows (doc id -1) keep the plain same-doc behaviour so no
    # query row ends up with an empty visible set (softmax NaN guard).
    return same_doc & (ok | (document_ids[q_idx] < 0))


def get_block_causal_mask_mod(document_ids, copy_tags, block_ids):
    from functools import partial

    return partial(_mask_mod_block_causal, document_ids, copy_tags, block_ids)


def build_block_causal_attn_mask(
    document_ids: torch.Tensor,
    copy_tags: torch.Tensor,
    block_ids: torch.Tensor,
) -> torch.Tensor:
    """Dense [1, 1, L, L] bool mask implementing the same rule (sdpa/tests)."""
    d, t, blk = document_ids, copy_tags, block_ids
    same_doc = d.unsqueeze(1) == d.unsqueeze(0)
    ok = _rule(
        t.unsqueeze(1), blk.unsqueeze(1), t.unsqueeze(0), blk.unsqueeze(0)
    )
    pad_q = (d < 0).unsqueeze(1)
    return (same_doc & (ok | pad_q)).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Training-side processor (two-copy layout)
# ---------------------------------------------------------------------------


class OmniVoiceBlockDualSampleProcessor(OmniVoiceSampleProcessor):
    """Official conditioning draws + two-copy block-causal sample layout."""

    def __init__(self, *args, block_size: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # --- official draw order (verbatim; global mask_ratio draw is kept
        # for stream stability but superseded by per-block ratios) ---
        if "clean_start_token_idx" in sample["label"]:
            drop_cond = False
        else:
            drop_cond = random.uniform(0, 1) < self.drop_cond_ratio

        if drop_cond:
            prompt_ratio = 0.0
            drop_text = True
            use_language = False
            use_instruct = False
        else:
            prompt_ratio = random.uniform(*self.prompt_ratio_range)
            drop_text = False
            use_language = random.uniform(0, 1) < self.language_ratio
            use_instruct = random.uniform(0, 1) < self.instruct_ratio
            if use_instruct and random.uniform(0, 1) < self.only_instruct_ratio:
                prompt_ratio = 0.0

        _ = random.uniform(*self.mask_ratio_range)  # superseded (see above)

        style = ""
        if use_language:
            language = sample["label"].get("language_id", "None")
        else:
            language = "None"
        if use_instruct:
            instruct = sample["label"].get("instruct", "None")
        else:
            instruct = "None"
        if "clean_start_token_idx" in sample["label"]:
            style += "<|denoise|>"
        style += f"<|lang_start|>{language}<|lang_end|>"
        style += f"<|instruct_start|>{instruct}<|instruct_end|>"

        style_inputs = self.text_tokenizer(style, return_tensors="pt").input_ids.repeat(
            self.num_channels, 1
        )
        if (
            "text_pinyin" in sample["label"]
            and random.uniform(0, 1) < self.use_pinyin_ratio
        ):
            text = sample["label"]["text_pinyin"]
        else:
            text = sample["label"]["text"]
        text_inputs = self.text_tokenizer(
            f"<|text_start|>{text}<|text_end|>", return_tensors="pt"
        ).input_ids.repeat(self.num_channels, 1)

        audio_tokens = sample["audio_tokens"].long()
        C, T = audio_tokens.shape

        if "clean_start_token_idx" in sample["label"]:
            prompt_length = sample["label"]["clean_start_token_idx"]
        else:
            prompt_length = int(T * prompt_ratio)

        bs = self.block_size
        n_blocks = T // bs + 1          # trailing block always reaches EOS fill
        clean_len = (n_blocks - 1) * bs  # complete content blocks only
        canvas_len = n_blocks * bs
        eos = block_eos_id(self.audio_mask_id)

        clean_copy = audio_tokens[:, :clean_len].clone()

        noisy_copy = torch.full((C, canvas_len), self.audio_mask_id, dtype=torch.long)
        noisy_copy[:, :T] = audio_tokens
        noisy_labels = torch.full((C, canvas_len), -100, dtype=torch.long)
        for b in range(n_blocks):
            lo, hi = b * bs, min((b + 1) * bs, T)
            m_lo = max(lo, prompt_length)
            if m_lo >= hi:
                continue
            ratio = random.uniform(0.0, 1.0)
            seg = audio_tokens[:, m_lo:hi]
            token_mask = torch.rand(seg.shape) < ratio
            region = noisy_copy[:, m_lo:hi]
            region[token_mask] = self.audio_mask_id
            lab = seg.clone()
            lab[~token_mask] = -100
            noisy_labels[:, m_lo:hi] = lab
        # EOS fill: cols >= T are mask on input, [eos] label on cb0 only.
        noisy_labels[0, T:] = eos

        if drop_text:
            P = 0
            input_ids = torch.cat([clean_copy, noisy_copy], dim=1)
            labels = torch.cat(
                [torch.full((C, clean_len), -100, dtype=torch.long), noisy_labels],
                dim=1,
            )
        else:
            prefix_inputs = torch.cat([style_inputs, text_inputs], dim=1)
            P = prefix_inputs.shape[1]
            input_ids = torch.cat([prefix_inputs, clean_copy, noisy_copy], dim=1)
            labels = torch.cat(
                [
                    torch.full((C, P), -100, dtype=torch.long),
                    torch.full((C, clean_len), -100, dtype=torch.long),
                    noisy_labels,
                ],
                dim=1,
            )

        total_length = input_ids.shape[1]
        audio_mask = torch.zeros(total_length, dtype=torch.bool)
        audio_mask[P:] = True

        position_ids = torch.cat(
            [
                torch.arange(P, dtype=torch.long),
                P + torch.arange(clean_len, dtype=torch.long),
                P + torch.arange(canvas_len, dtype=torch.long),
            ]
        )
        copy_tag = torch.cat(
            [
                torch.full((P,), TAG_PREFIX, dtype=torch.int32),
                torch.full((clean_len,), TAG_CLEAN, dtype=torch.int32),
                torch.full((canvas_len,), TAG_NOISY, dtype=torch.int32),
            ]
        )
        block_idx = torch.cat(
            [
                torch.full((P,), -1, dtype=torch.int32),
                torch.arange(clean_len, dtype=torch.int32) // bs,
                torch.arange(canvas_len, dtype=torch.int32) // bs,
            ]
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "audio_mask": audio_mask,
            "length": total_length,
            "position_ids": position_ids,
            "copy_tag": copy_tag,
            "block_idx": block_idx,
        }


# ---------------------------------------------------------------------------
# Inference: block-by-block decode, cache and recompute paths
# ---------------------------------------------------------------------------



def _crop_cache(cache, length: int) -> None:
    """DynamicCache.crop with a manual fallback for older transformers."""
    if hasattr(cache, "crop"):
        cache.crop(length)
        return
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][..., :length, :]
        cache.value_cache[i] = cache.value_cache[i][..., :length, :]


def _forward_slices(model, ids, positions, attn4d, past_key_values=None):
    """Run backbone + audio heads on explicit tensors.

    ids [C, L] (audio cells; mask id where unknown), positions [L],
    attn4d [1, 1, L, K] bool.  Returns logits [1, C, L, V].
    """
    amask = torch.ones((1, ids.size(1)), dtype=torch.bool, device=ids.device)
    embeds = model._prepare_embed_inputs(ids.unsqueeze(0), amask)
    out = model.llm(
        inputs_embeds=embeds,
        attention_mask=attn4d,
        position_ids=positions.unsqueeze(0),
        past_key_values=past_key_values,
        use_cache=past_key_values is not None,
        return_dict=True,
    )
    h = out[0]
    B, L, _ = h.shape
    logits = (
        model.audio_heads(h)
        .view(B, L, model.config.num_audio_codebook, model.config.audio_vocab_size)
        .permute(0, 2, 1, 3)
    )
    return logits


def _forward_text_prefix(model, text_ids, positions, attn4d, past_key_values):
    amask = torch.zeros((1, text_ids.size(1)), dtype=torch.bool, device=text_ids.device)
    embeds = model._prepare_embed_inputs(text_ids.unsqueeze(0), amask)
    model.llm(
        inputs_embeds=embeds,
        attention_mask=attn4d,
        position_ids=positions.unsqueeze(0),
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )


@torch.no_grad()
def _decode_block_causal(
    model,
    prefix_text_ids: Optional[torch.Tensor],
    gen_config,
    block_size: int = 32,
    max_blocks: int = 32,
    num_step_per_block: int = 8,
    use_kv_cache: bool = True,
    logit_trace: Optional[list] = None,
    seed_audio: Optional[torch.Tensor] = None,
):
    """Block-causal decode.  Returns (generated [C, G], stats).

    cache path: prefix and committed blocks live in a DynamicCache; per
    unmasking step only the current block is forwarded (queries see all
    cached keys + the block itself); intra-block steps are cropped off the
    cache, the final committed block is appended once.
    recompute path: every step rebuilds [prefix | committed | cur] with the
    dense rule mask -- geometrically identical, no cache. Gate 3 asserts the
    two produce identical tokens.
    """
    from omnivoice.blockdiff import _predict_tokens_blockwise
    from omnivoice.models.omnivoice import _get_time_steps, _gumbel_sample

    device = model.device
    C = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id
    eos = block_eos_id(mask_id)
    assert model.config.audio_vocab_size >= eos + 1, (
        "checkpoint has no [eos] row -- run scripts/migrate_block_ckpt.py first"
    )
    layer_ids = torch.arange(C, device=device).view(1, -1, 1)
    bs = block_size

    prefix = (
        prefix_text_ids.to(device).long()
        if prefix_text_ids is not None
        else torch.empty((C, 0), dtype=torch.long, device=device)
    )
    P = prefix.size(1)
    use_cfg = gen_config.guidance_scale != 0 and P > 0

    committed = torch.empty((C, 0), dtype=torch.long, device=device)
    seed_blocks, seed_rem, seed_total = 0, None, 0
    if seed_audio is not None and seed_audio.size(1) > 0:
        seed_total = seed_audio.size(1)
        S = (seed_total // bs) * bs
        committed = seed_audio[:, :S].to(device).long()
        seed_blocks = S // bs
        if seed_total > S:
            seed_rem = seed_audio[:, S:].to(device).long()

    caches = None
    if use_kv_cache:
        from transformers import DynamicCache

        caches = {"c": DynamicCache()}
        if P > 0:
            pre_attn = torch.ones((1, 1, P, P), dtype=torch.bool, device=device)
            pre_pos = torch.arange(P, device=device)
            _forward_text_prefix(model, prefix, pre_pos, pre_attn, caches["c"])
        if use_cfg:
            caches["u"] = DynamicCache()
        for sb in range(seed_blocks):
            blk_t = committed[:, sb * bs:(sb + 1) * bs]
            Kc = caches["c"].get_seq_length()
            attn = torch.ones((1, 1, bs, Kc + bs), dtype=torch.bool, device=device)
            _forward_slices(model, blk_t, P + sb * bs + torch.arange(bs, device=device), attn, caches["c"])
            if use_cfg:
                Ku = caches["u"].get_seq_length()
                attn_u = torch.ones((1, 1, bs, Ku + bs), dtype=torch.bool, device=device)
                _forward_slices(model, blk_t, sb * bs + torch.arange(bs, device=device), attn_u, caches["u"])

    stats = {"n_blocks": 0, "stopped_by_eos": False, "eos_col": None}

    for b in range(seed_blocks, max_blocks):
        cur = torch.full((C, bs), mask_id, dtype=torch.long, device=device)
        if b == seed_blocks and seed_rem is not None:
            cur[:, : seed_rem.size(1)] = seed_rem
        cur_pos = P + b * bs + torch.arange(bs, device=device)

        timesteps = _get_time_steps(
            t_start=0.0, t_end=1.0, num_step=num_step_per_block,
            t_shift=gen_config.t_shift,
        ).tolist()
        n_pre = seed_rem.size(1) if (b == seed_blocks and seed_rem is not None) else 0
        total_mask = (bs - n_pre) * C
        rem, sched = total_mask, []
        for step in range(num_step_per_block):
            num = (
                rem
                if step == num_step_per_block - 1
                else min(
                    math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                    rem,
                )
            )
            sched.append(int(num))
            rem -= int(num)

        for step in range(num_step_per_block):
            k = sched[step]
            if k <= 0:
                continue

            if use_kv_cache:
                Kc = caches["c"].get_seq_length()
                attn = torch.ones((1, 1, bs, Kc + bs), dtype=torch.bool, device=device)
                c_logits = _forward_slices(
                    model, cur, cur_pos, attn, caches["c"]
                )
                _crop_cache(caches["c"], Kc)
                if use_cfg:
                    Ku = caches["u"].get_seq_length()
                    attn_u = torch.ones(
                        (1, 1, bs, Ku + bs), dtype=torch.bool, device=device
                    )
                    u_pos = b * bs + torch.arange(bs, device=device)
                    u_logits = _forward_slices(
                        model, cur, u_pos, attn_u, caches["u"]
                    )
                    _crop_cache(caches["u"], Ku)
                else:
                    u_logits = c_logits
            else:
                seq = torch.cat([prefix, committed, cur], dim=1)
                L = seq.size(1)
                d = torch.zeros(L, dtype=torch.int32, device=device)
                t = torch.cat(
                    [
                        torch.full((P,), TAG_PREFIX, dtype=torch.int32, device=device),
                        torch.full(
                            (committed.size(1),), TAG_CLEAN, dtype=torch.int32,
                            device=device,
                        ),
                        torch.full((bs,), TAG_NOISY, dtype=torch.int32, device=device),
                    ]
                )
                blk = torch.cat(
                    [
                        torch.full((P,), -1, dtype=torch.int32, device=device),
                        torch.arange(committed.size(1), dtype=torch.int32, device=device)
                        // bs,
                        torch.full((bs,), b, dtype=torch.int32, device=device),
                    ]
                )
                attn = build_block_causal_attn_mask(d, t, blk).to(device)
                pos = torch.cat(
                    [
                        torch.arange(P, device=device),
                        P + torch.arange(committed.size(1), device=device),
                        cur_pos,
                    ]
                )
                mixed = seq.clone()
                logits_full = _forward_slices_mixed(model, mixed, P, pos, attn)
                c_logits = logits_full[:, :, L - bs :, :]
                if use_cfg:
                    sequ = torch.cat([committed, cur], dim=1)
                    Lu = sequ.size(1)
                    du = torch.zeros(Lu, dtype=torch.int32, device=device)
                    tu = t[P:]
                    blku = blk[P:]
                    attnu = build_block_causal_attn_mask(du, tu, blku).to(device)
                    posu = torch.cat(
                        [
                            torch.arange(committed.size(1), device=device),
                            b * bs + torch.arange(bs, device=device),
                        ]
                    )
                    logits_u = _forward_slices(model, sequ, posu, attnu)
                    u_logits = logits_u[:, :, Lu - bs :, :]
                else:
                    u_logits = c_logits

            pred_tokens, scores = _predict_tokens_blockwise(
                model, c_logits.to(torch.float32), u_logits.to(torch.float32),
                gen_config,
            )
            if logit_trace is not None:
                logit_trace.append(c_logits.to(torch.float32).cpu())
            scores = scores - (layer_ids * gen_config.layer_penalty_factor)
            if gen_config.position_temperature > 0.0:
                scores = _gumbel_sample(scores, gen_config.position_temperature)
            cur_view = cur.unsqueeze(0)
            scores = scores.masked_fill(cur_view != mask_id, -float("inf"))
            _, topk_idx = torch.topk(scores.flatten(), k)
            flat = cur_view.flatten().clone()
            flat[topk_idx] = pred_tokens.flatten()[topk_idx]
            cur = flat.view(C, bs)

        if use_kv_cache:
            # Commit: append the final block to the caches (clean geometry --
            # queries see cache + block itself, exactly the training rule).
            Kc = caches["c"].get_seq_length()
            attn = torch.ones((1, 1, bs, Kc + bs), dtype=torch.bool, device=device)
            _ = _forward_slices(model, cur, cur_pos, attn, caches["c"])
            if use_cfg:
                Ku = caches["u"].get_seq_length()
                attn_u = torch.ones(
                    (1, 1, bs, Ku + bs), dtype=torch.bool, device=device
                )
                u_pos = b * bs + torch.arange(bs, device=device)
                _ = _forward_slices(model, cur, u_pos, attn_u, caches["u"])

        committed = torch.cat([committed, cur], dim=1)
        stats["n_blocks"] = b + 1

        eos_hits = (cur[0] == eos).nonzero(as_tuple=True)[0]
        if eos_hits.numel() > 0:
            stop_abs = committed.size(1) - bs + int(eos_hits[0])
            stats["stopped_by_eos"] = True
            stats["eos_col"] = stop_abs
            committed = committed[:, :stop_abs]
            break

    return committed[:, seed_total:], stats


def _forward_slices_mixed(model, ids, P, positions, attn4d):
    """Forward for a mixed [text prefix | audio] row (recompute path)."""
    amask = torch.zeros((1, ids.size(1)), dtype=torch.bool, device=ids.device)
    amask[0, P:] = True
    embeds = model._prepare_embed_inputs(ids.unsqueeze(0), amask)
    out = model.llm(
        inputs_embeds=embeds,
        attention_mask=attn4d,
        position_ids=positions.unsqueeze(0),
        return_dict=True,
    )
    h = out[0]
    B, L, _ = h.shape
    return (
        model.audio_heads(h)
        .view(B, L, model.config.num_audio_codebook, model.config.audio_vocab_size)
        .permute(0, 2, 1, 3)
    )


@torch.no_grad()
def generate_blockwise_causal(
    model,
    text: str,
    language: Optional[str] = None,
    gen_config=None,
    block_size: int = 32,
    max_blocks: int = 32,
    num_step_per_block: int = 8,
    use_kv_cache: bool = True,
    instruct: Optional[str] = None,
):
    """User-facing wrapper: text -> audio tokens under block-causal geometry."""
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    if gen_config is None:
        gen_config = OmniVoiceGenerationConfig()

    inp = model._prepare_inference_inputs(
        text, block_size, None, None, language, instruct, False
    )
    input_ids = inp["input_ids"]
    if input_ids.dim() == 3:
        input_ids = input_ids[0]
    amask = inp["audio_mask"]
    if amask.dim() == 2:
        amask = amask[0]
    audio_cols = int(amask.sum())
    a0 = input_ids.size(1) - audio_cols
    prefix_text = input_ids[:, :a0]

    return _decode_block_causal(
        model,
        prefix_text,
        gen_config,
        block_size=block_size,
        max_blocks=max_blocks,
        num_step_per_block=num_step_per_block,
        use_kv_cache=use_kv_cache,
    )
