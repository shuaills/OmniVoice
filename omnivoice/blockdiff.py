#!/usr/bin/env python3
# Copyright    2026  (block-diffusion conversion, design/block-conversion-20260706)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Block-diffusion conversion of OmniVoice (v0, single-copy scheme).

Direction: block-causal generation with an [eos] stop token, minimal
machinery, quality judged by results.

Scheme ("current block" supervision, single copy):

- Vocabulary grows 1025 -> 1026; [eos] = audio_mask_id + 1 carries semantics
  on codebook 0 only. The official mask id (1024) is unchanged.
- Training draws ONE current block ``c`` per sample and truncates the canvas
  at its right edge. Blocks left of ``c`` stay clean (committed context, no
  loss); block ``c`` is per-cell masked with ratio ~ U(0,1). If block ``c``
  covers the content tail, the block is padded to the block boundary with
  EOS-fill cells (input = mask, cb0 label = [eos], other codebooks
  unsupervised), so the full-width block the model sees at inference always
  has a training analog, and "keep going vs stop" is learned as an ordinary
  cb0 class decision.
- Attention stays the OFFICIAL bidirectional mask over the (truncated)
  sequence. This scheme adds NO attention code at all: v0 inference
  recomputes the full canvas every step (no KV cache), so training and
  inference geometry match exactly. Block-level KV caching is a later,
  separate optimization (see note in ``generate_blockwise``).
- Why not the BD3-LM two-copy layout (clean stream + noisy stream): its
  streams must be attention-isolated from the prefix to be cacheable, so no
  configuration of it reproduces the official bidirectional geometry -- the
  degenerate-equivalence anchor gate (single block == official, exact) is
  unsatisfiable by construction. The single-copy scheme passes that gate
  byte-exactly, at the cost of supervising only one block per sample
  (roughly 2-3x fewer supervised cells per token; partially recovered
  because truncated samples pack denser under batch_tokens batching).

Inference CFG: ordinary classes use the official two-branch guidance.
Historical ``legacy`` scoring keeps [eos] on the conditional branch alone
(E1 scar: rare-class logit noise was amplified by ``c + s(c-u)``).  That
mixed-score behavior remains the compatibility default; routing EOS through
CFG or applying another calibration is explicit and opt-in at decode time.
"""

import math
import random
from typing import Any, Dict, Optional

import torch

from omnivoice.data.processor import OmniVoiceSampleProcessor

EOS_OFFSET = 1

EOS_CFG_CALIBRATION_LEGACY = "legacy"
EOS_CFG_CALIBRATION_RENORM = "renorm"
EOS_CFG_CALIBRATION_GUIDED = "guided"
EOS_CFG_CALIBRATION_MASS_PRESERVING = "mass_preserving"
_EOS_CFG_CALIBRATION_MODES = (
    EOS_CFG_CALIBRATION_LEGACY,
    EOS_CFG_CALIBRATION_RENORM,
    EOS_CFG_CALIBRATION_GUIDED,
    EOS_CFG_CALIBRATION_MASS_PRESERVING,
)


def block_eos_id(audio_mask_id: int) -> int:
    """[eos] sits immediately after the mask id (1024 -> 1025)."""
    return audio_mask_id + EOS_OFFSET


# ---------------------------------------------------------------------------
# Training-side: current-block sample processor
# ---------------------------------------------------------------------------


class OmniVoiceBlockSampleProcessor(OmniVoiceSampleProcessor):
    """Official processing + current-block truncation + EOS fill.

    RNG discipline: draws happen in the official order (drop_cond,
    prompt_ratio, language, instruct, mask_ratio, torch.rand over the
    maskable cells). The current-block index is drawn only when more than
    one block is eligible, so the degenerate configuration
    (block_size >= canvas, eos_enabled=False) consumes exactly the official
    RNG stream and produces byte-identical tensors -- that is the anchor
    for the degenerate-equivalence gate.
    """

    def __init__(
        self,
        *args,
        block_size: int = 32,
        eos_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size
        self.eos_enabled = eos_enabled

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # --- official draw order (kept verbatim for RNG alignment) ---
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

        mask_ratio = random.uniform(*self.mask_ratio_range)

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
        style_labels = torch.full(style_inputs.shape, -100)

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
        text_labels = torch.full(text_inputs.shape, -100)

        audio_tokens = sample["audio_tokens"].long()
        C, T = audio_tokens.shape

        if "clean_start_token_idx" in sample["label"]:
            prompt_length = sample["label"]["clean_start_token_idx"]
        else:
            prompt_length = int(T * prompt_ratio)

        # --- block bookkeeping ---
        bs = self.block_size
        if self.eos_enabled:
            # T // bs + 1 blocks: when T is block-aligned this adds a pure
            # EOS-fill block, so "the content is over, say stop" is a
            # reachable training state; otherwise the last (partial) content
            # block carries the fill tail.
            n_blocks = T // bs + 1
        else:
            n_blocks = max(1, math.ceil(T / bs))

        # A block is eligible as "current" iff it contains at least one
        # supervisable cell (maskable content at col >= prompt_length, or
        # fill). Blocks fully inside the prompt are excluded.
        first_eligible = min(prompt_length // bs, n_blocks - 1)
        eligible = list(range(first_eligible, n_blocks))
        if len(eligible) > 1:
            c = eligible[random.randrange(len(eligible))]
        else:
            c = eligible[0]

        if self.eos_enabled:
            canvas_len = (c + 1) * bs
        else:
            canvas_len = min((c + 1) * bs, T)
        content_end = min(canvas_len, T)
        fill_len = canvas_len - content_end

        audio_inputs = audio_tokens[:, :content_end].clone()
        audio_labels = torch.full((C, canvas_len), -100, dtype=torch.long)

        # Current-block content cells: per-cell Bernoulli(mask_ratio), loss
        # on masked cells only (official semantics restricted to block c).
        cur_lo = max(c * bs, prompt_length)
        if cur_lo < content_end:
            maskable = audio_tokens[:, cur_lo:content_end]
            token_mask = torch.rand(maskable.shape) < mask_ratio
            region = audio_inputs[:, cur_lo:content_end]
            region[token_mask] = self.audio_mask_id
            lab = maskable.clone()
            lab[~token_mask] = -100
            audio_labels[:, cur_lo:content_end] = lab
        # Committed blocks (cols < c*bs) stay clean with no loss. Official
        # parity note: official only supervises the prompt region when
        # drop_cond, and drop_cond forces prompt_length = 0, so "-100 on the
        # committed/prompt region" is not a deviation.

        if fill_len > 0:
            fill_in = torch.full(
                (C, fill_len), self.audio_mask_id, dtype=torch.long
            )
            audio_inputs = torch.cat([audio_inputs, fill_in], dim=1)
            audio_labels[0, content_end:] = block_eos_id(self.audio_mask_id)

        # --- official concatenation ---
        if drop_text:
            input_ids = audio_inputs
            labels = audio_labels
            total_length = input_ids.shape[1]
            audio_mask = torch.ones(total_length, dtype=torch.bool)
        else:
            input_ids = torch.cat([style_inputs, text_inputs, audio_inputs], dim=1)
            labels = torch.cat([style_labels, text_labels, audio_labels], dim=1)
            total_length = input_ids.shape[1]
            audio_start_idx = style_inputs.shape[1] + text_inputs.shape[1]
            audio_mask = torch.zeros(total_length, dtype=torch.bool)
            audio_mask[audio_start_idx:] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "audio_mask": audio_mask,
            "length": total_length,
        }


# ---------------------------------------------------------------------------
# Inference-side: block-by-block decoding with model-decided stopping
# ---------------------------------------------------------------------------


def _calibrate_blockwise_eos_log_probs(
    cfg_log_probs: torch.Tensor,
    conditional_log_probs: torch.Tensor,
    mask_id: int,
    eos_id: int,
    mode: str = EOS_CFG_CALIBRATION_LEGACY,
) -> torch.Tensor:
    """Splice conditional EOS scores into blockwise CFG scores.

    ``legacy`` preserves the historical mixed-score behavior exactly: the
    conditional EOS log-probability replaces the CFG EOS entry without a
    subsequent normalization.  The experimental modes make the score
    scale explicit:

    * ``renorm`` performs the legacy splice, then normalizes the EOS-capable
      codebook-0 row over its legal classes.
    * ``guided`` treats EOS exactly like every other codebook-0 class under
      CFG, then normalizes over the legal codebook-0 classes.
    * ``mass_preserving`` keeps conditional ``p(EOS)`` exactly and distributes
      the remaining probability mass over codebook 0's legal non-EOS classes
      in their CFG relative proportions.

    EOS is legal only on codebook 0.  The mask class and classes after EOS are
    structurally invalid on every codebook.  Experimental calibration is
    deliberately limited to codebook 0; all other codebooks retain their
    historical scores exactly so the A/B isolates the EOS-capable row.
    """
    if mode not in _EOS_CFG_CALIBRATION_MODES:
        choices = ", ".join(_EOS_CFG_CALIBRATION_MODES)
        raise ValueError(
            f"unknown eos_cfg_calibration={mode!r}; expected one of: {choices}"
        )

    log_probs = cfg_log_probs.clone()
    log_probs[..., mask_id] = -float("inf")
    if log_probs.size(-1) > eos_id + 1:
        log_probs[..., eos_id + 1 :] = -float("inf")

    conditional_eos = conditional_log_probs[:, 0:1, :, eos_id]

    # Historical surgery: EOS bypasses CFG on cb0 and is forbidden on all
    # other codebooks.  In legacy mode the resulting mixed scores are
    # deliberately left unnormalized for byte-compatible default output.
    log_probs[:, 1:, :, eos_id] = -float("inf")
    log_probs[:, 0:1, :, eos_id] = conditional_eos
    if mode == EOS_CFG_CALIBRATION_LEGACY:
        return log_probs
    if mode == EOS_CFG_CALIBRATION_RENORM:
        log_probs[:, 0:1] = torch.log_softmax(
            log_probs[:, 0:1], dim=-1
        )
        return log_probs
    if mode == EOS_CFG_CALIBRATION_GUIDED:
        guided_cb0 = cfg_log_probs[:, 0:1].clone()
        guided_cb0[..., mask_id] = -float("inf")
        if guided_cb0.size(-1) > eos_id + 1:
            guided_cb0[..., eos_id + 1 :] = -float("inf")
        log_probs[:, 0:1] = torch.log_softmax(guided_cb0, dim=-1)
        return log_probs

    # Normalize codebook 0's legal non-EOS CFG classes to recover their
    # relative distribution after structural classes have been removed.
    cb0_non_eos = cfg_log_probs[:, 0:1].clone()
    cb0_non_eos[..., mask_id] = -float("inf")
    cb0_non_eos[..., eos_id:] = -float("inf")
    cb0_non_eos = torch.log_softmax(cb0_non_eos, dim=-1)

    # log(1 - p_eos), evaluated stably from log(p_eos).  conditional_eos is a
    # log-softmax result, so it is <= 0; p_eos == 1 correctly yields -inf.
    non_eos_mass = torch.log(-torch.expm1(conditional_eos))
    log_probs[:, 0:1] = cb0_non_eos + non_eos_mass.unsqueeze(-1)
    log_probs[:, 0:1, :, eos_id] = conditional_eos
    return log_probs


def _predict_tokens_blockwise(model, c_logits, u_logits, gen_config):
    """Official scoring with [eos] surgery.

    Ordinary classes use official CFG (log-softmax extrapolation).  Historical
    ``legacy`` scoring bypasses CFG for [eos]; explicit experimental modes can
    normalize that splice or route codebook-0 EOS through CFG as well.  EOS is
    always banned off cb0.  The absent/default calibration remains ``legacy``.
    """
    from omnivoice.models.omnivoice import _filter_top_k, _gumbel_sample

    mask_id = model.config.audio_mask_id
    eos = block_eos_id(mask_id)

    c_log_probs = torch.nn.functional.log_softmax(c_logits, dim=-1)
    if gen_config.guidance_scale != 0:
        u_log_probs = torch.nn.functional.log_softmax(u_logits, dim=-1)
        log_probs = torch.log_softmax(
            c_log_probs
            + gen_config.guidance_scale * (c_log_probs - u_log_probs),
            dim=-1,
        )
    else:
        log_probs = c_log_probs

    calibration = getattr(
        gen_config,
        "eos_cfg_calibration",
        EOS_CFG_CALIBRATION_LEGACY,
    )
    log_probs = _calibrate_blockwise_eos_log_probs(
        log_probs,
        c_log_probs,
        mask_id,
        eos,
        mode=calibration,
    )

    if gen_config.class_temperature > 0.0:
        filtered = _filter_top_k(log_probs, ratio=0.1)
        pred_tokens = _gumbel_sample(filtered, gen_config.class_temperature).argmax(
            dim=-1
        )
    else:
        pred_tokens = log_probs.argmax(dim=-1)
    confidence = log_probs.max(dim=-1)[0]
    return pred_tokens, confidence


@torch.no_grad()
def _blockwise_decode(
    model,
    prefix_text_ids: Optional[torch.Tensor],
    init_canvas: torch.Tensor,
    gen_config,
    block_size: int = 32,
    max_blocks: int = 32,
    num_step_per_block: int = 8,
    first_chunk: Optional[int] = None,
):
    """Core loop. ``prefix_text_ids`` [C, P] (None when unconditional-only),
    ``init_canvas`` [C, R] committed audio (voice-clone prompt or empty).

    Returns (generated [C, G], stats dict). Full recompute per step; when
    block KV caching lands, the cache key is the immutable committed prefix
    [prefix | canvas] and committed-block KV is frozen at commit time (the
    only approximation this scheme will introduce).
    """
    from omnivoice.models.omnivoice import _get_time_steps

    device = model.device
    C = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id
    eos = block_eos_id(mask_id)
    assert model.config.audio_vocab_size >= eos + 1, (
        "checkpoint has no [eos] row -- run scripts/migrate_block_ckpt.py first"
    )
    layer_ids = torch.arange(C, device=device).view(1, -1, 1)

    canvas = init_canvas.to(device).long()
    R = canvas.size(1)
    prefix = (
        prefix_text_ids.to(device).long()
        if prefix_text_ids is not None
        else torch.empty((C, 0), dtype=torch.long, device=device)
    )
    P = prefix.size(1)
    use_cfg = gen_config.guidance_scale != 0 and P > 0

    stats = {"n_blocks": 0, "stopped_by_eos": False, "eos_col": None}

    for chunk_idx in range(max_blocks):
        chunk_len = (
            first_chunk if (chunk_idx == 0 and first_chunk) else block_size
        )
        cur = torch.full(
            (C, chunk_len), mask_id, dtype=torch.long, device=device
        )

        # per-chunk unmasking schedule (official shape, over this chunk only)
        timesteps = _get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=num_step_per_block,
            t_shift=gen_config.t_shift,
        ).tolist()
        total_mask = chunk_len * C
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
            cond_seq = torch.cat([prefix, canvas, cur], dim=1)
            Lc = cond_seq.size(1)
            rows = [cond_seq]
            lens = [Lc]
            if use_cfg:
                # Unconditional analog of training drop_cond: audio only,
                # committed context included.
                u_seq = torch.cat([canvas, cur], dim=1)
                rows.append(u_seq)
                lens.append(u_seq.size(1))
            Lmax = max(lens)
            B = len(rows)
            batch_ids = torch.full(
                (B, C, Lmax), mask_id, dtype=torch.long, device=device
            )
            batch_amask = torch.zeros((B, Lmax), dtype=torch.bool, device=device)
            batch_attn = torch.zeros(
                (B, 1, Lmax, Lmax), dtype=torch.bool, device=device
            )
            for i, (row, ln) in enumerate(zip(rows, lens)):
                batch_ids[i, :, :ln] = row
                batch_attn[i, :, :ln, :ln] = True
                if i == 0:
                    batch_amask[i, P:ln] = True
                else:
                    batch_amask[i, :ln] = True
                if Lmax > ln:
                    diag = torch.arange(ln, Lmax, device=device)
                    batch_attn[i, :, diag, diag] = True

            logits = model(
                input_ids=batch_ids,
                audio_mask=batch_amask,
                attention_mask=batch_attn,
            ).logits.to(torch.float32)

            c_logits = logits[0:1, :, lens[0] - chunk_len : lens[0], :]
            u_logits = (
                logits[1:2, :, lens[1] - chunk_len : lens[1], :]
                if use_cfg
                else c_logits
            )
            pred_tokens, scores = _predict_tokens_blockwise(
                model, c_logits, u_logits, gen_config
            )
            scores = scores - (layer_ids * gen_config.layer_penalty_factor)
            if gen_config.position_temperature > 0.0:
                from omnivoice.models.omnivoice import _gumbel_sample

                scores = _gumbel_sample(scores, gen_config.position_temperature)
            cur_view = cur.unsqueeze(0)
            scores = scores.masked_fill(cur_view != mask_id, -float("inf"))
            _, topk_idx = torch.topk(scores.flatten(), k)
            flat = cur_view.flatten().clone()
            flat[topk_idx] = pred_tokens.flatten()[topk_idx]
            cur = flat.view(C, chunk_len)

        canvas = torch.cat([canvas, cur], dim=1)
        stats["n_blocks"] = chunk_idx + 1

        eos_hits = (cur[0] == eos).nonzero(as_tuple=True)[0]
        if eos_hits.numel() > 0:
            stop_abs = canvas.size(1) - chunk_len + int(eos_hits[0])
            stats["stopped_by_eos"] = True
            stats["eos_col"] = stop_abs
            canvas = canvas[:, :stop_abs]
            break

    return canvas[:, R:], stats


@torch.no_grad()
def generate_blockwise(
    model,
    text: str,
    language: Optional[str] = None,
    gen_config=None,
    block_size: int = 32,
    max_blocks: int = 32,
    num_step_per_block: int = 8,
    ref_text: Optional[str] = None,
    ref_audio_tokens: Optional[torch.Tensor] = None,
    instruct: Optional[str] = None,
):
    """Block-by-block generation; total length decided by the model ([eos]).

    No Eq.4 rule-based duration anywhere in this path -- that is the point
    of the conversion. ``max_blocks`` is a safety net, not a target length.
    Returns (audio_tokens [C, T], stats).
    """
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    if gen_config is None:
        gen_config = OmniVoiceGenerationConfig()

    R = ref_audio_tokens.size(1) if ref_audio_tokens is not None else 0
    first_chunk = block_size - (R % block_size) if R % block_size else block_size

    inp = model._prepare_inference_inputs(
        text,
        first_chunk,
        ref_text,
        ref_audio_tokens,
        language,
        instruct,
        False,
    )
    input_ids = inp["input_ids"]
    if input_ids.dim() == 3:  # _prepare_inference_inputs returns [1, C, L]
        input_ids = input_ids[0]
    amask = inp["audio_mask"]
    if amask.dim() == 2:
        amask = amask[0]
    audio_cols = int(amask.sum())
    a0 = input_ids.size(1) - audio_cols
    prefix_text = input_ids[:, :a0]
    init_canvas = input_ids[:, a0 : a0 + R]

    return _blockwise_decode(
        model,
        prefix_text,
        init_canvas,
        gen_config,
        block_size=block_size,
        max_blocks=max_blocks,
        num_step_per_block=num_step_per_block,
        first_chunk=first_chunk,
    )
