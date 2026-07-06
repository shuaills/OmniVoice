#!/usr/bin/env python3
# Copyright    2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Elastic canvas for OmniVoice: model-decided length via [expand]/[delete].

Adapts DreamOn (arXiv:2602.01326) to the T x C multi-codebook acoustic matrix.
Two special classes are appended to every codebook's vocabulary slice (only
codebook 0's are ever used):

    EXPAND = audio_mask_id + 1   (default 1025)
    DELETE = audio_mask_id + 2   (default 1026)

During denoising, a committed EXPAND column splits into two fully-masked
columns; a committed DELETE column is removed. Structure ops are only allowed
while the global mask ratio exceeds ``elastic_theta`` (length then freezes and
the remaining cells are filled as usual).

This module contains ALL elastic logic. Hooks in official code are minimal
and dormant unless explicitly enabled (A/B-parallel implementation rule):

- ``OmniVoiceSampleProcessor`` subclass in ``omnivoice/data/processor.py``
- optional ``loss_weights`` passthrough in collators and ``OmniVoice.forward``
- a 3-line route at the top of ``OmniVoice._generate_iterative``
"""

import logging
import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Number of extra classes appended to each codebook's vocab slice.
NUM_ELASTIC_CLASSES = 2


def elastic_ids(audio_mask_id: int) -> Tuple[int, int]:
    """Return (expand_id, delete_id) given the mask id (mask=1024 -> 1025, 1026)."""
    return audio_mask_id + 1, audio_mask_id + 2


# ---------------------------------------------------------------------------
# Training-side: canvas corruption (runs inside the sample processor)
# ---------------------------------------------------------------------------


def corrupt_audio_region(
    audio_inputs: torch.Tensor,
    audio_labels: torch.Tensor,
    prompt_length: int,
    mask_id: int,
    merge_prob: float = 0.08,
    insert_prob: float = 0.04,
    end_append_max_ratio: float = 0.25,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply elastic canvas corruption to a masked audio region.

    Must be called AFTER the official per-cell Bernoulli masking, on the
    ``[C, T]`` audio inputs/labels. Three ops (all corrupted columns have
    fully-masked inputs; labels supervise only codebook 0 with a special id):

    1. merge: adjacent real columns (i, i+1) -> one column, cb0 label = EXPAND
    2. insert: a spurious mask column is inserted, cb0 label = DELETE
    3. end-append: 0..K DELETE columns appended after the last real column

    Returns (inputs [C, T'], labels [C, T'], loss_weights [C, T']). Weights are
    1.0 for ordinary supervised cells and EXPAND cells; DELETE cells share a
    total weight of 1.0 (1/N_delete each, DreamOn loss balancing).
    """
    rng = rng or random
    expand_id, delete_id = elastic_ids(mask_id)
    C, T = audio_inputs.shape

    def special_col(label_id: int):
        col_in = torch.full((C,), mask_id, dtype=audio_inputs.dtype)
        col_lab = torch.full((C,), -100, dtype=audio_labels.dtype)
        col_lab[0] = label_id
        return col_in, col_lab

    in_cols: List[torch.Tensor] = []
    lab_cols: List[torch.Tensor] = []

    # Prompt region is never corrupted.
    for j in range(prompt_length):
        in_cols.append(audio_inputs[:, j])
        lab_cols.append(audio_labels[:, j])

    j = prompt_length
    gen_len = T - prompt_length
    while j < T:
        if j + 1 < T and rng.random() < merge_prob:
            ci, cl = special_col(expand_id)
            in_cols.append(ci)
            lab_cols.append(cl)
            j += 2
            continue
        in_cols.append(audio_inputs[:, j])
        lab_cols.append(audio_labels[:, j])
        j += 1
        if rng.random() < insert_prob:
            ci, cl = special_col(delete_id)
            in_cols.append(ci)
            lab_cols.append(cl)

    k_end = rng.randint(0, max(0, int(end_append_max_ratio * gen_len)))
    for _ in range(k_end):
        ci, cl = special_col(delete_id)
        in_cols.append(ci)
        lab_cols.append(cl)

    new_inputs = torch.stack(in_cols, dim=1)
    new_labels = torch.stack(lab_cols, dim=1)

    weights = torch.ones_like(new_labels, dtype=torch.float32)
    delete_cells = new_labels == delete_id
    n_delete = int(delete_cells.sum())
    if n_delete > 0:
        weights[delete_cells] = 1.0 / n_delete
    return new_inputs, new_labels, weights


def corrupt_audio_region_targeted(
    audio_inputs: torch.Tensor,
    audio_labels: torch.Tensor,
    prompt_length: int,
    mask_id: int,
    delta_max: float = 0.3,
    mid_insert_frac: float = 0.3,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """E1.1 targeted corruption: per-sample canvas length error delta ~ U(-d, +d).

    The E1 (legacy) corruption composes independent per-position ops whose net
    |dT| almost never exceeds ~10% short — the expand head never saw a large
    deficit, so it never learned to fix one (RESULTS.md 2026-07-06 试金石 §3).
    Here the *sample-level* error is drawn first and ops are placed to realise
    it, aligning training |dT| with the inference rule-error target U(0.7,1.3):

    delta < 0 (canvas too short): round(|delta|*gen_len) non-overlapping
        adjacent pairs merged into [expand] columns.
    delta > 0 (canvas too long): round(delta*gen_len) spurious [delete]
        columns; ``mid_insert_frac`` of them scattered mid-sequence, the rest
        appended at the end (keeps DreamOn's end-delete regime represented).

    Loss weights follow ``corrupt_audio_region`` (delete cells share 1.0).
    """
    rng = rng or random
    expand_id, delete_id = elastic_ids(mask_id)
    C, T = audio_inputs.shape
    gen_len = T - prompt_length
    if gen_len < 4:
        weights = torch.ones_like(audio_labels, dtype=torch.float32)
        return audio_inputs, audio_labels, weights

    def special_col(label_id: int):
        col_in = torch.full((C,), mask_id, dtype=audio_inputs.dtype)
        col_lab = torch.full((C,), -100, dtype=audio_labels.dtype)
        col_lab[0] = label_id
        return col_in, col_lab

    delta = rng.uniform(-delta_max, delta_max)

    merge_starts: set = set()
    insert_after: dict = {}
    n_end = 0
    if delta < 0:
        n_merge = min(int(round(-delta * gen_len)), gen_len // 2)
        candidates = list(range(prompt_length, T - 1))
        rng.shuffle(candidates)
        for c in candidates:
            if len(merge_starts) >= n_merge:
                break
            if c in merge_starts or (c - 1) in merge_starts or (c + 1) in merge_starts:
                continue
            merge_starts.add(c)
    elif delta > 0:
        n_extra = int(round(delta * gen_len))
        n_mid = int(round(mid_insert_frac * n_extra))
        n_end = n_extra - n_mid
        for _ in range(n_mid):
            pos = rng.randint(prompt_length, T - 1)
            insert_after[pos] = insert_after.get(pos, 0) + 1

    in_cols: List[torch.Tensor] = []
    lab_cols: List[torch.Tensor] = []
    for j in range(prompt_length):
        in_cols.append(audio_inputs[:, j])
        lab_cols.append(audio_labels[:, j])

    j = prompt_length
    while j < T:
        if j in merge_starts and j + 1 < T:
            ci, cl = special_col(expand_id)
            in_cols.append(ci)
            lab_cols.append(cl)
            j += 2
            continue
        in_cols.append(audio_inputs[:, j])
        lab_cols.append(audio_labels[:, j])
        for _ in range(insert_after.get(j, 0)):
            ci, cl = special_col(delete_id)
            in_cols.append(ci)
            lab_cols.append(cl)
        j += 1

    for _ in range(n_end):
        ci, cl = special_col(delete_id)
        in_cols.append(ci)
        lab_cols.append(cl)

    new_inputs = torch.stack(in_cols, dim=1)
    new_labels = torch.stack(lab_cols, dim=1)
    weights = torch.ones_like(new_labels, dtype=torch.float32)
    delete_cells = new_labels == delete_id
    n_delete = int(delete_cells.sum())
    if n_delete > 0:
        weights[delete_cells] = 1.0 / n_delete
    return new_inputs, new_labels, weights


# ---------------------------------------------------------------------------
# Checkpoint migration: audio vocab V -> V + 2 (fused-table block remap)
# ---------------------------------------------------------------------------


def migrate_state_dict(
    state_dict: dict,
    num_codebook: int,
    old_vocab: int,
    num_new_classes: int = NUM_ELASTIC_CLASSES,
) -> dict:
    """Grow the fused audio embedding/head tables from old_vocab to
    old_vocab + num_new_classes per codebook, preserving the block layout
    ``row = c * vocab + v``. Elastic adds 2 classes ([expand]/[delete]);
    the block-diffusion conversion adds 1 ([eos]).

    New rows are initialised from the mean/std of each codebook's existing rows.
    """
    new_vocab = old_vocab + num_new_classes
    out = dict(state_dict)
    for key in ("audio_embeddings.weight", "audio_heads.weight"):
        if key not in state_dict:
            raise KeyError(f"{key} not found in state dict")
        w = state_dict[key]
        assert w.shape[0] == num_codebook * old_vocab, (
            f"{key}: expected {num_codebook * old_vocab} rows, got {w.shape[0]}"
        )
        new_w = w.new_empty((num_codebook * new_vocab,) + tuple(w.shape[1:]))
        for c in range(num_codebook):
            block = w[c * old_vocab : (c + 1) * old_vocab]
            new_w[c * new_vocab : c * new_vocab + old_vocab] = block
            init = block.float()
            new_rows = torch.randn(
                (num_new_classes,) + tuple(w.shape[1:]), dtype=torch.float32
            ) * init.std().item() + init.mean().item()
            new_w[c * new_vocab + old_vocab : (c + 1) * new_vocab] = new_rows.to(
                w.dtype
            )
        out[key] = new_w
    for key in ("audio_heads.bias",):
        if key in state_dict:
            b = state_dict[key]
            new_b = b.new_zeros(num_codebook * new_vocab)
            for c in range(num_codebook):
                new_b[c * new_vocab : c * new_vocab + old_vocab] = b[
                    c * old_vocab : (c + 1) * old_vocab
                ]
            out[key] = new_b
    # codebook_layer_offsets is a persisted buffer (arange(C) * vocab); if left
    # at the old vocab stride it silently corrupts every embedding lookup.
    if "codebook_layer_offsets" in state_dict:
        old_off = state_dict["codebook_layer_offsets"]
        out["codebook_layer_offsets"] = (
            torch.arange(num_codebook, dtype=old_off.dtype) * new_vocab
        )
    return out


# ---------------------------------------------------------------------------
# Inference-side: structure op execution + elastic iterative decoding
# ---------------------------------------------------------------------------


def execute_structure_ops(
    target_tokens: torch.Tensor, mask_id: int
) -> Tuple[torch.Tensor, int, int]:
    """Rewrite a ``[C, T]`` target canvas by executing committed specials on
    codebook 0: EXPAND -> two mask columns, DELETE -> column removed. The
    rightmost DELETE also broadcasts over a fully-masked suffix (DreamOn
    deletion broadcasting).

    Returns (new_tokens [C, T'], n_expand, n_delete).
    """
    expand_id, delete_id = elastic_ids(mask_id)
    C, T = target_tokens.shape
    row0 = target_tokens[0]

    n_expand = int((row0 == expand_id).sum())
    n_delete = int((row0 == delete_id).sum())
    if n_expand == 0 and n_delete == 0:
        return target_tokens, 0, 0

    # Deletion broadcasting: find rightmost DELETE whose suffix is all-mask.
    bcast_from = None
    delete_pos = (row0 == delete_id).nonzero(as_tuple=True)[0]
    if len(delete_pos) > 0:
        j = int(delete_pos[-1])
        suffix = target_tokens[:, j + 1 :]
        if suffix.numel() == 0 or bool((suffix == mask_id).all()):
            bcast_from = j

    cols: List[torch.Tensor] = []
    for j in range(T):
        if bcast_from is not None and j >= bcast_from:
            break
        v = int(row0[j])
        if v == delete_id:
            continue
        if v == expand_id:
            mask_col = torch.full(
                (C,), mask_id, dtype=target_tokens.dtype, device=target_tokens.device
            )
            cols.append(mask_col)
            cols.append(mask_col.clone())
            continue
        cols.append(target_tokens[:, j])

    if not cols:  # degenerate: everything deleted; keep one mask column
        cols = [
            torch.full(
                (C,), mask_id, dtype=target_tokens.dtype, device=target_tokens.device
            )
        ]
    return torch.stack(cols, dim=1), n_expand, n_delete


@torch.no_grad()
def generate_iterative_elastic(model, task, gen_config) -> List[torch.Tensor]:
    """Elastic-canvas counterpart of ``OmniVoice._generate_iterative``.

    Processes samples one at a time (cond+uncond as a 2-row batch) because the
    canvas length changes during decoding. Batched elastic decoding can be
    added later; correctness first.
    """
    if GUARDS is not None:
        return generate_iterative_elastic_guarded(model, task, gen_config)
    mask_id = model.config.audio_mask_id
    expand_id, delete_id = elastic_ids(mask_id)
    C = model.config.num_audio_codebook
    theta = getattr(gen_config, "elastic_theta", 0.4)
    lmax_ratio = getattr(gen_config, "elastic_lmax_ratio", 1.5)
    max_extra = getattr(gen_config, "elastic_max_extra_steps", 16)

    from omnivoice.models.omnivoice import _get_time_steps, _gumbel_sample

    timesteps = _get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=gen_config.num_step,
        t_shift=gen_config.t_shift,
    ).tolist()

    results: List[torch.Tensor] = []
    for i in range(task.batch_size):
        inputs = model._prepare_inference_inputs(
            task.texts[i],
            task.target_lens[i],
            task.ref_texts[i],
            task.ref_audio_tokens[i],
            task.langs[i],
            task.instructs[i],
            gen_config.denoise,
        )
        full_ids = inputs["input_ids"][0]  # [C, L] cond sequence
        t_len = task.target_lens[i]
        l_max = max(t_len + 1, int(t_len * lmax_ratio))
        prefix = full_ids[:, : full_ids.shape[1] - t_len]  # style+text+ref
        target = full_ids[:, full_ids.shape[1] - t_len :].clone()  # [C, t_len]

        layer_ids = torch.arange(C, device=model.device).view(1, -1, 1)

        step = 0
        n_ops_total = [0, 0]
        while True:
            n_mask = int((target == mask_id).sum())
            if n_mask == 0:
                break
            if step >= gen_config.num_step + max_extra:
                # Safety: force-commit everything left in one final pass.
                pass

            t_cur = target.shape[1]
            c_len = prefix.shape[1] + t_cur
            batch_ids = torch.full(
                (2, C, c_len), mask_id, dtype=torch.long, device=model.device
            )
            batch_ids[0] = torch.cat([prefix, target], dim=1)
            batch_ids[1, :, :t_cur] = target
            audio_mask = torch.zeros(2, c_len, dtype=torch.bool, device=model.device)
            audio_mask[0, prefix.shape[1] :] = True
            if task.ref_audio_tokens[i] is not None:
                ref_len = task.ref_audio_tokens[i].shape[-1]
                audio_mask[0, prefix.shape[1] - ref_len :] = True
            audio_mask[1, :t_cur] = True
            attn = torch.zeros(
                2, 1, c_len, c_len, dtype=torch.bool, device=model.device
            )
            attn[0] = True
            attn[1, :, :t_cur, :t_cur] = True
            if c_len > t_cur:
                diag = torch.arange(t_cur, c_len, device=model.device)
                attn[1, :, diag, diag] = True

            logits = model(
                input_ids=batch_ids, audio_mask=audio_mask, attention_mask=attn
            ).logits.to(torch.float32)
            c_logits = logits[0:1, :, prefix.shape[1] :, :]
            u_logits = logits[1:2, :, :t_cur, :]

            # Ban special classes where structure ops are not allowed:
            # rows 1..C-1 always; row 0 too once frozen (mask ratio <= theta),
            # EXPAND additionally once at L_max.
            mask_ratio = n_mask / float(t_cur * C)
            allow_ops = mask_ratio > theta and step < gen_config.num_step
            for lg in (c_logits, u_logits):
                lg[:, 1:, :, expand_id] = -float("inf")
                lg[:, 1:, :, delete_id] = -float("inf")
                if not allow_ops:
                    lg[:, 0, :, expand_id] = -float("inf")
                    lg[:, 0, :, delete_id] = -float("inf")
                elif t_cur >= l_max:
                    lg[:, 0, :, expand_id] = -float("inf")

            # E1.1 default: special classes bypass CFG extrapolation. The
            # uncond branch is badly calibrated for [expand]/[delete]; the
            # c + s*(c-u) blow-up was the single root cause of the op storm
            # (RESULTS.md 2026-07-06 试金石: bypass alone == all four guards).
            if getattr(gen_config, "elastic_cfg_bypass", True):
                pred_tokens, scores = _predict_guarded(
                    model, c_logits, u_logits, gen_config,
                    special_ids=(expand_id, delete_id), cfg_bypass=True,
                )
            else:
                pred_tokens, scores = model._predict_tokens_with_scoring(
                    c_logits, u_logits, gen_config
                )
            scores = scores - (layer_ids * gen_config.layer_penalty_factor)
            if gen_config.position_temperature > 0.0:
                scores = _gumbel_sample(scores, gen_config.position_temperature)

            sample_tokens = target.unsqueeze(0)
            scores = scores.masked_fill(sample_tokens != mask_id, -float("inf"))
            if allow_ops:
                # Gate rows 1..C-1 behind cb0 commitment so structure ops never
                # orphan committed detail cells.
                row0_masked = (sample_tokens[:, 0:1, :] == mask_id).expand(
                    -1, C - 1, -1
                )
                scores[:, 1:, :] = scores[:, 1:, :].masked_fill(
                    row0_masked, -float("inf")
                )

            # Remaining-proportional schedule on the CURRENT canvas.
            if step < gen_config.num_step - 1:
                t0, t1 = timesteps[step], timesteps[step + 1]
                k = min(n_mask, max(1, math.ceil(n_mask * (t1 - t0) / (1.0 - t0))))
            else:
                k = int((scores.flatten() > -float("inf")).sum())
                k = min(n_mask, max(1, k)) if k > 0 else n_mask
            k = min(k, int((scores.flatten() > -float("inf")).sum().clamp(min=1)))

            _, topk_idx = torch.topk(scores.flatten(), k)
            flat = sample_tokens.flatten().clone()
            flat[topk_idx] = pred_tokens.flatten()[topk_idx]
            target = flat.view_as(sample_tokens)[0]

            target, n_e, n_d = execute_structure_ops(target, mask_id)
            n_ops_total[0] += n_e
            n_ops_total[1] += n_d
            step += 1

        logger.info(
            "elastic[%d]: init_len=%d final_len=%d expand=%d delete=%d steps=%d",
            i,
            t_len,
            target.shape[1],
            n_ops_total[0],
            n_ops_total[1],
            step,
        )
        results.append(target)
    return results


# ---------------------------------------------------------------------------
# Guarded inference (diagnostic, probe-only; see RESULTS.md 2026-07-06 判决)
# Four inference-side guards, all OFF unless GUARDS dict is set:
#   budget_ratio: |T_cur - T_rule| capped at ratio*T_rule (directional ban)
#   cooldown:     newly expanded mask columns cannot re-expand for K steps
#   theta:        earlier canvas freeze
#   cfg_bypass:   special classes use conditional log-probs (no CFG extrapolation)
# ---------------------------------------------------------------------------
GUARDS = None


def _predict_guarded(model, c_logits, u_logits, gen_config, special_ids, cfg_bypass):
    import torch.nn.functional as F
    from omnivoice.models.omnivoice import _filter_top_k, _gumbel_sample
    if gen_config.guidance_scale != 0:
        c_lp = F.log_softmax(c_logits, dim=-1)
        u_lp = F.log_softmax(u_logits, dim=-1)
        comb = c_lp + gen_config.guidance_scale * (c_lp - u_lp)
        if cfg_bypass:
            for sid in special_ids:
                comb[..., sid] = c_lp[..., sid]
        log_probs = torch.log_softmax(comb, dim=-1)
    else:
        log_probs = F.log_softmax(c_logits, dim=-1)
    log_probs[..., model.config.audio_mask_id] = -float("inf")
    if gen_config.class_temperature > 0.0:
        pred_tokens = _gumbel_sample(
            _filter_top_k(log_probs, ratio=0.1), gen_config.class_temperature
        ).argmax(dim=-1)
    else:
        pred_tokens = log_probs.argmax(dim=-1)
    return pred_tokens, log_probs.max(dim=-1)[0]


def _rebuild_cooldown(old_target, old_cool, K, mask_id, new_len):
    """Companion walk of execute_structure_ops to carry per-column cooldowns."""
    expand_id, delete_id = elastic_ids(mask_id)
    row0 = old_target[0]
    if int((row0 == expand_id).sum()) == 0 and int((row0 == delete_id).sum()) == 0:
        return old_cool
    bcast_from = None
    dp = (row0 == delete_id).nonzero(as_tuple=True)[0]
    if len(dp) > 0:
        j = int(dp[-1])
        suffix = old_target[:, j + 1 :]
        if suffix.numel() == 0 or bool((suffix == mask_id).all()):
            bcast_from = j
    nc = []
    for j in range(row0.shape[0]):
        if bcast_from is not None and j >= bcast_from:
            break
        v = int(row0[j])
        if v == delete_id:
            continue
        if v == expand_id:
            nc += [K, K]
            continue
        nc.append(int(old_cool[j]))
    if not nc:
        nc = [0]
    out = torch.tensor(nc, dtype=torch.long, device=old_cool.device)
    if out.shape[0] != new_len:  # safety: never break generation over a guard
        return torch.zeros(new_len, dtype=torch.long, device=old_cool.device)
    return out


@torch.no_grad()
def generate_iterative_elastic_guarded(model, task, gen_config) -> List[torch.Tensor]:
    g = GUARDS
    mask_id = model.config.audio_mask_id
    expand_id, delete_id = elastic_ids(mask_id)
    C = model.config.num_audio_codebook
    theta = g.get("theta", 0.4)
    lmax_ratio = getattr(gen_config, "elastic_lmax_ratio", 1.5)
    max_extra = getattr(gen_config, "elastic_max_extra_steps", 16)
    from omnivoice.models.omnivoice import _get_time_steps, _gumbel_sample

    timesteps = _get_time_steps(
        t_start=0.0, t_end=1.0, num_step=gen_config.num_step, t_shift=gen_config.t_shift
    ).tolist()

    results: List[torch.Tensor] = []
    for i in range(task.batch_size):
        inputs = model._prepare_inference_inputs(
            task.texts[i], task.target_lens[i], task.ref_texts[i],
            task.ref_audio_tokens[i], task.langs[i], task.instructs[i],
            gen_config.denoise,
        )
        full_ids = inputs["input_ids"][0]
        t_len = task.target_lens[i]
        l_max = max(t_len + 1, int(t_len * lmax_ratio))
        budget = max(1, int(round(g["budget_ratio"] * t_len))) if g.get("budget_ratio", 0) > 0 else None
        prefix = full_ids[:, : full_ids.shape[1] - t_len]
        target = full_ids[:, full_ids.shape[1] - t_len :].clone()
        cool = torch.zeros(t_len, dtype=torch.long, device=model.device)
        layer_ids = torch.arange(C, device=model.device).view(1, -1, 1)

        step = 0
        n_ops_total = [0, 0]
        while True:
            n_mask = int((target == mask_id).sum())
            if n_mask == 0:
                break
            t_cur = target.shape[1]
            c_len = prefix.shape[1] + t_cur
            batch_ids = torch.full((2, C, c_len), mask_id, dtype=torch.long, device=model.device)
            batch_ids[0] = torch.cat([prefix, target], dim=1)
            batch_ids[1, :, :t_cur] = target
            audio_mask = torch.zeros(2, c_len, dtype=torch.bool, device=model.device)
            audio_mask[0, prefix.shape[1] :] = True
            if task.ref_audio_tokens[i] is not None:
                ref_len = task.ref_audio_tokens[i].shape[-1]
                audio_mask[0, prefix.shape[1] - ref_len :] = True
            audio_mask[1, :t_cur] = True
            attn = torch.zeros(2, 1, c_len, c_len, dtype=torch.bool, device=model.device)
            attn[0] = True
            attn[1, :, :t_cur, :t_cur] = True
            if c_len > t_cur:
                diag = torch.arange(t_cur, c_len, device=model.device)
                attn[1, :, diag, diag] = True

            logits = model(input_ids=batch_ids, audio_mask=audio_mask, attention_mask=attn).logits.to(torch.float32)
            c_logits = logits[0:1, :, prefix.shape[1] :, :]
            u_logits = logits[1:2, :, :t_cur, :]

            mask_ratio = n_mask / float(t_cur * C)
            allow_ops = mask_ratio > theta and step < gen_config.num_step
            for lg in (c_logits, u_logits):
                lg[:, 1:, :, expand_id] = -float("inf")
                lg[:, 1:, :, delete_id] = -float("inf")
                if not allow_ops:
                    lg[:, 0, :, expand_id] = -float("inf")
                    lg[:, 0, :, delete_id] = -float("inf")
                else:
                    if t_cur >= l_max:
                        lg[:, 0, :, expand_id] = -float("inf")
                    if budget is not None:
                        if t_cur - t_len >= budget:
                            lg[:, 0, :, expand_id] = -float("inf")
                        if t_len - t_cur >= budget:
                            lg[:, 0, :, delete_id] = -float("inf")
                    if g.get("cooldown", 0) > 0:
                        hot = (cool > 0).nonzero(as_tuple=True)[0]
                        if len(hot) > 0:
                            lg[:, 0, hot, expand_id] = -float("inf")

            if g.get("cfg_bypass", False):
                pred_tokens, scores = _predict_guarded(
                    model, c_logits, u_logits, gen_config, (expand_id, delete_id), True
                )
            else:
                pred_tokens, scores = model._predict_tokens_with_scoring(c_logits, u_logits, gen_config)
            scores = scores - (layer_ids * gen_config.layer_penalty_factor)
            if gen_config.position_temperature > 0.0:
                scores = _gumbel_sample(scores, gen_config.position_temperature)

            sample_tokens = target.unsqueeze(0)
            scores = scores.masked_fill(sample_tokens != mask_id, -float("inf"))
            if allow_ops:
                row0_masked = (sample_tokens[:, 0:1, :] == mask_id).expand(-1, C - 1, -1)
                scores[:, 1:, :] = scores[:, 1:, :].masked_fill(row0_masked, -float("inf"))

            if step < gen_config.num_step - 1:
                t0, t1 = timesteps[step], timesteps[step + 1]
                k = min(n_mask, max(1, math.ceil(n_mask * (t1 - t0) / (1.0 - t0))))
            else:
                k = int((scores.flatten() > -float("inf")).sum())
                k = min(n_mask, max(1, k)) if k > 0 else n_mask
            k = min(k, int((scores.flatten() > -float("inf")).sum().clamp(min=1)))

            _, topk_idx = torch.topk(scores.flatten(), k)
            flat = sample_tokens.flatten().clone()
            flat[topk_idx] = pred_tokens.flatten()[topk_idx]
            target = flat.view_as(sample_tokens)[0]

            old_target, old_cool = target, cool
            target, n_e, n_d = execute_structure_ops(target, mask_id)
            cool = _rebuild_cooldown(old_target, old_cool, g.get("cooldown", 0), mask_id, target.shape[1])
            cool = (cool - 1).clamp(min=0)
            n_ops_total[0] += n_e
            n_ops_total[1] += n_d
            step += 1

        logger.info(
            "elastic-guarded[%d]: init_len=%d final_len=%d expand=%d delete=%d steps=%d",
            i, t_len, target.shape[1], n_ops_total[0], n_ops_total[1], step,
        )
        results.append(target)
    return results
