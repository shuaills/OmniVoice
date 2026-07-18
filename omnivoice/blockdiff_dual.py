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

import hashlib
import math
import random
from typing import Any, Dict, Optional, Tuple

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


# Steady-state HiggsAudioV2 encoding of digital silence, one token per
# codebook (measured 2026-07-10 on 1.0s of zeros through the production
# tokenizer, first 2 edge frames excluded). Upper residual codebooks dither
# within a small silence family; any member is an acceptable target and they
# carry the lowest codebook loss weights.
# Supervision-kind codes (loss contract v2): why a cell is supervised,
# never inferable from token value (silence frames occur as real content).
KIND_IGNORE = 0
KIND_ACOUSTIC = 1
KIND_EOS = 2
KIND_VOID = 3


def build_loss_kind(
    noisy_labels,
    T,
    canvas_len,
    eos_decouple,
    v_hi,
    eos_window=4,
    eos_band_k=1,
):
    """Pure derivation of loss_kind for the noisy copy. Consumes NO RNG:
    every input is an already-computed tensor/int. IGNORE stays 0."""
    if eos_band_k < 1:
        raise ValueError(f"eos_band_k must be >= 1, got {eos_band_k}")
    C = noisy_labels.shape[0]
    kind = torch.zeros((C, canvas_len), dtype=torch.uint8)
    content = noisy_labels[:, :T] != -100
    kind[:, :T][content] = KIND_ACOUSTIC
    if eos_decouple:
        eos_hi = min(T + eos_band_k, canvas_len)
        kind[0, T:eos_hi] = KIND_EOS
        if v_hi > eos_hi:
            kind[:, eos_hi:v_hi] = KIND_VOID
    else:
        kind[0, T:min(T + eos_window, canvas_len)] = KIND_EOS
    return kind


def build_block_markov_prev_ids(
    noisy_copy: torch.Tensor,
    clean_copy: torch.Tensor,
    noisy_block_ids: torch.Tensor,
    *,
    mask_id: int,
) -> torch.Tensor:
    """Build inference-shaped previous-frame inputs without target leakage.

    Inside a noisy block the head sees the actually corrupted previous frame,
    so a masked neighbour remains unknown.  At a block boundary it sees the
    last frame of the preceding committed clean block, matching block-causal
    decode.  The first target-only block has no synthetic predecessor.
    """
    if noisy_copy.ndim != 2 or clean_copy.ndim != 2:
        raise ValueError("clean/noisy copies must have shape [C, T]")
    if noisy_copy.size(0) != clean_copy.size(0):
        raise ValueError("clean/noisy copies must have the same codebook count")
    if tuple(noisy_block_ids.shape) != (noisy_copy.size(1),):
        raise ValueError(
            "noisy_block_ids must have one id per noisy frame, got "
            f"{tuple(noisy_block_ids.shape)} for {tuple(noisy_copy.shape)}"
        )

    prev = torch.full_like(noisy_copy, int(mask_id))
    if noisy_copy.size(1) <= 1:
        return prev

    same_block = noisy_block_ids[1:].eq(noisy_block_ids[:-1])
    committed_anchor = torch.full_like(noisy_copy[:, :-1], int(mask_id))
    anchor_len = min(clean_copy.size(1), committed_anchor.size(1))
    if anchor_len > 0:
        committed_anchor[:, :anchor_len] = clean_copy[:, :anchor_len]
    prev[:, 1:] = torch.where(
        same_block.unsqueeze(0),
        noisy_copy[:, :-1],
        committed_anchor,
    )
    return prev


SILENCE_FRAME_TOKENS = torch.tensor(
    [244, 354, 998, 351, 433, 552, 926, 419], dtype=torch.long
)


# ---------------------------------------------------------------------------
# Training-side processor (two-copy layout)
# ---------------------------------------------------------------------------


class OmniVoiceBlockDualSampleProcessor(OmniVoiceSampleProcessor):
    """Official conditioning draws + two-copy block-causal sample layout."""

    def __init__(self, *args, block_size: int = 32,
                 turn_boundary_prompt_prob: float = 0.0,
                 eos_decouple_silence: bool = False,
                 eos_band_k: int = 1,
                 silence_void_window: int = 32,
                 cfg_branch_training: bool = False,
                 cfg_branch_cond_ratio: float = 0.90,
                 cfg_branch_shared_ratio: float = 0.05,
                 cfg_branch_drop_ref_ratio: float = 0.05,
                 cfg_branch_seed: int = 42,
                 cfg_drop_ref_short_bucket_ratio: float = 0.5,
                 cfg_drop_ref_q_min: int = 1,
                 cfg_drop_ref_q_max: int = 32,
                 cfg_drop_ref_short_q_max: int = 4,
                 block_markov_prev_ids: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size
        self.block_markov_prev_ids = bool(block_markov_prev_ids)
        # With this probability (and when the label carries per-sentence
        # 'turns' timestamps), snap the prompt cut to a sentence END instead of
        # a uniform mid-flow frame. Rationale: inference voice-cloning presents
        # exactly [complete-sentence audio prefix | more text | continue], a
        # junction geometry uniform prompt cuts never produce; without it the
        # model misreads that junction as utterance end (instant EOS / silence
        # onset, B2F-10k verdict 2026-07-08).
        self.turn_boundary_prompt_prob = turn_boundary_prompt_prob
        # EOS/padding decoupling (see TrainingConfig.eos_decouple_silence).
        self.eos_decouple_silence = eos_decouple_silence
        if eos_band_k < 1:
            raise ValueError(f"eos_band_k must be >= 1, got {eos_band_k}")
        if not eos_decouple_silence and eos_band_k != 1:
            raise ValueError(
                "eos_band_k != 1 requires eos_decouple_silence=True"
            )
        self.eos_band_k = eos_band_k
        self.silence_void_window = silence_void_window

        self.cfg_branch_training = cfg_branch_training
        self.cfg_branch_cond_ratio = cfg_branch_cond_ratio
        self.cfg_branch_shared_ratio = cfg_branch_shared_ratio
        self.cfg_branch_drop_ref_ratio = cfg_branch_drop_ref_ratio
        self.cfg_branch_seed = int(cfg_branch_seed)
        self.cfg_drop_ref_short_bucket_ratio = cfg_drop_ref_short_bucket_ratio
        self.cfg_drop_ref_q_min = cfg_drop_ref_q_min
        self.cfg_drop_ref_q_max = cfg_drop_ref_q_max
        self.cfg_drop_ref_short_q_max = cfg_drop_ref_short_q_max
        if cfg_branch_training:
            branch_ratios = (
                cfg_branch_cond_ratio,
                cfg_branch_shared_ratio,
                cfg_branch_drop_ref_ratio,
            )
            if any(not math.isfinite(ratio) or ratio < 0 for ratio in branch_ratios):
                raise ValueError("CFG branch ratios must be finite and non-negative")
            if not math.isclose(sum(branch_ratios), 1.0, abs_tol=1e-9):
                raise ValueError(
                    "CFG branch ratios must sum to 1, got "
                    f"{sum(branch_ratios)}"
                )
            if not 0.0 <= cfg_drop_ref_short_bucket_ratio <= 1.0:
                raise ValueError(
                    "cfg_drop_ref_short_bucket_ratio must be in [0, 1]"
                )
            if not 1 <= cfg_drop_ref_q_min <= cfg_drop_ref_q_max <= block_size:
                raise ValueError(
                    "CFG drop-ref q range must satisfy "
                    f"1 <= min <= max <= block_size, got "
                    f"[{cfg_drop_ref_q_min}, {cfg_drop_ref_q_max}] and "
                    f"block_size={block_size}"
                )
            if cfg_drop_ref_short_q_max < 1:
                raise ValueError("cfg_drop_ref_short_q_max must be >= 1")

    def _cfg_rng(self, sample: Dict[str, Any]) -> random.Random:
        """Per-sample CFG stream; never advances Python's global RNG."""
        label = sample.get("label", {})
        sample_key = label.get("id", label.get("idx"))
        if sample_key is None:
            raise ValueError(
                "cfg_branch_training requires a stable label.id or label.idx"
            )
        material = f"{self.cfg_branch_seed}\0{sample_key}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
        return random.Random(seed)

    def _choose_cfg_branch(self, rng: random.Random) -> str:
        draw = rng.random()
        if draw < self.cfg_branch_cond_ratio:
            return "C"
        if draw < self.cfg_branch_cond_ratio + self.cfg_branch_shared_ratio:
            return "U_shared"
        return "U_drop_ref"

    def _choose_drop_ref_phase(
        self, target_length: int, rng: random.Random
    ) -> Dict[str, Any]:
        """Choose bucket, q, then a legal cut using only the keyed RNG."""
        if target_length < 2:
            raise ValueError(
                "U_drop_ref needs at least two acoustic frames so 1 <= S < T"
            )
        full_q = list(range(self.cfg_drop_ref_q_min, self.cfg_drop_ref_q_max + 1))
        short_q = [q for q in full_q if q <= self.cfg_drop_ref_short_q_max]
        requested_short = rng.random() < self.cfg_drop_ref_short_bucket_ratio
        requested_pool = short_q if requested_short and short_q else full_q
        requested_q = rng.choice(requested_pool)

        legal_cuts = {q: [] for q in full_q}
        for cut in range(1, target_length):
            remainder = cut % self.block_size
            q = self.block_size - remainder if remainder else self.block_size
            if q in legal_cuts:
                legal_cuts[q].append(cut)
        legal_q = [q for q in full_q if legal_cuts[q]]
        if not legal_q:
            raise ValueError(
                "U_drop_ref has no legal cut for configured q range "
                f"[{self.cfg_drop_ref_q_min}, {self.cfg_drop_ref_q_max}] "
                f"at T={target_length}"
            )

        actual_q = requested_q
        rebucketed = not legal_cuts[requested_q]
        if rebucketed:
            short_mix = (
                self.cfg_drop_ref_short_bucket_ratio if short_q else 0.0
            )
            weights = []
            for q in legal_q:
                weight = (1.0 - short_mix) / len(full_q)
                if q in short_q:
                    weight += short_mix / len(short_q)
                weights.append(weight)
            total_weight = sum(weights)
            if total_weight == 0.0:
                weights = [1.0] * len(legal_q)
                total_weight = float(len(legal_q))
            threshold = rng.random() * total_weight
            cumulative = 0.0
            actual_q = legal_q[-1]
            for q, weight in zip(legal_q, weights):
                cumulative += weight
                if threshold < cumulative:
                    actual_q = q
                    break

        cut = rng.choice(legal_cuts[actual_q])
        return {
            "requested_short_bucket": requested_short,
            "requested_q": requested_q,
            "actual_q": actual_q,
            "prompt_cut": cut,
            "rebucketed": rebucketed,
        }

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # --- official draw order (verbatim; global mask_ratio draw is kept
        # for stream stability but superseded by per-block ratios) ---
        cfg_rng = None
        if "clean_start_token_idx" in sample["label"]:
            branch = "C"
        else:
            # Preserve the historical branch draw even when the new contract
            # is enabled.  The keyed CFG stream selects the candidate branch;
            # this discarded legacy draw keeps all later global draws aligned.
            legacy_drop_cond = random.uniform(0, 1) < self.drop_cond_ratio
            if self.cfg_branch_training:
                cfg_rng = self._cfg_rng(sample)
                branch = self._choose_cfg_branch(cfg_rng)
            else:
                branch = "U_drop_ref" if legacy_drop_cond else "C"

        if branch == "U_drop_ref":
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
            drop_text = branch == "U_shared"

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

        bs = self.block_size
        source_audio_tokens = sample["audio_tokens"].long()
        C, source_T = source_audio_tokens.shape
        phase = None
        ragged_q = None

        if self.cfg_branch_training and branch == "U_drop_ref":
            if cfg_rng is None:
                raise AssertionError("U_drop_ref branch is missing its keyed RNG")
            phase = self._choose_drop_ref_phase(source_T, cfg_rng)
            source_prompt_length = phase["prompt_cut"]
            ragged_q = phase["actual_q"]
            # Physical reference removal: the target-only U timeline restarts
            # at position zero, exactly like inference cfg_policy=drop_ref.
            audio_tokens = source_audio_tokens[:, source_prompt_length:].clone()
            prompt_length = 0
        else:
            audio_tokens = source_audio_tokens
            if "clean_start_token_idx" in sample["label"]:
                prompt_length = sample["label"]["clean_start_token_idx"]
            else:
                prompt_length = int(source_T * prompt_ratio)
                turns = sample["label"].get("turns")
                if (
                    turns
                    and prompt_ratio > 0
                    and random.uniform(0, 1) < self.turn_boundary_prompt_prob
                ):
                    bounds = []
                    for t in turns:
                        try:
                            b = int(
                                round((t["start_s"] + t["duration_s"]) * 25)
                            )
                        except (KeyError, TypeError):
                            continue
                        if 0.05 * source_T <= b <= 0.7 * source_T:
                            bounds.append(min(b, source_T))
                    if bounds:
                        prompt_length = random.choice(bounds)
            if self.cfg_branch_training and branch == "U_shared":
                if source_T < 2:
                    raise ValueError("U_shared needs at least two acoustic frames")
                prompt_length = min(max(prompt_length, 1), source_T - 1)
            source_prompt_length = prompt_length

        T = audio_tokens.shape[1]
        if ragged_q is None:
            n_blocks = T // bs + 1  # trailing block always reaches EOS fill
            clean_len = (n_blocks - 1) * bs  # complete content blocks only
            canvas_len = n_blocks * bs
            block_bounds = [b * bs for b in range(n_blocks + 1)]
        else:
            # First target-only block has q columns.  Every later block has bs
            # columns, and the canvas boundary is strictly after the target;
            # landing exactly on a boundary therefore appends a pure EOS block.
            block_bounds = [0, ragged_q]
            while block_bounds[-1] <= T:
                block_bounds.append(block_bounds[-1] + bs)
            n_blocks = len(block_bounds) - 1
            clean_len = block_bounds[-2]
            canvas_len = block_bounds[-1]
        eos = block_eos_id(self.audio_mask_id)

        clean_copy = audio_tokens[:, :clean_len].clone()

        noisy_copy = torch.full((C, canvas_len), self.audio_mask_id, dtype=torch.long)
        noisy_copy[:, :T] = audio_tokens
        eos_band_k = getattr(self, "eos_band_k", 1)
        eos_hi = min(T + eos_band_k, canvas_len)
        if self.eos_decouple_silence:
            # Terminal inputs have always been hard-masked in this processor.
            # Keep the whole EOS band in that single atomic state: introducing
            # an independent Bernoulli per cell would leak clean EOS siblings,
            # while introducing a new shared draw would break k=1 RNG parity.
            noisy_copy[:, T:eos_hi] = self.audio_mask_id
        noisy_labels = torch.full((C, canvas_len), -100, dtype=torch.long)
        for b in range(n_blocks):
            lo, hi = block_bounds[b], min(block_bounds[b + 1], T)
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
        # EOS fill: cols >= T are mask on input. [eos] label on cb0 only, and
        # only on a short window after content end (point-ish event, not a
        # region): the old full-region fill (labels[0, T:] = eos, ~32 cols/sample)
        # taught an "end zone" prior over any [content | masked-suffix] block
        # pattern, which at continuation onset expressed as instant EOS
        # (30.6% first-block truncation) or, when EOS-banned, a displaced
        # silence run (the universal onset hum). Verified 2026-07-07:
        # tests/eos_displacement_test.py (ban=0 -> 3/4 instant EOS).
        if self.eos_decouple_silence:
            # Decoupled roles: [eos] is one stop EVENT beginning at T.  It may
            # occupy a short, atomically-masked band, normalized as one event
            # by split_loss.  Void starts after the band and is supervised as
            # the real silence frame on every codebook. The void must have a
            # defined, data-real value or parallel demasking commits junk
            # there at inference (tail beep between end-of-speech and EOS).
            sil = SILENCE_FRAME_TOKENS
            if C > sil.numel():
                raise ValueError(
                    f"SILENCE_FRAME_TOKENS covers {sil.numel()} codebooks, "
                    f"got C={C}"
                )
            noisy_labels[0, T:eos_hi] = eos
            v_hi = min(eos_hi + self.silence_void_window, canvas_len)
            if v_hi > eos_hi:
                noisy_labels[:, eos_hi:v_hi] = sil[:C].unsqueeze(1)
        else:
            eos_window = 4
            noisy_labels[0, T:min(T + eos_window, canvas_len)] = eos

        _v_hi = (min(eos_hi + self.silence_void_window, canvas_len)
                 if self.eos_decouple_silence else T + 1)
        kind_noisy = build_loss_kind(
            noisy_labels,
            T,
            canvas_len,
            self.eos_decouple_silence,
            _v_hi,
            eos_band_k=eos_band_k,
        )

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
        if ragged_q is None:
            clean_block_idx = torch.arange(clean_len, dtype=torch.int32) // bs
            noisy_block_idx = torch.arange(canvas_len, dtype=torch.int32) // bs
        else:
            def _ragged_ids(length):
                positions = torch.arange(length, dtype=torch.int32)
                return torch.where(
                    positions < ragged_q,
                    torch.zeros_like(positions),
                    1 + (positions - ragged_q) // bs,
                )

            clean_block_idx = _ragged_ids(clean_len)
            noisy_block_idx = _ragged_ids(canvas_len)
        block_idx = torch.cat(
            [
                torch.full((P,), -1, dtype=torch.int32),
                clean_block_idx,
                noisy_block_idx,
            ]
        )

        loss_kind = torch.cat(
            [
                torch.zeros((C, P), dtype=torch.uint8),
                torch.zeros((C, clean_len), dtype=torch.uint8),
                kind_noisy,
            ],
            dim=1,
        )
        if (loss_kind == KIND_IGNORE) .ne(labels == -100).any():
            raise AssertionError("loss_kind invariant violated: IGNORE <-> label==-100")

        output = {
            "input_ids": input_ids,
            "labels": labels,
            "audio_mask": audio_mask,
            "length": total_length,
            "position_ids": position_ids,
            "copy_tag": copy_tag,
            "block_idx": block_idx,
            "loss_kind": loss_kind,
        }
        if self.block_markov_prev_ids:
            noisy_prev_ids = build_block_markov_prev_ids(
                noisy_copy,
                clean_copy,
                noisy_block_idx,
                mask_id=self.audio_mask_id,
            )
            output["markov_prev_ids"] = torch.cat(
                [
                    torch.full(
                        (C, P + clean_len),
                        self.audio_mask_id,
                        dtype=torch.long,
                    ),
                    noisy_prev_ids,
                ],
                dim=1,
            )
        if self.cfg_branch_training:
            output.update(
                {
                    "cfg_branch": branch,
                    "cfg_prompt_cut": source_prompt_length,
                    "cfg_target_frames": source_T - source_prompt_length,
                    "cfg_reference_frames": (
                        0 if branch == "U_drop_ref" else source_prompt_length
                    ),
                    "cfg_reference_leak_tokens": (
                        0 if branch == "U_drop_ref" else source_prompt_length * C
                    ),
                    "cfg_requested_q": (
                        phase["requested_q"] if phase is not None else None
                    ),
                    "cfg_actual_q": (
                        phase["actual_q"] if phase is not None else None
                    ),
                    "cfg_rebucketed": (
                        phase["rebucketed"] if phase is not None else False
                    ),
                    "cfg_eos_band_width": eos_hi - T,
                    "cfg_supervised_cells": int((labels != -100).sum().item()),
                }
            )
        return output


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


def _forward_slices(
    model,
    ids,
    positions,
    attn4d,
    past_key_values=None,
    first_prev_ids=None,
):
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
    markov_prev_ids = None
    if model.block_markov_head is not None and first_prev_ids is not None:
        if tuple(first_prev_ids.shape) != (model.config.num_audio_codebook,):
            raise ValueError(
                "first_prev_ids must have one id per codebook, got "
                f"{tuple(first_prev_ids.shape)}"
            )
        from omnivoice.models.block_markov import infer_adjacent_audio_prev_ids

        markov_prev_ids = infer_adjacent_audio_prev_ids(
            ids.unsqueeze(0),
            amask,
            mask_id=model.config.audio_mask_id,
            first_prev_ids=first_prev_ids.unsqueeze(0),
        )
    return model._compute_audio_logits(
        h,
        input_ids=ids.unsqueeze(0),
        audio_mask=amask,
        markov_prev_ids=markov_prev_ids,
    )


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


def gen_frame_positions(committed_len, seed_total, bs, device):
    """True generated-frame index for each column of the current noisy block.

    committed_len counts full committed blocks (which, after the first
    generated block, INCLUDE any partial-prompt prefill seed_rem), so the
    generated count must be taken against seed_total, not seed_blocks*bs.
    Prefilled prompt columns come out negative; callers must additionally
    exclude them (jr < n_pre) from any generated-position logic.
    """
    jr = torch.arange(bs, device=device)
    return committed_len - seed_total + jr, jr


def _find_silence_run_start(
    audio_tokens: torch.Tensor,
    min_run_frames: int,
    *,
    match_codebooks: int = 2,
    start_frame: int = 0,
) -> Optional[int]:
    """Return the earliest qualifying digital-silence run, or ``None``.

    Only the leading ``match_codebooks`` are compared.  The steady-state
    silence probe is stable in the coarse codebooks, while the upper residual
    codebooks legitimately dither within a small silence family.  Requiring a
    full eight-codebook match would therefore miss real quiet tails.

    ``start_frame`` excludes the minimum-generation region from consideration;
    callers can safely trim at the returned index without violating that guard.
    """
    if audio_tokens.ndim != 2:
        raise ValueError(
            f"audio_tokens must have shape [C,T], got {tuple(audio_tokens.shape)}"
        )
    if min_run_frames < 1:
        raise ValueError(f"min_run_frames must be >= 1, got {min_run_frames}")
    if not 1 <= match_codebooks <= audio_tokens.shape[0]:
        raise ValueError(
            "match_codebooks must be in [1, C], got "
            f"{match_codebooks} for C={audio_tokens.shape[0]}"
        )
    if match_codebooks > SILENCE_FRAME_TOKENS.numel():
        raise ValueError(
            f"silence reference covers {SILENCE_FRAME_TOKENS.numel()} codebooks, "
            f"got match_codebooks={match_codebooks}"
        )

    start_frame = max(0, int(start_frame))
    if audio_tokens.shape[1] - start_frame < min_run_frames:
        return None
    reference = SILENCE_FRAME_TOKENS[:match_codebooks].to(
        device=audio_tokens.device, dtype=audio_tokens.dtype
    )
    matches = (
        audio_tokens[:match_codebooks, start_frame:]
        == reference.unsqueeze(1)
    ).all(dim=0)
    qualifying = matches.unfold(0, min_run_frames, 1).all(dim=1)
    hits = qualifying.nonzero(as_tuple=True)[0]
    if hits.numel() == 0:
        return None
    return start_frame + int(hits[0])


def _choose_termination(
    eos_col: Optional[int],
    silence_run_start: Optional[int],
    silence_run_frames: int,
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Choose the first observed stop signal.

    Returns ``(reason, trim_col, trigger_col)`` in generated-frame coordinates.
    An EOS is both observed and trimmed at its own column.  A silence stop is
    observed only after the final frame of the qualifying run, but trims back
    to the run's start so the verified waiting gap is not returned as audio.
    EOS wins ties: if it appears by the time the silence threshold is reached,
    the model terminated normally and the fallback did not fire.
    """
    if silence_run_frames < 1 and silence_run_start is not None:
        raise ValueError(
            "silence_run_frames must be >= 1 when silence_run_start is set"
        )

    candidates = []
    if eos_col is not None:
        eos_col = int(eos_col)
        candidates.append((eos_col, 0, "eos", eos_col))
    if silence_run_start is not None:
        silence_run_start = int(silence_run_start)
        silence_trigger = silence_run_start + silence_run_frames - 1
        candidates.append(
            (silence_trigger, 1, "silence", silence_run_start)
        )
    if not candidates:
        return None, None, None

    trigger_col, _, reason, trim_col = min(candidates)
    return reason, trim_col, trigger_col


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
    min_gen_frames: int = 0,
    silence_run_frames: int = 0,
    silence_match_codebooks: int = 2,
    cfg_unconditional_seed_policy: str = "shared",
    eos_cfg_trace: Optional[list] = None,
):
    """Block-causal decode.  Returns (generated [C, G], stats).

    cache path: prefix and committed blocks live in a DynamicCache; per
    unmasking step only the current block is forwarded (queries see all
    cached keys + the block itself); intra-block steps are cropped off the
    cache, the final committed block is appended once.
    recompute path: every step rebuilds [prefix | committed | cur] with the
    dense rule mask -- geometrically identical, no cache. Gate 3 asserts the
    two produce identical tokens.

    ``cfg_unconditional_seed_policy`` controls only the CFG-unconditional
    branch. ``shared`` preserves the historical geometry exactly: reference
    seed blocks (including a partial seed in the first current block) are
    shared by the conditional and unconditional branches. ``drop_ref`` gives
    the unconditional branch a separate, generated-only timeline: no full or
    partial reference token enters its inputs/cache, and target positions start
    at zero and advance only when generated target blocks are committed.  With
    a partial reference, its first target-only block is intentionally short;
    this is an experimental CFG geometry, not a claim of fixed-width training
    parity.  Its cache and full-recompute implementations are still exact
    counterparts of each other.
    """
    from omnivoice.blockdiff import (
        _calibrate_blockwise_eos_log_probs,
        _predict_tokens_blockwise,
    )
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
    if cfg_unconditional_seed_policy not in {"shared", "drop_ref"}:
        raise ValueError(
            "cfg_unconditional_seed_policy must be 'shared' or 'drop_ref', "
            f"got {cfg_unconditional_seed_policy!r}"
        )
    if eos_cfg_trace is not None and float(
        getattr(gen_config, "class_temperature", 0.0)
    ) > 0.0:
        raise ValueError(
            "eos_cfg_trace counterfactuals require class_temperature=0"
        )
    drop_ref_from_unconditional = (
        use_cfg and cfg_unconditional_seed_policy == "drop_ref"
    )

    committed = torch.empty((C, 0), dtype=torch.long, device=device)
    seed_blocks, seed_rem, seed_total = 0, None, 0
    if seed_audio is not None and seed_audio.size(1) > 0:
        seed_total = seed_audio.size(1)
        S = (seed_total // bs) * bs
        committed = seed_audio[:, :S].to(device).long()
        seed_blocks = S // bs
        if seed_total > S:
            seed_rem = seed_audio[:, S:].to(device).long()

    # The drop-ref CFG branch owns a generated-only history.  Block ids track
    # decode commits rather than fixed-width frame buckets because a partial
    # reference makes the first generated target block shorter than ``bs``.
    # RoPE positions remain dense in target-frame time (0, 1, ...).  The first
    # short block is an explicit experimental approximation, not fixed-width
    # training parity; cache and recompute nevertheless use the same block id.
    u_committed = torch.empty((C, 0), dtype=torch.long, device=device)
    u_committed_block_ids = torch.empty(
        (0,), dtype=torch.int32, device=device
    )
    u_next_block = 0

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
            if use_cfg and not drop_ref_from_unconditional:
                Ku = caches["u"].get_seq_length()
                attn_u = torch.ones((1, 1, bs, Ku + bs), dtype=torch.bool, device=device)
                _forward_slices(model, blk_t, sb * bs + torch.arange(bs, device=device), attn_u, caches["u"])

    if silence_run_frames < 0:
        raise ValueError(
            f"silence_run_frames must be >= 0, got {silence_run_frames}"
        )
    if not 1 <= silence_match_codebooks <= C:
        raise ValueError(
            "silence_match_codebooks must be in [1, C], got "
            f"{silence_match_codebooks} for C={C}"
        )

    stats = {
        "n_blocks": 0,
        "stopped_by_eos": False,
        "eos_col": None,
        "stopped_by_silence": False,
        "silence_col": None,
        "silence_trigger_col": None,
        "stop_reason": None,
    }

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
        u_cur_start = n_pre if drop_ref_from_unconditional else 0
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
                    model,
                    cur,
                    cur_pos,
                    attn,
                    caches["c"],
                    first_prev_ids=(
                        committed[:, -1] if committed.size(1) > 0 else None
                    ),
                )
                _crop_cache(caches["c"], Kc)
                if use_cfg:
                    Ku = caches["u"].get_seq_length()
                    if drop_ref_from_unconditional:
                        if Ku != u_committed.size(1):
                            raise RuntimeError(
                                "drop_ref unconditional cache/history mismatch: "
                                f"cache={Ku}, generated={u_committed.size(1)}"
                            )
                        u_cur = cur[:, u_cur_start:]
                        u_len = u_cur.size(1)
                        attn_u = torch.ones(
                            (1, 1, u_len, Ku + u_len),
                            dtype=torch.bool,
                            device=device,
                        )
                        u_pos = u_committed.size(1) + torch.arange(
                            u_len, device=device
                        )
                        u_logits = _forward_slices(
                            model,
                            u_cur,
                            u_pos,
                            attn_u,
                            caches["u"],
                            first_prev_ids=(
                                u_committed[:, -1]
                                if u_committed.size(1) > 0
                                else None
                            ),
                        )
                    else:
                        attn_u = torch.ones(
                            (1, 1, bs, Ku + bs),
                            dtype=torch.bool,
                            device=device,
                        )
                        u_pos = b * bs + torch.arange(bs, device=device)
                        u_logits = _forward_slices(
                            model,
                            cur,
                            u_pos,
                            attn_u,
                            caches["u"],
                            first_prev_ids=(
                                committed[:, -1]
                                if committed.size(1) > 0
                                else None
                            ),
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
                    if drop_ref_from_unconditional:
                        u_cur = cur[:, u_cur_start:]
                        u_len = u_cur.size(1)
                        sequ = torch.cat([u_committed, u_cur], dim=1)
                        Lu = sequ.size(1)
                        du = torch.zeros(
                            Lu, dtype=torch.int32, device=device
                        )
                        tu = torch.cat(
                            [
                                torch.full(
                                    (u_committed.size(1),),
                                    TAG_CLEAN,
                                    dtype=torch.int32,
                                    device=device,
                                ),
                                torch.full(
                                    (u_len,),
                                    TAG_NOISY,
                                    dtype=torch.int32,
                                    device=device,
                                ),
                            ]
                        )
                        blku = torch.cat(
                            [
                                u_committed_block_ids,
                                torch.full(
                                    (u_len,),
                                    u_next_block,
                                    dtype=torch.int32,
                                    device=device,
                                ),
                            ]
                        )
                        attnu = build_block_causal_attn_mask(
                            du, tu, blku
                        ).to(device)
                        posu = torch.arange(Lu, device=device)
                        logits_u = _forward_slices(
                            model, sequ, posu, attnu
                        )
                        u_logits = logits_u[:, :, Lu - u_len :, :]
                    else:
                        sequ = torch.cat([committed, cur], dim=1)
                        Lu = sequ.size(1)
                        du = torch.zeros(
                            Lu, dtype=torch.int32, device=device
                        )
                        tu = t[P:]
                        blku = blk[P:]
                        attnu = build_block_causal_attn_mask(
                            du, tu, blku
                        ).to(device)
                        posu = torch.cat(
                            [
                                torch.arange(
                                    committed.size(1), device=device
                                ),
                                b * bs + torch.arange(bs, device=device),
                            ]
                        )
                        logits_u = _forward_slices(
                            model, sequ, posu, attnu
                        )
                        u_logits = logits_u[:, :, Lu - bs :, :]
                else:
                    u_logits = c_logits

            if min_gen_frames > 0:
                gen_pos, jr = gen_frame_positions(
                    committed.size(1), seed_total, bs, device)
                ban_cols = (jr >= n_pre) & (gen_pos < min_gen_frames)
                if ban_cols.any():
                    # Pre-CFG bans must stay finite: -inf in both c and u makes
                    # the log-prob combine produce NaN (inf - inf), and the
                    # outer log_softmax spreads it across the whole row, whose
                    # argmax then lands on index 0. finfo.min rather than a
                    # hand constant: -1e30 overflows back to -inf in fp16.
                    _neg = torch.finfo(c_logits.dtype).min
                    c_logits[0, 0, ban_cols, eos] = _neg
                    if u_logits is not c_logits:
                        u_ban_cols = (
                            ban_cols[u_cur_start:]
                            if drop_ref_from_unconditional
                            else ban_cols
                        )
                        u_logits[0, 0, u_ban_cols, eos] = _neg

            # Under drop_ref the CFG pair is evaluated only on aligned target
            # columns.  The conditional reference prefix is neither padded nor
            # represented in the unconditional input/cache.
            c_cfg_logits = (
                c_logits[:, :, u_cur_start:, :]
                if drop_ref_from_unconditional
                else c_logits
            )
            pred_tokens, scores = _predict_tokens_blockwise(
                model,
                c_cfg_logits.to(torch.float32),
                u_logits.to(torch.float32),
                gen_config,
            )
            if logit_trace is not None:
                logit_trace.append(c_logits.to(torch.float32).cpu())
            scores = scores - (layer_ids * gen_config.layer_penalty_factor)
            if gen_config.position_temperature > 0.0:
                scores = _gumbel_sample(scores, gen_config.position_temperature)
            cur_view = (
                cur[:, u_cur_start:].unsqueeze(0)
                if drop_ref_from_unconditional
                else cur.unsqueeze(0)
            )
            scores = scores.masked_fill(cur_view != mask_id, -float("inf"))
            _, topk_idx = torch.topk(scores.flatten(), k)
            if eos_cfg_trace is not None:
                eos_cfg_trace.append(
                    _eos_cfg_step_trace(
                        c_cfg_logits.to(torch.float32),
                        u_logits.to(torch.float32),
                        gen_config,
                        mask_id=mask_id,
                        eos_id=eos,
                        active_cb0=(cur_view[0, 0] == mask_id),
                        pred_tokens=pred_tokens,
                        queue_scores=scores,
                        selected_flat_indices=topk_idx,
                        block_index=b - seed_blocks,
                        step_index=step,
                        scheduled_positions=k,
                        calibrate=_calibrate_blockwise_eos_log_probs,
                    )
                )
            flat = cur_view.flatten().clone()
            flat[topk_idx] = pred_tokens.flatten()[topk_idx]
            if drop_ref_from_unconditional:
                cur[:, u_cur_start:] = flat.view(C, bs - u_cur_start)
            else:
                cur = flat.view(C, bs)

        if use_kv_cache:
            # Commit: append the final block to the caches (clean geometry --
            # queries see cache + block itself, exactly the training rule).
            Kc = caches["c"].get_seq_length()
            attn = torch.ones((1, 1, bs, Kc + bs), dtype=torch.bool, device=device)
            _ = _forward_slices(model, cur, cur_pos, attn, caches["c"])
            if use_cfg:
                Ku = caches["u"].get_seq_length()
                if drop_ref_from_unconditional:
                    if Ku != u_committed.size(1):
                        raise RuntimeError(
                            "drop_ref unconditional cache/history mismatch: "
                            f"cache={Ku}, generated={u_committed.size(1)}"
                        )
                    u_cur = cur[:, u_cur_start:]
                    u_len = u_cur.size(1)
                    attn_u = torch.ones(
                        (1, 1, u_len, Ku + u_len),
                        dtype=torch.bool,
                        device=device,
                    )
                    u_pos = u_committed.size(1) + torch.arange(
                        u_len, device=device
                    )
                    _ = _forward_slices(
                        model, u_cur, u_pos, attn_u, caches["u"]
                    )
                else:
                    attn_u = torch.ones(
                        (1, 1, bs, Ku + bs),
                        dtype=torch.bool,
                        device=device,
                    )
                    u_pos = b * bs + torch.arange(bs, device=device)
                    _ = _forward_slices(
                        model, cur, u_pos, attn_u, caches["u"]
                    )

        if use_cfg and drop_ref_from_unconditional:
            u_cur = cur[:, u_cur_start:]
            u_committed = torch.cat([u_committed, u_cur], dim=1)
            u_committed_block_ids = torch.cat(
                [
                    u_committed_block_ids,
                    torch.full(
                        (u_cur.size(1),),
                        u_next_block,
                        dtype=torch.int32,
                        device=device,
                    ),
                ]
            )
            u_next_block += 1

        committed = torch.cat([committed, cur], dim=1)
        stats["n_blocks"] = b + 1

        eos_generated_col = None
        eos_hits = (cur[0] == eos).nonzero(as_tuple=True)[0]
        if eos_hits.numel() > 0:
            eos_abs = committed.size(1) - bs + int(eos_hits[0])
            eos_generated_col = eos_abs - seed_total

        silence_start = None
        if silence_run_frames > 0:
            generated = committed[:, seed_total:]
            silence_start = _find_silence_run_start(
                generated,
                silence_run_frames,
                match_codebooks=silence_match_codebooks,
                start_frame=min_gen_frames,
            )

        stop_reason, trim_generated_col, trigger_generated_col = _choose_termination(
            eos_generated_col,
            silence_start,
            silence_run_frames,
        )
        if stop_reason is not None:
            stop_abs = seed_total + trim_generated_col
            stats["stop_reason"] = stop_reason
            if stop_reason == "eos":
                stats["stopped_by_eos"] = True
                stats["eos_col"] = stop_abs
            else:
                stats["stopped_by_silence"] = True
                stats["silence_col"] = stop_abs
                stats["silence_trigger_col"] = (
                    seed_total + trigger_generated_col
                )
            committed = committed[:, :stop_abs]
            break

    return committed[:, seed_total:], stats


def _finite_float(value: torch.Tensor) -> Optional[float]:
    """Convert a scalar tensor to a JSON-safe float."""
    result = float(value.item())
    return result if math.isfinite(result) else None


def _eos_cfg_step_trace(
    c_logits: torch.Tensor,
    u_logits: torch.Tensor,
    gen_config,
    *,
    mask_id: int,
    eos_id: int,
    active_cb0: torch.Tensor,
    pred_tokens: torch.Tensor,
    queue_scores: torch.Tensor,
    selected_flat_indices: torch.Tensor,
    block_index: int,
    step_index: int,
    scheduled_positions: int,
    calibrate,
) -> Dict[str, Any]:
    """Summarize EOS score calibration for one reveal step.

    Only codebook 0 can emit EOS.  To keep traces compact, the scalar score
    details describe the strongest actual EOS candidate when one exists, or
    otherwise the active column with the strongest calibrated EOS margin.
    Besides guided/legacy/calibrated row scores, the trace reconstructs the
    legacy reveal queue with the *same* sampled position noise.  The function
    is called only when tracing is explicitly enabled.
    """
    c_log_probs = torch.log_softmax(c_logits, dim=-1)
    if gen_config.guidance_scale != 0.0:
        u_log_probs = torch.log_softmax(u_logits, dim=-1)
        cfg_log_probs = torch.log_softmax(
            c_log_probs
            + gen_config.guidance_scale * (c_log_probs - u_log_probs),
            dim=-1,
        )
    else:
        cfg_log_probs = c_log_probs

    policy = getattr(gen_config, "eos_cfg_calibration", "legacy")
    post_log_scores = calibrate(
        cfg_log_probs,
        c_log_probs,
        mask_id,
        eos_id,
        mode=policy,
    )
    legacy_log_scores = calibrate(
        cfg_log_probs,
        c_log_probs,
        mask_id,
        eos_id,
        mode="legacy",
    )

    # Match the legal class set used by the scorer before comparing margins.
    legal_cfg = cfg_log_probs.clone()
    legal_cfg[..., mask_id] = -float("inf")
    if legal_cfg.size(-1) > eos_id + 1:
        legal_cfg[..., eos_id + 1 :] = -float("inf")
    guided_non_eos = legal_cfg[:, 0, :, :eos_id].amax(dim=-1)
    legacy_non_eos = legacy_log_scores[:, 0, :, :eos_id].amax(dim=-1)
    post_non_eos = post_log_scores[:, 0, :, :eos_id].amax(dim=-1)
    guided_margin = legal_cfg[:, 0, :, eos_id] - guided_non_eos
    legacy_margin = legacy_log_scores[:, 0, :, eos_id] - legacy_non_eos
    post_margin = post_log_scores[:, 0, :, eos_id] - post_non_eos

    legacy_confidence = legacy_log_scores.max(dim=-1).values
    calibrated_confidence = post_log_scores.max(dim=-1).values
    confidence_delta = legacy_confidence - calibrated_confidence
    position_temperature = float(
        getattr(gen_config, "position_temperature", 0.0)
    )
    if position_temperature > 0.0:
        confidence_delta = confidence_delta / position_temperature
    legacy_queue_scores = queue_scores + confidence_delta
    k = int(selected_flat_indices.numel())
    legacy_topk_indices = torch.topk(
        legacy_queue_scores.flatten(), k
    ).indices

    active_columns = active_cb0.nonzero(as_tuple=True)[0]
    class_eos = (pred_tokens[0, 0] == eos_id) & active_cb0
    legacy_pred_tokens = legacy_log_scores.argmax(dim=-1)
    legacy_class_eos = (legacy_pred_tokens[0, 0] == eos_id) & active_cb0
    width = pred_tokens.size(-1)
    selected_columns = sorted(
        int(index.item()) % width
        for index in selected_flat_indices
        if int(index.item()) < width
        and int(pred_tokens.flatten()[index].item()) == eos_id
    )
    legacy_selected_columns = sorted(
        int(index.item()) % width
        for index in legacy_topk_indices
        if int(index.item()) < width
        and int(legacy_pred_tokens.flatten()[index].item()) == eos_id
    )
    record: Dict[str, Any] = {
        "block": int(block_index),
        "step": int(step_index),
        "policy": policy,
        "scheduled": int(scheduled_positions),
        "active_cb0": int(active_columns.numel()),
        "class_eos": int(class_eos.sum().item()),
        "legacy_class_eos": int(legacy_class_eos.sum().item()),
        "selected_eos_cols": selected_columns,
        "legacy_selected_eos_cols": legacy_selected_columns,
    }
    if active_columns.numel() == 0:
        record.update(
            {
                "candidate_col": None,
                "guided_eos_mass": None,
                "conditional_eos_mass": None,
                "legacy_eos_mass": None,
                "post_eos_mass": None,
                "legacy_total_mass": None,
                "post_total_mass": None,
                "guided_margin": None,
                "legacy_margin": None,
                "post_margin": None,
                "legacy_confidence": None,
                "post_confidence": None,
                "queue_score": None,
                "queue_cutoff": None,
                "queue_rank": None,
                "selected": False,
                "legacy_queue_score": None,
                "legacy_queue_cutoff": None,
                "legacy_queue_rank": None,
                "legacy_selected": False,
            }
        )
        return record

    eos_candidate_columns = class_eos.nonzero(as_tuple=True)[0]
    if eos_candidate_columns.numel() > 0:
        candidate_scores = queue_scores[0, 0, eos_candidate_columns]
        candidate_col = int(
            eos_candidate_columns[candidate_scores.argmax()].item()
        )
    else:
        active_post_margin = post_margin[0, active_columns]
        candidate_col = int(
            active_columns[active_post_margin.argmax()].item()
        )

    candidate_flat_index = candidate_col

    def queue_metrics(
        candidate_scores: torch.Tensor, selected: torch.Tensor
    ) -> Dict[str, Any]:
        flat_scores = candidate_scores.flatten()
        value = flat_scores[candidate_flat_index]
        selected_values = flat_scores[selected]
        finite_scores = flat_scores[torch.isfinite(flat_scores)]
        return {
            "score": _finite_float(value),
            "cutoff": _finite_float(selected_values.min()),
            "rank": int((finite_scores > value).sum().item()) + 1,
            "selected": bool(
                (selected == candidate_flat_index).any().item()
            ),
        }

    actual_queue = queue_metrics(queue_scores, selected_flat_indices)
    legacy_queue = queue_metrics(
        legacy_queue_scores, legacy_topk_indices
    )
    post_column = post_log_scores[0, 0, candidate_col]
    legacy_column = legacy_log_scores[0, 0, candidate_col]
    record.update(
        {
            "candidate_col": candidate_col,
            "candidate_is_class_eos": bool(class_eos[candidate_col].item()),
            "guided_eos_mass": _finite_float(
                legal_cfg[0, 0, candidate_col, eos_id].exp()
            ),
            "conditional_eos_mass": _finite_float(
                c_log_probs[0, 0, candidate_col, eos_id].exp()
            ),
            "legacy_eos_mass": _finite_float(
                legacy_column[eos_id].exp()
            ),
            "post_eos_mass": _finite_float(post_column[eos_id].exp()),
            "legacy_total_mass": _finite_float(
                legacy_column.exp().sum()
            ),
            "post_total_mass": _finite_float(post_column.exp().sum()),
            "guided_margin": _finite_float(
                guided_margin[0, candidate_col]
            ),
            "legacy_margin": _finite_float(
                legacy_margin[0, candidate_col]
            ),
            "post_margin": _finite_float(post_margin[0, candidate_col]),
            "legacy_confidence": _finite_float(
                legacy_confidence[0, 0, candidate_col]
            ),
            "post_confidence": _finite_float(
                calibrated_confidence[0, 0, candidate_col]
            ),
            "queue_score": actual_queue["score"],
            "queue_cutoff": actual_queue["cutoff"],
            "queue_rank": actual_queue["rank"],
            "selected": actual_queue["selected"],
            "legacy_queue_score": legacy_queue["score"],
            "legacy_queue_cutoff": legacy_queue["cutoff"],
            "legacy_queue_rank": legacy_queue["rank"],
            "legacy_selected": legacy_queue["selected"],
        }
    )
    return record


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
    return model._compute_audio_logits(
        h,
        input_ids=ids.unsqueeze(0),
        audio_mask=amask,
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
    silence_run_frames: int = 0,
    silence_match_codebooks: int = 2,
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
        silence_run_frames=silence_run_frames,
        silence_match_codebooks=silence_match_codebooks,
    )
