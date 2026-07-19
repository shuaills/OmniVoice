#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training configuration dataclass.

Defines ``TrainingConfig``, a dataclass that holds all hyperparameters and paths
for training. Loaded from a JSON config file via ``TrainingConfig.from_json()``
in ``omnivoice.cli.train``.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TrainingConfig:
    # Key Paths
    output_dir: Optional[str] = None
    data_config: Optional[str] = None

    # Model Specific
    llm_name_or_path: str = "Qwen/Qwen3-0.6B"
    audio_vocab_size: int = 1025  # valid vocab size + 1 (mask token)
    audio_mask_id: int = 1024  # 1024 is the 1025-th token
    num_audio_codebook: int = 8

    # Model Training Specific
    audio_codebook_weights: List[float | int] = field(
        default_factory=lambda: [8, 8, 6, 6, 4, 4, 2, 2]
    )
    drop_cond_ratio: float = 0.1
    prompt_ratio_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.3))
    mask_ratio_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))
    language_ratio: float = 0.8
    use_pinyin_ratio: float = 0.3
    instruct_ratio: float = 1.0
    only_instruct_ratio: float = 0.5

    # Elastic canvas (omnivoice/elastic.py). When enabled the checkpoint must
    # already be migrated to audio_vocab_size = 1027
    # (scripts/migrate_elastic_ckpt.py).
    elastic: bool = False
    p_elastic: float = 0.5
    elastic_merge_prob: float = 0.08
    elastic_insert_prob: float = 0.04
    elastic_end_append_max_ratio: float = 0.25
    # E1.1 targeted corruption (elastic_mode="targeted"): per-sample canvas
    # length error ~ U(-delta_max, +delta_max); scheduler_mix couples half the
    # elastic samples to the low-mask regime (DreamOn dynamic-inverse analog).
    elastic_mode: str = "legacy"
    elastic_delta_max: float = 0.3
    elastic_mid_insert_frac: float = 0.3
    elastic_scheduler_mix: float = 0.5

    # Block-diffusion conversion (design/block-conversion-20260706).
    # block_training selects OmniVoiceBlockSampleProcessor: one current
    # block per sample, canvas truncated at its right edge, EOS fill on the
    # content tail. Checkpoint must be migrated to audio_vocab_size = 1026
    # (scripts/migrate_block_ckpt.py). Mutually exclusive with elastic.
    block_training: bool = False
    block_size: int = 32
    # DSpark-inspired revealed-neighbour acoustic correction.  Zero keeps the
    # historical architecture byte-for-byte; positive values are the low-rank
    # transition width.  This is experimental and only supported by B2 dual
    # block training.
    block_markov_rank: int = 0
    # DSpark-style proposal scan. The frozen parallel backbone proposes sparse
    # anchors; a tiny block-local state carries only earlier anchors forward.
    # Zero disables the structure. It is intentionally separate from the
    # legacy block_markov_* checkpoint contract.
    block_anchor_scan_dim: int = 0
    block_anchor_proposal_dim: int = 32
    block_anchor_stride: int = 8
    block_anchor_mode: str = "causal"
    # Mechanism-proof option: freeze every parameter except the anchor head.
    # This is fail-closed and only legal when block_anchor_scan_dim > 0.
    block_anchor_freeze_base: bool = False
    # "single" = B1 right-truncation (attention untouched);
    # "dual"   = B2 two-copy block-causal attention (flex_attention only).
    block_scheme: str = "single"
    # EOS/padding role decoupling (DESIGN_junction_eos_battle.md; Rainbow
    # Padding / VoidPadding isomorphism). When enabled, the [eos] label
    # shrinks to a single stop-event column at content end T, and the canvas
    # void T+1..T+1+silence_void_window is supervised as the real
    # digital-silence frame on ALL codebooks, so the posterior between
    # end-of-speech and canvas end prefers silence over residual junk
    # (tail-beep pathology, RESULTS 2026-07-10). Only meaningful with
    # block_scheme="dual".
    eos_decouple_silence: bool = False
    # Number of consecutive cb0 EOS targets beginning at the first frame after
    # content.  The band is one independently-normalized event; k=1 preserves
    # the original point-EOS recipe exactly.
    eos_band_k: int = 1
    silence_void_window: int = 32
    # Drop samples whose transcript has an utterance-edge disfluency filler
    # (zh/ja/ko leading, en leading+trailing). See omnivoice/data/edge_filler_filter.py.
    filter_edge_fillers: bool = False
    # Probability of snapping the training prompt cut to a sentence boundary
    # (needs 'turns' in labels; internal dataset has them). See blockdiff_dual.
    turn_boundary_prompt_prob: float = 0.0

    # Opt-in CFG conditioning contract for block_scheme="dual".  Disabled is
    # the exact historical 90/10 prompt-free drop_cond path.  Enabled replaces
    # it with explicit C/U_shared/U_drop_ref sample branches; q controls the
    # first target-only block width after physically dropping the reference.
    cfg_branch_training: bool = False
    cfg_branch_cond_ratio: float = 0.90
    cfg_branch_shared_ratio: float = 0.05
    cfg_branch_drop_ref_ratio: float = 0.05
    cfg_branch_seed: Optional[int] = None
    cfg_drop_ref_short_bucket_ratio: float = 0.5
    cfg_drop_ref_q_min: int = 1
    cfg_drop_ref_q_max: int = 32
    cfg_drop_ref_short_q_max: int = 4

    # Independently normalized acoustic/EOS/void objective.  Coefficients are
    # neutral defaults; production recipes must carry their measured values.
    split_loss: bool = False
    split_gamma: float = 1.0
    lambda_eos: float = 1.0
    lambda_void: float = 1.0

    # Init settings
    resume_from_checkpoint: Optional[str] = None
    # On resume, re-force the LR envelope from this config: checkpoint
    # optimizer.bin/scheduler.bin carry the original run's base_lrs, which
    # otherwise silently override a changed learning_rate (b2j lesson).
    force_lr_from_config_on_resume: bool = False
    init_from_checkpoint: Optional[str] = None

    # Training Hyperparams
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    steps: int = 300000
    seed: int = 42
    lr_scheduler_type: str = "cosine"
    warmup_type: str = "ratio"
    warmup_ratio: float = 0.03
    warmup_steps: int = 2000

    # Data
    batch_tokens: int = 8192
    gradient_accumulation_steps: int = 1
    num_workers: int = 8

    # Perf experiments (perf/step-time-20260708). Default OFF => inert.
    # NOTE: from_json silently drops unknown keys, so these MUST be real
    # fields -- otherwise an A/B arm would silently run as baseline.
    perf_blockmask_cache: bool = False
    perf_liger: bool = False
    perf_fused_adamw: bool = False
    perf_mask_buffers: bool = False
    perf_compile_skip_attn: bool = False
    perf_flex_bf16_qkv: bool = False
    # >0 enables cost-balanced packing with this window (in packs).
    perf_balanced_packing: int = 0
    perf_torch_compile: bool = False
    perf_compile_mode: str = "default"
    perf_compile_dynamic: bool = True
    # Enables HF gradient checkpointing on the LLM backbone (activation
    # recompute in backward). Capacity lever for long-form training.
    perf_grad_checkpoint: bool = False
    # Disables KV-cache construction in training forwards (use_cache=False
    # on the backbone). Training never consumes the cache; building it is
    # pure waste. Kept flag-gated for clean A/B attribution.
    perf_train_no_cache: bool = False

    # System
    mixed_precision: str = "bf16"
    allow_tf32: bool = True
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    attn_implementation: str = "flex_attention"

    # Length-grouped batching (only used when attn_implementation != "flex_attention")
    max_sample_tokens: int = 2000
    min_sample_tokens: int = 50
    max_batch_size: int = 64

    # Logging
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 10000
    keep_last_n_checkpoints: int = -1

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as f:
            cfg_dict = json.load(f)
        valid_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        instance = cls(**filtered_dict)
        return instance

    def save_to_json(self, json_path: str):
        data = asdict(self)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
