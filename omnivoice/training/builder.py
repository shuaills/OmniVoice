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

"""Builders for constructing training components.

Provides factory functions to assemble the model, tokenizer, and data loaders
from a ``TrainingConfig``. Called by ``omnivoice.cli.train`` to set up training.

Key functions:
- ``build_model_and_tokenizer()``: Loads the model and text tokenizer.
- ``build_dataloaders()``: Builds train/eval data loaders from a data config JSON.
  The batching strategy is chosen based on ``TrainingConfig.attn_implementation``:

  - ``"flex_attention"``: sequence packing via ``PackingIterableDataset`` +
    ``PackingDataCollator``. Batch shape is ``[1, C, batch_tokens]``.
  - other (e.g. ``"sdpa"``): length-grouped padding via
    ``StreamLengthGroupDataset`` + ``PaddingDataCollator``. Batch shape
    is ``[B, C, max_len]`` where B ≥ 1 and max_len ≤ batch_tokens.
"""

import logging
from functools import partial
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import logging as hf_logging
from transformers.trainer_utils import seed_worker

from omnivoice.data.batching import PackingIterableDataset, StreamLengthGroupDataset
from omnivoice.data.collator import PackingDataCollator, PaddingDataCollator
from omnivoice.data.dataset import WebDatasetReader, prepare_data_manifests_from_json
from omnivoice.data.processor import OmniVoiceSampleProcessor
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig, _resolve_model_path
from omnivoice.training.config import TrainingConfig

logger = logging.getLogger(__name__)


def build_model_and_tokenizer(
    config: TrainingConfig,
) -> Tuple[OmniVoice, AutoTokenizer]:
    """Load Tokenizer and Model, handle resizing and special tokens."""
    logger.info("Initializing Model & Tokenizer...")

    # 1. Tokenizer
    tokenizer_path = (
        config.init_from_checkpoint
        if config.init_from_checkpoint
        else config.llm_name_or_path
    )
    tokenizer_path = _resolve_model_path(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    new_tokens = [
        "<|denoise|>",
        "<|lang_start|>",
        "<|lang_end|>",
        "<|instruct_start|>",
        "<|instruct_end|>",
        "<|text_start|>",
        "<|text_end|>",
    ]

    tokens_to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    if config.init_from_checkpoint:
        logger.info(f"Loading weights from {config.init_from_checkpoint}")
        model = OmniVoice.from_pretrained(
            config.init_from_checkpoint,
            attn_implementation=config.attn_implementation,
            dtype=torch.float32,
            train=True,
        )
    else:
        resolved_llm = _resolve_model_path(config.llm_name_or_path)
        llm_config = AutoConfig.from_pretrained(resolved_llm)

        ov_config = OmniVoiceConfig(
            audio_vocab_size=config.audio_vocab_size,
            audio_mask_id=config.audio_mask_id,
            num_audio_codebook=config.num_audio_codebook,
            audio_codebook_weights=config.audio_codebook_weights,
            llm_config=llm_config,
        )

        original_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # suppress expected lm_head.weight warnings

        llm = AutoModel.from_pretrained(
            resolved_llm,
            attn_implementation=config.attn_implementation,
            dtype=torch.float32,
        )

        hf_logging.set_verbosity(original_level)
        model = OmniVoice(config=ov_config, llm=llm)

    # 3. Resize Embeddings
    if len(tokenizer) != model.config.llm_config.vocab_size:
        model.llm.resize_token_embeddings(len(tokenizer))

    # ---- perf experiment hooks (perf/step-time-20260708; default OFF) ----
    if config.perf_blockmask_cache:
        model._perf_blockmask_cache = True
        logger.info("PERF: BlockMask memoization enabled")
    if config.perf_liger:
        # Requires liger-kernel on PYTHONPATH (perf pylibs dir; NOT installed
        # into the donor venv). rope patches the transformers module globally;
        # rms_norm/swiglu rebind instance forwards.
        import transformers.models.qwen3.modeling_qwen3 as _mq3
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_qwen3(
            rope=True,
            rms_norm=True,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=False,  # tiny 1026 vocab: irrelevant
            model=model.llm,
        )
        _base = getattr(model.llm, model.llm.base_model_prefix, model.llm)
        _l0 = _base.layers[0]
        _took = (
            _mq3.apply_rotary_pos_emb.__module__.startswith("liger_kernel")
            and _l0.input_layernorm.forward.__func__.__module__.startswith(
                "liger_kernel"
            )
            and _l0.mlp.forward.__func__.__module__.startswith("liger_kernel")
        )
        if not _took:
            raise RuntimeError(
                "PERF: liger patch did not take (rope/rms/swiglu check "
                "failed) -- refusing to run a silently-baseline arm"
            )
        logger.info("PERF: liger kernels applied (rope+rms_norm+swiglu verified)")

    if config.perf_torch_compile:
        import torch as _torch
        logger.info(
            "PERF: torch.compile(model.llm, mode=%s, dynamic=%s)",
            config.perf_compile_mode,
            config.perf_compile_dynamic,
        )
        model.llm = _torch.compile(
            model.llm,
            mode=config.perf_compile_mode,
            dynamic=config.perf_compile_dynamic,
        )
        model.config.llm_config.vocab_size = len(tokenizer)

    # 4. Config IDs
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer


def build_dataloaders(
    config: TrainingConfig, tokenizer: AutoTokenizer
) -> Tuple[DataLoader, DataLoader]:
    """Setup Data Pipeline: Manifests -> WDS -> Batching -> Loaders.

    Batching strategy depends on ``config.attn_implementation``:
    - ``"flex_attention"``: sequence packing (PackingIterableDataset +
      PackingDataCollator). All samples are concatenated into one long sequence.
    - other (e.g. ``"sdpa"``): length-grouped padding
      (LengthGroupedIterableDataset + PaddingDataCollator). Samples with
      similar token lengths are batched together and padded to the same length.
    """
    logger.info("Initializing Data Readers...")

    processor_kwargs = dict(
        text_tokenizer=tokenizer,
        num_channels=config.num_audio_codebook,
        audio_mask_id=config.audio_mask_id,
        prompt_ratio_range=config.prompt_ratio_range,
        mask_ratio_range=config.mask_ratio_range,
        drop_cond_ratio=config.drop_cond_ratio,
        language_ratio=config.language_ratio,
        use_pinyin_ratio=config.use_pinyin_ratio,
        instruct_ratio=config.instruct_ratio,
        only_instruct_ratio=config.only_instruct_ratio,
    )
    if getattr(config, "block_training", False):
        if getattr(config, "elastic", False):
            raise ValueError("block_training and elastic are mutually exclusive")
        scheme = getattr(config, "block_scheme", "single")
        if scheme == "dual":
            if config.attn_implementation != "flex_attention":
                raise ValueError(
                    "block_scheme='dual' requires flex_attention (the "
                    "block-causal mask is built from packed metadata)"
                )
            from omnivoice.blockdiff_dual import OmniVoiceBlockDualSampleProcessor

            logger.info(
                "Block-diffusion DUAL (block-causal) training ENABLED "
                "(block_size=%s)", config.block_size,
            )
            processor = OmniVoiceBlockDualSampleProcessor(
                **processor_kwargs,
                block_size=config.block_size,
                turn_boundary_prompt_prob=getattr(
                    config, "turn_boundary_prompt_prob", 0.0
                ),
            )
        else:
            from omnivoice.blockdiff import OmniVoiceBlockSampleProcessor

            logger.info("Block-diffusion training ENABLED (block_size=%s)", config.block_size)
            processor = OmniVoiceBlockSampleProcessor(
                **processor_kwargs,
                block_size=config.block_size,
            )
    elif getattr(config, "elastic", False):
        from omnivoice.data.processor import OmniVoiceElasticSampleProcessor

        logger.info("Elastic canvas ENABLED (p_elastic=%s, mode=%s)", config.p_elastic, config.elastic_mode)
        processor = OmniVoiceElasticSampleProcessor(
            **processor_kwargs,
            p_elastic=config.p_elastic,
            elastic_merge_prob=config.elastic_merge_prob,
            elastic_insert_prob=config.elastic_insert_prob,
            elastic_end_append_max_ratio=config.elastic_end_append_max_ratio,
            elastic_mode=config.elastic_mode,
            elastic_delta_max=config.elastic_delta_max,
            elastic_mid_insert_frac=config.elastic_mid_insert_frac,
            elastic_scheduler_mix=config.elastic_scheduler_mix,
        )
    else:
        processor = OmniVoiceSampleProcessor(**processor_kwargs)

    train_manifests, dev_manifests = prepare_data_manifests_from_json(
        config.data_config
    )

    filter_edge = getattr(config, "filter_edge_fillers", False)
    if filter_edge:
        from omnivoice.data.edge_filler_filter import EdgeFillerFilterDataset
        logger.info("Edge-filler sample filter ENABLED (zh/ja/ko lead, en lead+trail)")
    raw_train_ds = WebDatasetReader(manifests=train_manifests, evaluation=False)
    if filter_edge:
        raw_train_ds = EdgeFillerFilterDataset(raw_train_ds)

    use_packing = config.attn_implementation == "flex_attention"

    if use_packing:
        train_dataset = PackingIterableDataset(
            raw_train_ds, processor, config.batch_tokens
        )
        collate_fn = PackingDataCollator(processor, config.batch_tokens)
    else:
        train_dataset = StreamLengthGroupDataset(
            raw_train_ds,
            batch_duration=config.batch_tokens,
            min_length=config.min_sample_tokens,
            max_length=config.max_sample_tokens,
            max_sample=config.max_batch_size,
            processor=processor,
            length_fn=lambda s: s["length"],
        )
        collate_fn = PaddingDataCollator(processor, config.batch_tokens)

    logger.info(
        "Using %s (attn_implementation=%s)",
        "sequence packing" if use_packing else "length-grouped padding",
        config.attn_implementation,
    )

    init_fn = partial(
        seed_worker,
        num_workers=config.num_workers,
        rank=(
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=init_fn,
        pin_memory=True,
        prefetch_factor=4,
    )

    eval_loader = None
    if dev_manifests:
        raw_dev_ds = WebDatasetReader(
            manifests=dev_manifests, evaluation=True
        )
        if filter_edge:
            raw_dev_ds = EdgeFillerFilterDataset(raw_dev_ds)
        if use_packing:
            dev_dataset = PackingIterableDataset(
                raw_dev_ds, processor, config.batch_tokens
            )
        else:
            dev_dataset = StreamLengthGroupDataset(
                raw_dev_ds,
                batch_duration=config.batch_tokens,
                min_length=config.min_sample_tokens,
                max_length=config.max_sample_tokens,
                max_sample=config.max_batch_size,
                processor=processor,
                length_fn=lambda s: s["length"],
            )
        eval_loader = DataLoader(
            dev_dataset,
            batch_size=None,  # Each item is already a collated batch
            num_workers=1,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=2,
        )

    return train_loader, eval_loader
