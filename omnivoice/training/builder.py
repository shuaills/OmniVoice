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

import copy
import json
import logging
import math
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Tuple

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


def _emit_perf_contract(message: str) -> None:
    """Make runtime performance axes visible even before logging is configured."""
    logger.info(message)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(message, flush=True)


_REQUIRED_TEXT_SPECIAL_TOKENS = (
    "<|denoise|>",
    "<|lang_start|>",
    "<|lang_end|>",
    "<|instruct_start|>",
    "<|instruct_end|>",
    "<|text_start|>",
    "<|text_end|>",
)


class TextVocabContractError(RuntimeError):
    """Raised when a checkpoint cannot preserve its text embedding weights."""


@dataclass(frozen=True)
class CheckpointTextVocabContract:
    """Read-only text-vocabulary facts collected from a saved checkpoint."""

    checkpoint_path: str
    tokenizer_size: int
    config_vocab_size: int
    embedding_rows: int
    lm_head_rows: int | None
    embedding_keys: tuple[str, ...]
    lm_head_keys: tuple[str, ...]

    @property
    def requires_config_shim(self) -> bool:
        """Whether loading needs an in-memory nested-config correction."""
        return self.config_vocab_size != self.embedding_rows


def _is_llm_tensor(key: str, suffix: str) -> bool:
    return (key.startswith("llm.") or ".llm." in key) and key.endswith(suffix)


def _checkpoint_safetensor_files(checkpoint_path: Path) -> list[Path]:
    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as f:
            index = json.load(f)
        filenames = sorted(set(index.get("weight_map", {}).values()))
        files = [checkpoint_path / filename for filename in filenames]
    else:
        files = sorted(checkpoint_path.glob("model*.safetensors"))

    if not files:
        raise TextVocabContractError(
            f"{checkpoint_path} has no model safetensors; cannot verify text "
            "embedding rows without loading the checkpoint"
        )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise TextVocabContractError(
            "checkpoint safetensor index references missing files: "
            + ", ".join(missing)
        )
    return files


def _checkpoint_text_tensor_shapes(
    checkpoint_path: Path,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    from safetensors import safe_open

    embeddings: dict[str, tuple[int, ...]] = {}
    lm_heads: dict[str, tuple[int, ...]] = {}
    for tensor_path in _checkpoint_safetensor_files(checkpoint_path):
        with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                target = None
                if _is_llm_tensor(key, "embed_tokens.weight"):
                    target = embeddings
                elif _is_llm_tensor(key, "lm_head.weight"):
                    target = lm_heads
                if target is not None:
                    if key in target:
                        raise TextVocabContractError(
                            "checkpoint repeats a text-vocab tensor key across "
                            f"safetensor files: {key}"
                        )
                    target[key] = tuple(tensors.get_slice(key).get_shape())
    return embeddings, lm_heads


def _single_row_count(
    tensors: dict[str, tuple[int, ...]], *, kind: str, required: bool
) -> int | None:
    if not tensors:
        if required:
            raise TextVocabContractError(
                f"checkpoint has no LLM {kind} tensor in its model safetensors"
            )
        return None
    malformed = {key: shape for key, shape in tensors.items() if len(shape) != 2}
    if malformed:
        raise TextVocabContractError(
            f"checkpoint has non-matrix LLM {kind} tensors: {malformed}"
        )
    rows = {shape[0] for shape in tensors.values()}
    if len(rows) != 1:
        raise TextVocabContractError(
            f"checkpoint LLM {kind} tensors disagree on row count: {tensors}"
        )
    return rows.pop()


def inspect_checkpoint_text_vocab_contract(
    checkpoint_path: str | os.PathLike[str],
    *,
    tokenizer: Any | None = None,
) -> CheckpointTextVocabContract:
    """Verify tokenizer/weight compatibility without mutating the checkpoint.

    A stale nested config is repairable because it controls allocation only. The
    tokenizer and saved embedding/head rows are not repairable here: changing
    either would resize or reinitialize trained weights, so this function fails.
    """
    resolved_path = Path(checkpoint_path).resolve()
    if not resolved_path.is_dir():
        raise TextVocabContractError(
            f"checkpoint path is not a local directory: {resolved_path}"
        )

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(resolved_path))
    missing_tokens = [
        token
        for token in _REQUIRED_TEXT_SPECIAL_TOKENS
        if token not in tokenizer.get_vocab()
    ]
    if missing_tokens:
        raise TextVocabContractError(
            "checkpoint tokenizer is missing required special tokens; refusing "
            f"to grow/reinitialize its embeddings: {missing_tokens}"
        )

    checkpoint_config = AutoConfig.from_pretrained(str(resolved_path))
    llm_config = getattr(checkpoint_config, "llm_config", None)
    if llm_config is None or getattr(llm_config, "vocab_size", None) is None:
        raise TextVocabContractError("checkpoint config has no llm_config.vocab_size")

    embeddings, lm_heads = _checkpoint_text_tensor_shapes(resolved_path)
    embedding_rows = _single_row_count(
        embeddings, kind="input embedding", required=True
    )
    assert embedding_rows is not None
    lm_head_rows = _single_row_count(lm_heads, kind="LM head", required=False)
    tokenizer_size = len(tokenizer)
    if tokenizer_size != embedding_rows:
        raise TextVocabContractError(
            "checkpoint tokenizer/embedding mismatch: "
            f"tokenizer={tokenizer_size}, embedding_rows={embedding_rows}; "
            "refusing to resize or reinitialize checkpoint weights"
        )
    if lm_head_rows is not None and lm_head_rows != embedding_rows:
        raise TextVocabContractError(
            "checkpoint embedding/LM-head mismatch: "
            f"embedding_rows={embedding_rows}, lm_head_rows={lm_head_rows}"
        )

    return CheckpointTextVocabContract(
        checkpoint_path=str(resolved_path),
        tokenizer_size=tokenizer_size,
        config_vocab_size=int(llm_config.vocab_size),
        embedding_rows=embedding_rows,
        lm_head_rows=lm_head_rows,
        embedding_keys=tuple(sorted(embeddings)),
        lm_head_keys=tuple(sorted(lm_heads)),
    )


def _checkpoint_config_shim(
    checkpoint_path: str,
    contract: CheckpointTextVocabContract,
):
    """Return a corrected config copy; never edit the source checkpoint."""
    checkpoint_config = copy.deepcopy(AutoConfig.from_pretrained(checkpoint_path))
    checkpoint_config.llm_config.vocab_size = contract.embedding_rows
    return checkpoint_config


def _model_text_vocab_rows(model: OmniVoice) -> tuple[int, int | None]:
    input_embeddings = model.llm.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        raise TextVocabContractError("loaded LLM has no input embedding weight")
    embedding_rows = int(input_embeddings.weight.shape[0])

    output_embeddings = model.llm.get_output_embeddings()
    lm_head_rows = None
    if output_embeddings is not None:
        if not hasattr(output_embeddings, "weight"):
            raise TextVocabContractError("loaded LLM head has no weight")
        lm_head_rows = int(output_embeddings.weight.shape[0])
    return embedding_rows, lm_head_rows


def assert_model_text_vocab_contract(model: OmniVoice, tokenizer: Any) -> None:
    """Require tokenizer, nested configs, embedding, and optional head to agree."""
    tokenizer_size = len(tokenizer)
    embedding_rows, lm_head_rows = _model_text_vocab_rows(model)
    outer_vocab_size = int(model.config.llm_config.vocab_size)
    inner_vocab_size = int(model.llm.config.vocab_size)
    observed = {
        "tokenizer": tokenizer_size,
        "outer_llm_config": outer_vocab_size,
        "inner_llm_config": inner_vocab_size,
        "embedding_rows": embedding_rows,
    }
    if lm_head_rows is not None:
        observed["lm_head_rows"] = lm_head_rows
    if any(value != tokenizer_size for value in observed.values()):
        raise TextVocabContractError(f"loaded text-vocab contract mismatch: {observed}")


def _assert_clean_checkpoint_loading_info(loading_info: Any) -> None:
    """Reject every HF load anomaly instead of accepting initialized weights."""

    if not isinstance(loading_info, dict):
        raise TextVocabContractError(
            "checkpoint loader did not return structured loading information"
        )
    required_fields = (
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    )
    absent_fields = [field for field in required_fields if field not in loading_info]
    if absent_fields:
        raise TextVocabContractError(
            "checkpoint loading information is incomplete: "
            + ", ".join(absent_fields)
        )
    anomalies = {
        field: loading_info[field]
        for field in required_fields
        if loading_info[field]
    }
    if anomalies:
        raise TextVocabContractError(
            "checkpoint did not load strictly; refusing initialized, ignored, "
            f"or mismatched weights: {anomalies}"
        )


def _synchronize_model_text_vocab_config(model: OmniVoice, tokenizer: Any) -> None:
    """Synchronize config metadata after a deliberate base-model resize."""
    tokenizer_size = len(tokenizer)
    embedding_rows, lm_head_rows = _model_text_vocab_rows(model)
    if embedding_rows != tokenizer_size or (
        lm_head_rows is not None and lm_head_rows != tokenizer_size
    ):
        raise TextVocabContractError(
            "cannot synchronize text-vocab config to incompatible weights: "
            f"tokenizer={tokenizer_size}, embedding_rows={embedding_rows}, "
            f"lm_head_rows={lm_head_rows}"
        )
    model.config.llm_config.vocab_size = tokenizer_size
    model.llm.config.vocab_size = tokenizer_size
    assert_model_text_vocab_contract(model, tokenizer)


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

    tokens_to_add = [
        token
        for token in _REQUIRED_TEXT_SPECIAL_TOKENS
        if token not in tokenizer.get_vocab()
    ]
    if tokens_to_add:
        if config.init_from_checkpoint:
            raise TextVocabContractError(
                "checkpoint tokenizer is missing required special tokens; "
                "refusing to grow/reinitialize its embeddings: "
                f"{tokens_to_add}"
            )
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    if config.init_from_checkpoint:
        contract = inspect_checkpoint_text_vocab_contract(
            tokenizer_path, tokenizer=tokenizer
        )
        load_kwargs = {}
        if contract.requires_config_shim:
            load_kwargs["config"] = _checkpoint_config_shim(
                tokenizer_path, contract
            )
            logger.warning(
                "Applying read-only checkpoint vocab config shim: "
                "config=%d, tokenizer/weights=%d",
                contract.config_vocab_size,
                contract.embedding_rows,
            )
        logger.info("Loading weights from %s", tokenizer_path)
        model, loading_info = OmniVoice.from_pretrained(
            tokenizer_path,
            attn_implementation=config.attn_implementation,
            dtype=torch.float32,
            train=True,
            output_loading_info=True,
            **load_kwargs,
        )
        _assert_clean_checkpoint_loading_info(loading_info)
        # A checkpoint load must preserve every trained embedding/head row.
        # Only the copied config above may be corrected; never resize here.
        assert_model_text_vocab_contract(model, tokenizer)
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

        # Resizing is allowed only for a fresh base-model initialization. The
        # tokenizer's required special tokens define the new training vocab.
        embedding_rows, _ = _model_text_vocab_rows(model)
        if len(tokenizer) != embedding_rows:
            model.llm.resize_token_embeddings(len(tokenizer))
        _synchronize_model_text_vocab_config(model, tokenizer)

    # ---- perf experiment hooks (perf/step-time-20260708; default OFF) ----
    if config.perf_blockmask_cache:
        model._perf_blockmask_cache = True
        logger.info("PERF: BlockMask memoization enabled")
    if config.perf_mask_buffers:
        model._perf_mask_buffers = True
        logger.info("PERF: persistent mask buffers enabled")

    if config.perf_train_no_cache:
        model.llm.config.use_cache = False
        logger.info("PERF: training-forward KV cache disabled")

    if config.perf_grad_checkpoint:
        model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("PERF: gradient checkpointing enabled (use_reentrant=False)")

    gradient_checkpointing_active = bool(
        getattr(model.llm, "is_gradient_checkpointing", False)
    )
    if gradient_checkpointing_active != config.perf_grad_checkpoint:
        raise RuntimeError(
            "gradient-checkpointing runtime/config mismatch: "
            f"configured={config.perf_grad_checkpoint} "
            f"active={gradient_checkpointing_active}"
        )
    _emit_perf_contract(
        "PERF: gradient checkpointing "
        f"active={gradient_checkpointing_active}"
    )

    if config.perf_flex_bf16_qkv:
        # ROOT-CAUSE FIX (16x anomaly): under accelerate bf16 mixed precision
        # with fp32 master weights, transformers-5.3 Qwen3 feeds flex
        # attention FP32 q/k/v (RMSNorm weight-dtype promotion + fp32 rope
        # constants). fp32 flex backward at head_dim=128 runs ~7x slower.
        # Cast q/k/v to bf16 at the attention boundary; autograd casts the
        # grads back to fp32 automatically. Output stays bf16 (o_proj input
        # under autocast would be bf16 anyway).
        import transformers.integrations.flex_attention as _fa2
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _AAF2
        import torch as _torch2

        _orig_flex = _fa2.flex_attention_forward

        def _bf16_flex(module, query, key, value, attention_mask, *a2, **kw2):
            return _orig_flex(
                module,
                query.to(_torch2.bfloat16),
                key.to(_torch2.bfloat16),
                value.to(_torch2.bfloat16),
                attention_mask,
                *a2,
                **kw2,
            )

        _fa2.flex_attention_forward = _bf16_flex
        try:
            _AAF2["flex_attention"] = _bf16_flex
        except Exception:
            _AAF2.register("flex_attention", _bf16_flex)
        logger.info("PERF: flex q/k/v bf16 cast enabled (fp32-attention fix)")

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

        if config.perf_compile_skip_attn:
            # Graph-break around attention: flex runs eager (HF singleton),
            # inductor compiles only the glue. Bisects whether inductor's
            # flex lowering is the parity-corrupting op.
            import transformers.integrations.flex_attention as _fa
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _AAF

            _disabled = _torch._dynamo.disable(_fa.flex_attention_forward)
            _fa.flex_attention_forward = _disabled
            try:
                _AAF["flex_attention"] = _disabled
            except Exception:
                _AAF.register("flex_attention", _disabled)
            logger.info("PERF: attention excluded from torch.compile scope")
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

    if config.split_loss:
        if not config.block_training or config.block_scheme != "dual":
            raise ValueError(
                "split_loss requires block_training with block_scheme='dual'"
            )
        if not config.eos_decouple_silence:
            raise ValueError("split_loss requires eos_decouple_silence=True")
        if config.eos_band_k < 1:
            raise ValueError("eos_band_k must be >= 1")
        if config.attn_implementation != "flex_attention":
            raise ValueError("split_loss requires flex_attention packing")
        if config.elastic:
            raise ValueError("split_loss is undefined with elastic loss_weights")
        coefficients = {
            "split_gamma": config.split_gamma,
            "lambda_eos": config.lambda_eos,
            "lambda_void": config.lambda_void,
        }
        if not math.isfinite(config.split_gamma) or config.split_gamma <= 0:
            raise ValueError("split_gamma must be finite and > 0")
        if any(
            not math.isfinite(value) or value < 0
            for name, value in coefficients.items()
            if name != "split_gamma"
        ):
            raise ValueError("split-loss lambdas must be finite and >= 0")
        model._split_loss = True
        model._eos_band_k = config.eos_band_k
        logger.info(
            "Split loss enabled (gamma=%s, lambda_eos=%s, lambda_void=%s)",
            config.split_gamma,
            config.lambda_eos,
            config.lambda_void,
        )
    else:
        model._split_loss = False

    # 4. Config IDs
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    assert_model_text_vocab_contract(model, tokenizer)

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
                "(block_size=%s, eos_band_k=%s, cfg_branch_training=%s)",
                config.block_size,
                config.eos_band_k,
                config.cfg_branch_training,
            )
            processor = OmniVoiceBlockDualSampleProcessor(
                **processor_kwargs,
                block_size=config.block_size,
                turn_boundary_prompt_prob=getattr(
                    config, "turn_boundary_prompt_prob", 0.0
                ),
                eos_decouple_silence=getattr(
                    config, "eos_decouple_silence", False
                ),
                eos_band_k=getattr(config, "eos_band_k", 1),
                silence_void_window=getattr(config, "silence_void_window", 32),
                cfg_branch_training=getattr(config, "cfg_branch_training", False),
                cfg_branch_cond_ratio=getattr(
                    config, "cfg_branch_cond_ratio", 0.90
                ),
                cfg_branch_shared_ratio=getattr(
                    config, "cfg_branch_shared_ratio", 0.05
                ),
                cfg_branch_drop_ref_ratio=getattr(
                    config, "cfg_branch_drop_ref_ratio", 0.05
                ),
                cfg_branch_seed=(
                    config.seed
                    if getattr(config, "cfg_branch_seed", None) is None
                    else config.cfg_branch_seed
                ),
                cfg_drop_ref_short_bucket_ratio=getattr(
                    config, "cfg_drop_ref_short_bucket_ratio", 0.5
                ),
                cfg_drop_ref_q_min=getattr(config, "cfg_drop_ref_q_min", 1),
                cfg_drop_ref_q_max=getattr(config, "cfg_drop_ref_q_max", 32),
                cfg_drop_ref_short_q_max=getattr(
                    config, "cfg_drop_ref_short_q_max", 4
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
        balanced_window = getattr(config, "perf_balanced_packing", 0)
        _emit_perf_contract(f"PERF: balanced packing window={balanced_window}")
        train_dataset = PackingIterableDataset(
            raw_train_ds,
            processor,
            config.batch_tokens,
            balanced_window=balanced_window,
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
