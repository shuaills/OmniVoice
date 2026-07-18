#!/usr/bin/env python3
"""Prove zero-output attachment on the real warm-start checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from omnivoice.training.builder import build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()

    config = TrainingConfig.from_json(args.control_config)
    if args.checkpoint is not None:
        config.init_from_checkpoint = str(args.checkpoint.resolve())
    if config.block_markov_rank != 0:
        raise SystemExit("attachment proof must start from rank-0 config")
    # This is a checkpoint/load/logit proof, not a training-throughput run.
    # Disable optional training patches so the proof has no dependency on
    # Liger, fused optimizers, compile state, or gradient checkpointing.
    config.perf_liger = False
    config.perf_fused_adamw = False
    config.perf_flex_bf16_qkv = False
    config.perf_torch_compile = False
    config.perf_grad_checkpoint = False
    config.perf_train_no_cache = False
    model, _ = build_model_and_tokenizer(config)
    model.eval()

    batch, frames = 1, 4
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260719)
        hidden = torch.randn(batch, frames, model.config.llm_config.hidden_size)
        input_ids = torch.randint(
            0,
            model.config.audio_mask_id,
            (batch, model.config.num_audio_codebook, frames),
        )
    audio_mask = torch.ones(batch, frames, dtype=torch.bool)
    prev_ids = torch.full_like(input_ids, model.config.audio_mask_id)
    prev_ids[:, :, 1:] = input_ids[:, :, :-1]
    with torch.inference_mode():
        baseline = model._compute_audio_logits(
            hidden,
            input_ids=input_ids,
            audio_mask=audio_mask,
            markov_prev_ids=prev_ids,
        )
        model.enable_block_markov_head(args.rank, seed=config.seed)
        attached = model._compute_audio_logits(
            hidden,
            input_ids=input_ids,
            audio_mask=audio_mask,
            markov_prev_ids=prev_ids,
        )
    if not torch.equal(baseline, attached):
        difference = (baseline - attached).abs().max().item()
        raise SystemExit(
            f"zero-output attachment changed real-checkpoint logits: {difference}"
        )
    head = model.block_markov_head
    expected_parameters = (
        2
        * model.config.num_audio_codebook
        * model.config.audio_vocab_size
        * args.rank
    )
    if head is None or head.parameter_count != expected_parameters:
        raise SystemExit(
            "unexpected rank-32 head parameter count: "
            f"{None if head is None else head.parameter_count}"
        )
    if torch.count_nonzero(head.output.weight).item() != 0:
        raise SystemExit("attached output projection is not exactly zero")
    embedding_hash = hashlib.sha256(
        head.prev_embeddings.weight.detach().contiguous().numpy().tobytes()
    ).hexdigest()
    print(
        "BLOCK_MARKOV_REAL_ATTACH_OK "
        f"rank={args.rank} params={head.parameter_count} "
        f"embedding_sha256={embedding_hash}"
    )


if __name__ == "__main__":
    main()
