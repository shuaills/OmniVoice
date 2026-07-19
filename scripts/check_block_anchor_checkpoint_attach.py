#!/usr/bin/env python3
"""Real-checkpoint attachment and synthetic mechanism diagnostics."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from omnivoice.blockdiff_dual import TAG_NOISY
from omnivoice.training.builder import build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-config", required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    config = TrainingConfig.from_json(args.causal_config)
    if args.checkpoint is not None:
        config.init_from_checkpoint = str(args.checkpoint.resolve())
    config.perf_liger = False
    config.perf_fused_adamw = False
    config.perf_flex_bf16_qkv = True
    config.perf_torch_compile = False
    config.perf_compile_dynamic = False
    config.perf_grad_checkpoint = False
    config.perf_train_no_cache = True
    model, _ = build_model_and_tokenizer(config)
    model.eval()
    head = model.block_anchor_scan_head
    if head is None:
        raise SystemExit("causal config did not attach an anchor head")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith("block_anchor_scan_head.") for name in trainable):
        raise SystemExit(f"optimizer whitelist is not head-only: {trainable[:20]}")

    batch, frames = 1, 32
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260719)
        hidden = torch.randn(batch, frames, model.config.llm_config.hidden_size)
        input_ids = torch.full(
            (batch, model.config.num_audio_codebook, frames),
            model.config.audio_mask_id,
        )
        audio_mask = torch.ones(batch, frames, dtype=torch.bool)
        positions = torch.arange(frames).view(1, 1, frames)
        boundaries = torch.full(
            (batch, 1, model.config.num_audio_codebook),
            model.config.audio_mask_id,
        )
    with torch.inference_mode():
        base = model._project_audio_logits(hidden)
        attached = head(base, input_ids, audio_mask, positions, boundaries)
    if not torch.equal(base, attached):
        raise SystemExit(
            "zero-output attachment changed real-checkpoint logits: "
            f"{(base - attached).abs().max().item()}"
        )
    if torch.count_nonzero(head.output.weight).item() != 0:
        raise SystemExit("attached output projection is not exactly zero")

    # Exercise the same full FlexAttention path used by the NLL proof before
    # spending time on any training arm.  A head-only tensor diagnostic cannot
    # catch Qwen3 q/k fp32 promotion against a bf16 value tensor.
    if not torch.cuda.is_available():
        raise SystemExit("real bf16 attachment preflight requires CUDA")
    device = torch.device("cuda:0")
    model.to(device)
    packed = {
        "input_ids": torch.full(
            (batch, model.config.num_audio_codebook, frames),
            model.config.audio_mask_id,
            dtype=torch.long,
            device=device,
        ),
        "audio_mask": torch.ones(
            batch, frames, dtype=torch.bool, device=device
        ),
        "document_ids": torch.zeros(
            batch, frames, dtype=torch.int32, device=device
        ),
        "position_ids": torch.arange(
            frames, dtype=torch.long, device=device
        ).view(1, -1),
        "copy_tags": torch.full(
            (batch, frames), TAG_NOISY, dtype=torch.int32, device=device
        ),
        "block_ids": torch.zeros(
            batch, frames, dtype=torch.int32, device=device
        ),
        "anchor_positions": positions.to(device),
        "anchor_boundary_ids": boundaries.to(device),
    }
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        full_on = model(**packed).logits
        model.block_anchor_scan_head = None
        try:
            full_off = model(**packed).logits
        finally:
            model.block_anchor_scan_head = head
    if full_on.dtype != torch.float32:
        raise SystemExit(
            f"full head-on forward must return fp32 logits, got {full_on.dtype}"
        )
    if not torch.equal(full_on, full_off.float()):
        raise SystemExit("zero-output head changed the full packed forward")
    if not torch.isfinite(full_on).all() or not torch.isfinite(full_off).all():
        raise SystemExit("full packed head on/off forward produced non-finite logits")
    del full_on, full_off, packed

    # Full-softmax proposals must be detached from the frozen backbone.
    probe_logits = torch.randn(1, 1, head.num_codebooks, head.mask_id, requires_grad=True)
    probe_ids = torch.full((1, 1, head.num_codebooks), head.mask_id)
    head._proposal_features(probe_ids, probe_logits).sum().backward()
    if probe_logits.grad is not None:
        raise SystemExit("proposal features leaked gradients into base logits")

    # Exercise the real checkpoint-attached module in its production bf16
    # input regime.  A fresh synthetic fp32 head would not catch the accidental
    # bf16 downcast that previously hid ~1e-3 partition drift.
    diagnostic = head.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        diagnostic.output.weight.normal_(0.0, 1e-3)
    raw = torch.randn(
        1,
        head.num_codebooks,
        frames,
        head.vocab_size,
        dtype=torch.bfloat16,
        device=device,
    )
    diagnostic_inputs = (
        input_ids.to(device),
        audio_mask.to(device),
        positions.to(device),
        boundaries.to(device),
    )
    causal = diagnostic(raw, *diagnostic_inputs, mode="causal")
    stateless = diagnostic(raw, *diagnostic_inputs, mode="stateless")
    if causal.dtype != torch.float32 or stateless.dtype != torch.float32:
        raise SystemExit(
            "bf16 correction must return fp32 logits, got "
            f"causal={causal.dtype} stateless={stateless.dtype}"
        )
    if torch.equal(causal, stateless):
        raise SystemExit("causal/stateless diagnostic modes are not distinguishable")
    if not torch.equal(
        causal[..., head.mask_id :], raw[..., head.mask_id :].float()
    ):
        raise SystemExit("diagnostic changed structural logits")
    partition_delta = (
        torch.logsumexp(causal[..., : head.mask_id].float(), dim=-1)
        - torch.logsumexp(raw[..., : head.mask_id].float(), dim=-1)
    ).abs().max().item()
    if partition_delta > 5e-6:
        raise SystemExit(f"diagnostic partition drift {partition_delta} > 5e-6")

    embedding_hash = hashlib.sha256(
        head.proposal_embeddings.weight.detach()
        .cpu()
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    ).hexdigest()
    print(
        "BLOCK_ANCHOR_REAL_ATTACH_OK "
        f"params={head.parameter_count} trainable_tensors={len(trainable)} "
        f"embedding_sha256={embedding_hash} correction_dtype={causal.dtype} "
        f"partition_delta={partition_delta:.6g} "
        "full_model_synthetic_packed_forward=PASS"
    )


if __name__ == "__main__":
    main()
