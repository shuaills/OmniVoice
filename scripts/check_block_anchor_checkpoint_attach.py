#!/usr/bin/env python3
"""Real-checkpoint attachment and synthetic mechanism diagnostics."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from omnivoice.blockdiff import block_eos_id
from omnivoice.blockdiff_dual import (
    KIND_ACOUSTIC,
    KIND_EOS,
    KIND_IGNORE,
    KIND_VOID,
    SILENCE_FRAME_TOKENS,
    TAG_NOISY,
)
from omnivoice.training.builder import build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig
from omnivoice.training.split_loss import category_counts


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
    if not getattr(model, "_split_loss", False):
        raise SystemExit("causal config did not enable the split-loss contract")
    if getattr(model, "_eos_band_k", None) != config.eos_band_k:
        raise SystemExit("model/config EOS-band contract mismatch")
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
    targets = (
        torch.arange(
            model.config.num_audio_codebook * frames,
            dtype=torch.long,
            device=device,
        )
        .view(1, model.config.num_audio_codebook, frames)
        .remainder(model.config.audio_mask_id)
    )
    synthetic_input_ids = torch.full_like(targets, model.config.audio_mask_id)
    synthetic_input_ids[:, :, 0] = targets[:, :, 0]
    labels = torch.full_like(targets, -100)
    loss_kind = torch.full_like(labels, KIND_IGNORE, dtype=torch.uint8)
    eos_start = frames - config.eos_band_k - 1
    if eos_start <= 1:
        raise SystemExit("synthetic split-loss batch has no acoustic supervision")
    void_start = eos_start + config.eos_band_k
    labels[:, :, 1:eos_start] = targets[:, :, 1:eos_start]
    loss_kind[:, :, 1:eos_start] = KIND_ACOUSTIC
    labels[:, 0, eos_start:void_start] = block_eos_id(
        model.config.audio_mask_id
    )
    loss_kind[:, 0, eos_start:void_start] = KIND_EOS
    labels[:, :, void_start:] = SILENCE_FRAME_TOKENS[
        : model.config.num_audio_codebook
    ].to(device).view(1, -1, 1)
    loss_kind[:, :, void_start:] = KIND_VOID
    if not torch.equal(loss_kind.eq(KIND_IGNORE), labels.eq(-100)):
        raise SystemExit("synthetic split-loss labels violate IGNORE <-> -100")
    valid_labels = labels.ne(-100)
    if (
        labels[valid_labels].min().item() < 0
        or labels[valid_labels].max().item() >= model.config.audio_vocab_size
    ):
        raise SystemExit("synthetic split-loss target escaped the audio vocabulary")
    document_ids = torch.zeros(batch, frames, dtype=torch.int32, device=device)
    counts = category_counts(loss_kind, document_ids)
    expected_audio_count = torch.full(
        (model.config.num_audio_codebook,),
        eos_start - 1,
        dtype=torch.int64,
        device=device,
    )
    expected_void_count = torch.ones(
        model.config.num_audio_codebook,
        dtype=torch.int64,
        device=device,
    )
    if (
        counts.invariant_errors.item() != 0
        or counts.eos_count.item() != 1
        or counts.void_events.item() != 1
        or not torch.equal(counts.audio_count, expected_audio_count)
        or not torch.equal(counts.void_count, expected_void_count)
        or counts.void_displaced.item()
        != (config.eos_band_k - 1) * model.config.num_audio_codebook
    ):
        raise SystemExit(
            "synthetic split-loss batch violates training coordinator gates: "
            f"invariant_errors={counts.invariant_errors.item()} "
            f"eos_count={counts.eos_count.item()} "
            f"void_events={counts.void_events.item()} "
            f"audio_count={counts.audio_count} "
            f"void_count={counts.void_count} "
            f"void_displaced={counts.void_displaced.item()}"
        )
    packed = {
        "input_ids": synthetic_input_ids,
        "audio_mask": torch.ones(
            batch, frames, dtype=torch.bool, device=device
        ),
        "document_ids": document_ids,
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
        "labels": labels,
        "loss_kind": loss_kind,
    }
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        full_on_output = model(**packed)
        model.block_anchor_scan_head = None
        try:
            full_off_output = model(**packed)
        finally:
            model.block_anchor_scan_head = head
    full_on = full_on_output.logits
    full_off = full_off_output.logits
    if full_on.dtype != torch.float32:
        raise SystemExit(
            f"full head-on forward must return fp32 logits, got {full_on.dtype}"
        )
    if not torch.equal(full_on, full_off.float()):
        raise SystemExit("zero-output head changed the full packed forward")
    if not torch.isfinite(full_on).all() or not torch.isfinite(full_off).all():
        raise SystemExit("full packed head on/off forward produced non-finite logits")
    for name, output in (("head_on", full_on_output), ("head_off", full_off_output)):
        if output.legacy_loss is None or not torch.isfinite(output.legacy_loss):
            raise SystemExit(f"{name} split-loss legacy loss is missing or non-finite")
        if output.audio_sum is None or not torch.isfinite(output.audio_sum).all():
            raise SystemExit(f"{name} split-loss acoustic numerator is invalid")
        if output.eos_sum is None or not torch.isfinite(output.eos_sum):
            raise SystemExit(f"{name} split-loss EOS numerator is invalid")
        if output.void_event_sum is None or not torch.isfinite(output.void_event_sum):
            raise SystemExit(f"{name} split-loss VOID numerator is invalid")
        if not torch.equal(output.audio_count, expected_audio_count):
            raise SystemExit(
                f"{name} split-loss acoustic count mismatch: "
                f"{output.audio_count} != {expected_audio_count}"
            )
        if output.eos_count.item() != 1 or output.void_events.item() != 1:
            raise SystemExit(f"{name} synthetic split-loss EOS/VOID count mismatch")
        if not torch.equal(
            output.void_count,
            expected_void_count,
        ):
            raise SystemExit(f"{name} synthetic split-loss VOID cells mismatch")
    del full_on, full_off, full_on_output, full_off_output, packed

    # Full-softmax proposals must be detached from the frozen backbone.
    probe_logits = torch.randn(
        1,
        1,
        head.num_codebooks,
        head.mask_id,
        device=device,
        requires_grad=True,
    )
    probe_ids = torch.full(
        (1, 1, head.num_codebooks),
        head.mask_id,
        dtype=torch.long,
        device=device,
    )
    head._proposal_features(probe_ids, probe_logits).sum().backward()
    if probe_logits.grad is not None:
        raise SystemExit("proposal features leaked gradients into base logits")
    head.zero_grad(set_to_none=True)

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
        "full_model_synthetic_packed_forward=PASS "
        "split_loss_training_contract=PASS"
    )


if __name__ == "__main__":
    main()
