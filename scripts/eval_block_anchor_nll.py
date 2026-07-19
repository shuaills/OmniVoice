#!/usr/bin/env python3
"""Fixed-pack seed-0 NLL proof for causal versus stateless anchor scans."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from transformers import AutoTokenizer

from omnivoice.blockdiff_dual import (
    KIND_ACOUSTIC,
    KIND_EOS,
    KIND_VOID,
    TAG_CLEAN,
    TAG_NOISY,
)
from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig


GROUPS = (
    "overall",
    "suffix",
    "soft_only_suffix",
    "offset_1",
    "offset_2",
    "offset_3",
    "offset_4",
    "offset_5",
    "offset_6",
    "offset_7",
    "eos",
    "void",
)
FORWARD_KEYS = {
    "input_ids",
    "audio_mask",
    "attention_mask",
    "document_ids",
    "position_ids",
    "copy_tags",
    "block_ids",
    "anchor_positions",
    "anchor_boundary_ids",
    "labels",
    "loss_kind",
}
BENCHMARK_FORWARD_KEYS = FORWARD_KEYS - {"labels", "loss_kind"}
PARTITION_MAX_ABS_DELTA = 5e-6
TRAINED_STATELESS_NONINFERIORITY_MARGIN_PCT = 0.05
MIN_HEAD_ON_THROUGHPUT_RATIO = 0.95
MAX_HEAD_ON_MEMORY_DELTA_MIB = 1024.0


def _config(path: Path, checkpoint: Path, *, enabled: bool) -> TrainingConfig:
    config = TrainingConfig.from_json(path)
    config.init_from_checkpoint = str(checkpoint.resolve())
    config.resume_from_checkpoint = None
    if not enabled:
        config.block_anchor_scan_dim = 0
        config.block_anchor_freeze_base = False
    config.num_workers = 1
    config.batch_tokens = 4096
    config.perf_grad_checkpoint = False
    config.perf_liger = False
    config.perf_fused_adamw = False
    # This is a correctness requirement for the fp32-master Qwen3 checkpoint,
    # not merely a training-speed toggle.  Qwen3 RMSNorm promotes q/k to fp32
    # while value remains bf16; FlexAttention rejects that mixed triplet.  The
    # production training contract therefore casts q/k/v together at the
    # attention boundary, and evaluation must preserve the same forward path.
    config.perf_flex_bf16_qkv = True
    config.perf_torch_compile = False
    config.perf_compile_dynamic = False
    config.perf_train_no_cache = True
    return config


def _snapshot_batches(
    config_path: Path, checkpoint: Path, num_packs: int
) -> tuple[list[dict[str, torch.Tensor]], str]:
    config = _config(config_path, checkpoint, enabled=True)
    set_seed(20260719)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    _, dev_loader = build_dataloaders(config, tokenizer)
    if dev_loader is None:
        raise RuntimeError("data config has no dev manifests")
    batches: list[dict[str, torch.Tensor]] = []
    digest = hashlib.sha256()
    iterator = iter(dev_loader)
    required = {
        "input_ids",
        "audio_mask",
        "labels",
        "document_ids",
        "copy_tags",
        "block_ids",
        "loss_kind",
        "anchor_positions",
        "anchor_boundary_ids",
    }
    for pack_index in range(num_packs):
        batch = next(iterator)
        frozen = {
            name: value.detach().cpu().clone()
            for name, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        missing = sorted(required - set(frozen))
        if missing:
            raise RuntimeError(f"dev pack is missing anchor tensors: {missing}")
        for name in sorted(frozen):
            tensor = frozen[name].contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        _validate_layout(frozen, mask_id=config.audio_mask_id)
        batches.append(frozen)
        print(f"BLOCK_ANCHOR_DEV_PACK_READY index={pack_index}", flush=True)
    del iterator, dev_loader, tokenizer
    gc.collect()
    return batches, digest.hexdigest()


def _validate_layout(batch: dict[str, torch.Tensor], *, mask_id: int) -> None:
    positions = batch["anchor_positions"]
    boundaries = batch["anchor_boundary_ids"]
    if positions.ndim != 3 or boundaries.ndim != 3:
        raise RuntimeError("packed anchor layout must have rank three")
    if positions.shape[:2] != boundaries.shape[:2]:
        raise RuntimeError("anchor positions/boundaries disagree on block count")
    sequence_length = batch["input_ids"].size(-1)
    for batch_index in range(positions.size(0)):
        noisy_groups: dict[tuple[int, int], list[int]] = {}
        noisy_indices = torch.nonzero(
            batch["copy_tags"][batch_index].eq(TAG_NOISY),
            as_tuple=False,
        ).flatten()
        for position in noisy_indices.tolist():
            document = int(batch["document_ids"][batch_index, position])
            block = int(batch["block_ids"][batch_index, position])
            if document < 0 or block < 0:
                raise RuntimeError(
                    "noisy anchor source has a negative document/block id"
                )
            noisy_groups.setdefault((document, block), []).append(position)

        seen_groups: set[tuple[int, int]] = set()
        for block_index in range(positions.size(1)):
            row = positions[batch_index, block_index]
            valid = row[row.ge(0)]
            if valid.numel() == 0:
                raise RuntimeError("anchor layout contains an empty block row")
            if valid.numel() > 32 or int(valid[-1]) >= sequence_length:
                raise RuntimeError("anchor layout points outside a 32-frame block")
            if not batch["copy_tags"][batch_index, valid].eq(TAG_NOISY).all():
                raise RuntimeError("anchor layout points outside the noisy copy")
            documents = batch["document_ids"][batch_index, valid].unique()
            blocks = batch["block_ids"][batch_index, valid].unique()
            if documents.numel() != 1 or blocks.numel() != 1:
                raise RuntimeError("anchor layout crosses a packed document/block")
            group = (int(documents.item()), int(blocks.item()))
            if group in seen_groups:
                raise RuntimeError(f"anchor layout repeats noisy group {group}")
            if group not in noisy_groups:
                raise RuntimeError(f"anchor layout invents noisy group {group}")
            expected_positions = torch.tensor(
                noisy_groups[group], dtype=row.dtype, device=row.device
            )
            expected_row = torch.full_like(row, -1)
            expected_row[: expected_positions.numel()] = expected_positions
            if not torch.equal(row, expected_row):
                raise RuntimeError(
                    "anchor layout is not exact padded coverage for noisy group "
                    f"{group}: expected={expected_row.tolist()} actual={row.tolist()}"
                )

            document, block = group
            if block == 0:
                expected_boundary = torch.full_like(
                    boundaries[batch_index, block_index], mask_id
                )
            else:
                clean_positions = torch.nonzero(
                    batch["document_ids"][batch_index].eq(document)
                    & batch["copy_tags"][batch_index].eq(TAG_CLEAN)
                    & batch["block_ids"][batch_index].eq(block - 1),
                    as_tuple=False,
                ).flatten()
                if clean_positions.numel() == 0:
                    raise RuntimeError(
                        "anchor boundary has no preceding committed CLEAN frame "
                        f"for noisy group {group}"
                    )
                expected_boundary = batch["input_ids"][
                    batch_index, :, clean_positions[-1]
                ]
            if not torch.equal(
                boundaries[batch_index, block_index], expected_boundary
            ):
                raise RuntimeError(
                    "anchor boundary is not the immediately preceding committed "
                    f"CLEAN frame for noisy group {group}"
                )
            seen_groups.add(group)
        if seen_groups != set(noisy_groups):
            missing = sorted(set(noisy_groups) - seen_groups)
            extra = sorted(seen_groups - set(noisy_groups))
            raise RuntimeError(
                "anchor layout does not cover noisy groups exactly: "
                f"missing={missing} extra={extra}"
            )
    if boundaries.lt(0).any() or boundaries.gt(mask_id).any():
        raise RuntimeError("anchor boundary contains a non-acoustic/non-mask id")


def _local_layout(
    batch: dict[str, torch.Tensor], *, mask_id: int, stride: int
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = batch["anchor_positions"]
    sequence_length = batch["input_ids"].size(-1)
    local = torch.full(
        (positions.size(0), sequence_length), -1, dtype=torch.long
    )
    soft_suffix = torch.zeros_like(local, dtype=torch.bool)
    for batch_index in range(positions.size(0)):
        for block_index in range(positions.size(1)):
            valid = positions[batch_index, block_index]
            valid = valid[valid.ge(0)]
            local[batch_index, valid] = torch.arange(valid.numel())
            for offset in range(1, valid.numel()):
                anchor_offset = (offset // stride) * stride
                anchor_position = valid[anchor_offset]
                masked = batch["input_ids"][batch_index, :, anchor_position].eq(
                    mask_id
                ).all()
                soft_suffix[batch_index, valid[offset]] = bool(masked)
    return local, soft_suffix


def _masks(
    batch: dict[str, torch.Tensor], *, mask_id: int, stride: int
) -> dict[str, torch.Tensor]:
    acoustic = batch["loss_kind"].eq(KIND_ACOUSTIC)
    local, soft_suffix = _local_layout(batch, mask_id=mask_id, stride=stride)
    offset = torch.remainder(local, stride)
    suffix_frame = local.ge(0) & offset.ne(0)
    result = {
        "overall": acoustic,
        "suffix": acoustic & suffix_frame.unsqueeze(1),
        "soft_only_suffix": acoustic & soft_suffix.unsqueeze(1),
        "eos": batch["loss_kind"].eq(KIND_EOS),
        "void": batch["loss_kind"].eq(KIND_VOID),
    }
    for value in range(1, stride):
        result[f"offset_{value}"] = acoustic & offset.eq(value).unsqueeze(1)
    return result


class Metrics:
    def __init__(self, weights: list[float]) -> None:
        raw = torch.tensor(weights, dtype=torch.float64)
        self.weights = raw / raw.sum()
        self.sums = {
            group: torch.zeros(len(raw), dtype=torch.float64) for group in GROUPS
        }
        self.counts = {
            group: torch.zeros(len(raw), dtype=torch.int64) for group in GROUPS
        }
        self.pack_values: dict[str, list[float | None]] = {
            group: [] for group in GROUPS
        }

    def add(self, nll: torch.Tensor, masks: dict[str, torch.Tensor]) -> None:
        for group, mask in masks.items():
            means: list[torch.Tensor | None] = []
            for codebook in range(nll.size(1)):
                selected = nll[:, codebook][mask[:, codebook]]
                if selected.numel() == 0:
                    means.append(None)
                    continue
                self.sums[group][codebook] += selected.double().sum()
                self.counts[group][codebook] += selected.numel()
                means.append(selected.double().mean())
            active = [index for index, value in enumerate(means) if value is not None]
            if not active:
                self.pack_values[group].append(None)
                continue
            weights = self.weights[active]
            weights = weights / weights.sum()
            self.pack_values[group].append(
                float(
                    sum(
                        means[index] * weights[position]
                        for position, index in enumerate(active)
                    )
                )
            )

    def summary(self) -> dict:
        result: dict[str, dict] = {}
        for group in GROUPS:
            active = self.counts[group].gt(0)
            if not active.any():
                raise RuntimeError(f"zero aggregate count for group {group}")
            per_codebook = torch.full_like(self.sums[group], float("nan"))
            per_codebook[active] = self.sums[group][active] / self.counts[group][active]
            weights = self.weights[active]
            weights = weights / weights.sum()
            result[group] = {
                "weighted_nll": float((per_codebook[active] * weights).sum()),
                "codebook_nll": [
                    float(value) if torch.isfinite(value) else None
                    for value in per_codebook
                ],
                "counts": [int(value) for value in self.counts[group]],
                "pack_weighted_nll": self.pack_values[group],
            }
        return result


def _forward(model, batch, device) -> torch.Tensor:
    inputs = {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if name in FORWARD_KEYS
    }
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        return model(**inputs).logits


def _forward_device(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    inputs = {
        name: value for name, value in batch.items() if name in BENCHMARK_FORWARD_KEYS
    }
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        return model(**inputs).logits


def _benchmark_head_on_off(
    config_path: Path,
    checkpoint: Path,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    packs: int,
    warmup: int,
    repeats: int,
) -> dict:
    if packs <= 0 or packs > len(batches):
        raise RuntimeError(f"benchmark packs must be in [1,{len(batches)}], got {packs}")
    if warmup < 1 or repeats < 3:
        raise RuntimeError("benchmark requires warmup >= 1 and repeats >= 3")
    model = _load_model(config_path, checkpoint, enabled=True, device=device)
    head = model.block_anchor_scan_head
    if head is None:
        raise RuntimeError("benchmark checkpoint loaded without an anchor head")
    fixed = [
        {
            name: value.to(device, non_blocking=False)
            for name, value in batch.items()
            if name in BENCHMARK_FORWARD_KEYS
        }
        for batch in batches[:packs]
    ]

    def set_enabled(enabled: bool) -> None:
        model.block_anchor_scan_head = head if enabled else None

    def sweep(enabled: bool) -> float:
        set_enabled(enabled)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for batch in fixed:
            logits = _forward_device(model, batch)
            del logits
        torch.cuda.synchronize(device)
        return time.perf_counter() - started

    for index in range(warmup):
        order = (False, True) if index % 2 == 0 else (True, False)
        for enabled in order:
            sweep(enabled)
    off_times: list[float] = []
    on_times: list[float] = []
    for index in range(repeats):
        order = (False, True) if index % 2 == 0 else (True, False)
        measured = {enabled: sweep(enabled) for enabled in order}
        off_times.append(measured[False])
        on_times.append(measured[True])

    def peak(enabled: bool) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        sweep(enabled)
        return float(torch.cuda.max_memory_allocated(device) / (1 << 20))

    off_peak_mib = peak(False)
    on_peak_mib = peak(True)
    off_median = statistics.median(off_times)
    on_median = statistics.median(on_times)
    throughput_ratio = off_median / on_median
    memory_delta_mib = max(0.0, on_peak_mib - off_peak_mib)
    memory_limit_mib = min(
        MAX_HEAD_ON_MEMORY_DELTA_MIB,
        0.08 * off_peak_mib,
    )
    set_enabled(True)
    del model, head, fixed
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "fixed_pack_count": packs,
        "warmup_sweeps_per_mode": warmup,
        "measured_sweeps_per_mode": repeats,
        "measurement_order": "alternating_head_off_first/head_on_first",
        "head_off_seconds": off_times,
        "head_on_seconds": on_times,
        "head_off_median_seconds": off_median,
        "head_on_median_seconds": on_median,
        "head_on_throughput_ratio": throughput_ratio,
        "head_off_peak_mib": off_peak_mib,
        "head_on_peak_mib": on_peak_mib,
        "head_on_peak_delta_mib": memory_delta_mib,
        "head_on_peak_delta_limit_mib": memory_limit_mib,
        "throughput_ratio_gate": MIN_HEAD_ON_THROUGHPUT_RATIO,
        "memory_gate": "delta <= min(1024 MiB, 8% of head-off peak)",
    }


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().permute(0, 3, 1, 2),
        labels.to(logits.device),
        reduction="none",
        ignore_index=-100,
    ).cpu()


def _logits_hash(logits: torch.Tensor) -> str:
    raw = logits.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(config_path: Path, checkpoint: Path, *, enabled: bool, device):
    config = _config(config_path, checkpoint, enabled=enabled)
    model, _ = build_model_and_tokenizer(config)
    model._split_loss = False
    model.eval().to(device)
    return model


def _evaluate_base(config_path, checkpoint, batches, device, stride):
    model = _load_model(config_path, checkpoint, enabled=False, device=device)
    metrics = Metrics(model.config.audio_codebook_weights)
    hashes = []
    for batch in batches:
        logits = _forward(model, batch, device)
        hashes.append(_logits_hash(logits))
        metrics.add(
            _nll(logits, batch["labels"]),
            _masks(batch, mask_id=model.config.audio_mask_id, stride=stride),
        )
        del logits
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics.summary(), hashes


def _evaluate_pair(
    config_path,
    checkpoint,
    batches,
    device,
    stride,
    expected_base_hashes,
    *,
    diagnostic_stateless=False,
):
    model = _load_model(config_path, checkpoint, enabled=True, device=device)
    head = model.block_anchor_scan_head
    if head is None:
        raise RuntimeError("anchor checkpoint loaded without its head")
    on_metrics = Metrics(model.config.audio_codebook_weights)
    off_metrics = Metrics(model.config.audio_codebook_weights)
    diagnostic_metrics = (
        Metrics(model.config.audio_codebook_weights)
        if diagnostic_stateless
        else None
    )
    structural_exact = True
    partition_delta = 0.0
    off_base_exact = True
    label_independent = True
    for pack_index, batch in enumerate(batches):
        logits_on = _forward(model, batch, device)
        logits_diagnostic = None
        if diagnostic_metrics is not None:
            original_mode = head.mode
            head.mode = "stateless"
            logits_diagnostic = _forward(model, batch, device)
            head.mode = original_mode
        model.block_anchor_scan_head = None
        logits_off = _forward(model, batch, device)
        model.block_anchor_scan_head = head

        structural_exact &= torch.equal(
            logits_on[..., model.config.audio_mask_id :],
            logits_off[..., model.config.audio_mask_id :],
        )
        delta = torch.logsumexp(
            logits_on.float()[..., : model.config.audio_mask_id], dim=-1
        ) - torch.logsumexp(
            logits_off.float()[..., : model.config.audio_mask_id], dim=-1
        )
        if not torch.isfinite(delta).all():
            raise RuntimeError("non-finite acoustic partition delta")
        partition_delta = max(partition_delta, float(delta.abs().max().item()))
        off_base_exact &= _logits_hash(logits_off) == expected_base_hashes[pack_index]

        mutated = dict(batch)
        mutated_labels = batch["labels"].clone()
        valid_target = mutated_labels.ne(-100)
        replacement = torch.remainder(
            mutated_labels.clamp_min(0) + 137,
            model.config.audio_mask_id,
        )
        mutated_labels[valid_target] = replacement[valid_target]
        mutated["labels"] = mutated_labels
        label_independent &= _logits_hash(_forward(model, mutated, device)) == _logits_hash(
            logits_on
        )
        masks = _masks(batch, mask_id=model.config.audio_mask_id, stride=stride)
        on_metrics.add(_nll(logits_on, batch["labels"]), masks)
        off_metrics.add(_nll(logits_off, batch["labels"]), masks)
        if diagnostic_metrics is not None:
            assert logits_diagnostic is not None
            diagnostic_metrics.add(_nll(logits_diagnostic, batch["labels"]), masks)
            del logits_diagnostic
        del logits_on, logits_off, delta
    del model, head
    gc.collect()
    torch.cuda.empty_cache()
    return (
        on_metrics.summary(),
        off_metrics.summary(),
        None if diagnostic_metrics is None else diagnostic_metrics.summary(),
        {
            "structural_logits_exact": structural_exact,
            "acoustic_log_partition_max_abs_delta": partition_delta,
            "head_off_equals_frozen_base_bitwise": off_base_exact,
            "labels_cannot_affect_logits": label_independent,
        },
    )


def _bootstrap_delta(left, right, *, samples: int = 10_000) -> dict:
    pairs = [
        (a, b)
        for a, b in zip(left, right)
        if a is not None and b is not None
    ]
    if len(pairs) < 8:
        raise RuntimeError(f"paired bootstrap needs at least 8 packs, got {len(pairs)}")
    rng = random.Random(20260719)
    n = len(pairs)
    values = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        values.append(sum(pairs[index][0] - pairs[index][1] for index in indices) / n)
    values.sort()
    return {
        "pairs": n,
        "mean": sum(a - b for a, b in pairs) / n,
        "ci95_low": values[int(0.025 * samples)],
        "ci95_high": values[int(0.975 * samples)],
    }


def _improvement(on: dict, off: dict, group: str) -> float:
    baseline = off[group]["weighted_nll"]
    return 100.0 * (baseline - on[group]["weighted_nll"]) / baseline


def _assert_finite(value, path="report") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite value at {path}: {value!r}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported value at {path}: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-config", type=Path, required=True)
    parser.add_argument("--stateless-config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--causal-checkpoint", type=Path, required=True)
    parser.add_argument("--stateless-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-packs", type=int, default=32)
    parser.add_argument("--benchmark-packs", type=int, default=4)
    parser.add_argument("--benchmark-warmup", type=int, default=2)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.num_packs < 32:
        raise SystemExit("num-packs must be at least 32")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for flex-attention evaluation")
    if len(
        {
            args.base_checkpoint.resolve(),
            args.causal_checkpoint.resolve(),
            args.stateless_checkpoint.resolve(),
        }
    ) != 3:
        raise SystemExit("base/causal/stateless checkpoints must be distinct")

    device = torch.device("cuda:0")
    stride = int(TrainingConfig.from_json(args.causal_config).block_anchor_stride)
    batches, snapshot_hash = _snapshot_batches(
        args.causal_config, args.causal_checkpoint, args.num_packs
    )
    performance = _benchmark_head_on_off(
        args.causal_config,
        args.causal_checkpoint,
        batches,
        device,
        packs=args.benchmark_packs,
        warmup=args.benchmark_warmup,
        repeats=args.benchmark_repeats,
    )
    base, base_hashes = _evaluate_base(
        args.causal_config, args.base_checkpoint, batches, device, stride
    )
    causal_on, causal_off, causal_same_weights_stateless, causal_invariants = _evaluate_pair(
        args.causal_config,
        args.causal_checkpoint,
        batches,
        device,
        stride,
        base_hashes,
        diagnostic_stateless=True,
    )
    stateless_on, stateless_off, _, stateless_invariants = _evaluate_pair(
        args.stateless_config,
        args.stateless_checkpoint,
        batches,
        device,
        stride,
        base_hashes,
    )

    causal_improvements = {
        group: _improvement(causal_on, causal_off, group)
        for group in ("overall", "suffix", "soft_only_suffix")
    }
    stateless_improvements = {
        group: _improvement(stateless_on, stateless_off, group)
        for group in ("overall", "suffix", "soft_only_suffix")
    }
    offset_improvements = {
        f"offset_{offset}": _improvement(
            causal_on, causal_off, f"offset_{offset}"
        )
        for offset in range(1, stride)
    }
    causal_advantage_soft_pct = 100.0 * (
        causal_same_weights_stateless["soft_only_suffix"]["weighted_nll"]
        - causal_on["soft_only_suffix"]["weighted_nll"]
    ) / causal_same_weights_stateless["soft_only_suffix"]["weighted_nll"]
    trained_causal_advantage_soft_pct = 100.0 * (
        stateless_on["soft_only_suffix"]["weighted_nll"]
        - causal_on["soft_only_suffix"]["weighted_nll"]
    ) / stateless_on["soft_only_suffix"]["weighted_nll"]
    overall_regression_vs_base_pct = 100.0 * (
        causal_on["overall"]["weighted_nll"] - base["overall"]["weighted_nll"]
    ) / base["overall"]["weighted_nll"]
    suffix_ci = _bootstrap_delta(
        causal_on["suffix"]["pack_weighted_nll"],
        causal_off["suffix"]["pack_weighted_nll"],
    )
    soft_ci = _bootstrap_delta(
        causal_on["soft_only_suffix"]["pack_weighted_nll"],
        causal_off["soft_only_suffix"]["pack_weighted_nll"],
    )
    causal_vs_stateless_soft_ci = _bootstrap_delta(
        causal_on["soft_only_suffix"]["pack_weighted_nll"],
        causal_same_weights_stateless["soft_only_suffix"]["pack_weighted_nll"],
    )
    trained_causal_vs_stateless_soft_ci = _bootstrap_delta(
        causal_on["soft_only_suffix"]["pack_weighted_nll"],
        stateless_on["soft_only_suffix"]["pack_weighted_nll"],
    )
    trained_stateless_soft_nll = stateless_on["soft_only_suffix"]["weighted_nll"]
    trained_noninferiority_margin_abs = (
        TRAINED_STATELESS_NONINFERIORITY_MARGIN_PCT
        * trained_stateless_soft_nll
        / 100.0
    )

    invalid: list[str] = []
    for name, invariants in (
        ("causal", causal_invariants),
        ("stateless", stateless_invariants),
    ):
        if not invariants["structural_logits_exact"]:
            invalid.append(f"{name} changed MASK/EOS logits")
        if invariants["acoustic_log_partition_max_abs_delta"] > PARTITION_MAX_ABS_DELTA:
            invalid.append(
                f"{name} acoustic partition drift "
                f"{invariants['acoustic_log_partition_max_abs_delta']:.6g} "
                f"> {PARTITION_MAX_ABS_DELTA}"
            )
        if not invariants["head_off_equals_frozen_base_bitwise"]:
            invalid.append(f"{name} head-off logits differ from frozen base")
        if not invariants["labels_cannot_affect_logits"]:
            invalid.append(f"{name} logits depend on labels")
    if causal_off != stateless_off or causal_off != base:
        invalid.append("head-off/base NLL summaries are not exactly identical")

    engineering: list[str] = []
    if performance["head_on_throughput_ratio"] < MIN_HEAD_ON_THROUGHPUT_RATIO:
        engineering.append(
            "head-on/off forward throughput ratio "
            f"{performance['head_on_throughput_ratio']:.4f} "
            f"< {MIN_HEAD_ON_THROUGHPUT_RATIO}"
        )
    if (
        performance["head_on_peak_delta_mib"]
        > performance["head_on_peak_delta_limit_mib"]
    ):
        engineering.append(
            "head-on forward peak memory delta "
            f"{performance['head_on_peak_delta_mib']:.1f} MiB > "
            f"{performance['head_on_peak_delta_limit_mib']:.1f} MiB"
        )

    strong_failures = []
    if causal_improvements["overall"] < 0.05:
        strong_failures.append("overall improvement < 0.05%")
    if causal_improvements["suffix"] < 0.15 or suffix_ci["ci95_high"] >= 0:
        strong_failures.append("suffix improvement/CI gate failed")
    if causal_improvements["soft_only_suffix"] < 0.10 or soft_ci["ci95_high"] >= 0:
        strong_failures.append("soft-only suffix improvement/CI gate failed")
    if (
        trained_causal_vs_stateless_soft_ci["ci95_high"]
        > trained_noninferiority_margin_abs
    ):
        strong_failures.append(
            "independently-trained causal/stateless paired noninferiority gate failed"
        )
    if (
        causal_advantage_soft_pct < 0.05
        or causal_vs_stateless_soft_ci["ci95_high"] >= 0
    ):
        strong_failures.append(
            "same-weight causal propagation advantage/CI gate failed"
        )
    if overall_regression_vs_base_pct > 0.10:
        strong_failures.append("overall NLL regression vs frozen base > 0.10%")
    first_three = [offset_improvements[f"offset_{offset}"] for offset in (1, 2, 3)]
    if any(value < -0.05 for value in first_three):
        strong_failures.append("one of offsets 1..3 regressed by > 0.05%")
    if sum(value >= 0.05 for value in first_three) < 2:
        strong_failures.append("fewer than two of offsets 1..3 improved >= 0.05%")

    confident_kill = []
    if suffix_ci["ci95_low"] >= 0:
        confident_kill.append("causal suffix is confidently non-improving")
    if soft_ci["ci95_low"] >= 0:
        confident_kill.append("causal soft-only suffix is confidently non-improving")
    if (
        trained_causal_vs_stateless_soft_ci["ci95_low"]
        > trained_noninferiority_margin_abs
    ):
        confident_kill.append(
            "independently-trained causal scan is confidently inferior to stateless"
        )
    if causal_vs_stateless_soft_ci["ci95_low"] >= 0:
        confident_kill.append(
            "same-weight causal propagation is confidently non-improving"
        )
    if causal_improvements["overall"] < -0.10:
        confident_kill.append("causal overall NLL regressed by > 0.10%")

    if invalid:
        verdict = "INVALID_IMPLEMENTATION"
    elif engineering:
        verdict = "ENGINEERING_BLOCK"
    elif not strong_failures:
        verdict = "PROMOTE_TO_3SEED"
    elif confident_kill:
        verdict = "SCIENTIFIC_KILL"
    else:
        verdict = "INCONCLUSIVE_1K"
    source_root = Path(__file__).resolve().parents[1]
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    config_provenance = {
        name: {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path.resolve()),
        }
        for name, path in (
            ("causal", args.causal_config),
            ("stateless", args.stateless_config),
        )
    }
    checkpoint_provenance = {
        name: {
            "path": str(path.resolve()),
            "model_sha256": _file_sha256(path.resolve() / "model.safetensors"),
        }
        for name, path in (
            ("base", args.base_checkpoint),
            ("causal", args.causal_checkpoint),
            ("stateless", args.stateless_checkpoint),
        )
    }
    report = {
        "verdict": verdict,
        "generation_status": "NEEDS_GENERATION",
        "scope": "seed-0, 300-step, fixed 32-pack mechanism proof; no quality claim",
        "seed_index": 0,
        "train_seed": 42,
        "eval_seed": 20260719,
        "english_only_mechanism_probe": True,
        "num_packs": args.num_packs,
        "snapshot_sha256": snapshot_hash,
        "proposal_contract": "full_acoustic_softmax_expectation",
        "provenance": {
            "source_root": str(source_root),
            "source_commit": source_commit,
            "configs": config_provenance,
            "checkpoints": checkpoint_provenance,
        },
        "base": base,
        "causal_on": causal_on,
        "causal_off": causal_off,
        "causal_same_weights_stateless": causal_same_weights_stateless,
        "stateless_on": stateless_on,
        "stateless_off": stateless_off,
        "causal_invariants": causal_invariants,
        "stateless_invariants": stateless_invariants,
        "causal_improvement_pct": causal_improvements,
        "stateless_improvement_pct": stateless_improvements,
        "causal_offset_improvement_pct": offset_improvements,
        "causal_advantage_soft_only_pct_vs_stateless": causal_advantage_soft_pct,
        "trained_causal_advantage_soft_only_pct_vs_stateless": (
            trained_causal_advantage_soft_pct
        ),
        "overall_regression_pct_vs_frozen_base": overall_regression_vs_base_pct,
        "paired_suffix_delta_on_minus_off": suffix_ci,
        "paired_soft_only_delta_on_minus_off": soft_ci,
        "paired_soft_only_delta_causal_minus_same_weights_stateless": (
            causal_vs_stateless_soft_ci
        ),
        "same_causal_checkpoint_mode_override_role": "mechanism_diagnostic_only",
        "paired_soft_only_delta_trained_causal_minus_trained_stateless": (
            trained_causal_vs_stateless_soft_ci
        ),
        "trained_stateless_noninferiority_margin_pct": (
            TRAINED_STATELESS_NONINFERIORITY_MARGIN_PCT
        ),
        "trained_stateless_noninferiority_margin_abs_nll": (
            trained_noninferiority_margin_abs
        ),
        "performance": performance,
        "invalid_failures": invalid,
        "engineering_failures": engineering,
        "strong_promotion_failures": strong_failures,
        "confident_kill_reasons": confident_kill,
    }
    _assert_finite(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("BLOCK_ANCHOR_NLL_VERDICT " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
