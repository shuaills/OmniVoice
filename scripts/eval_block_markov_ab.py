#!/usr/bin/env python3
"""Fixed-pack acoustic-NLL verdict for the 300-step Markov-head A/B."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from transformers import AutoTokenizer

from omnivoice.blockdiff_dual import (
    KIND_ACOUSTIC,
    KIND_EOS,
    KIND_VOID,
    TAG_NOISY,
)
from omnivoice.training.builder import (
    build_dataloaders,
    build_model_and_tokenizer,
)
from omnivoice.training.config import TrainingConfig


GROUPS = (
    "overall",
    "known",
    "unknown",
    "early",
    "late",
    "eos",
    "void",
)
STRUCTURAL_LOGPROB_MAX_ABS_DELTA = 0.002


def _config(path: Path, checkpoint: Path, rank: int) -> TrainingConfig:
    config = TrainingConfig.from_json(path)
    config.init_from_checkpoint = str(checkpoint)
    config.resume_from_checkpoint = None
    config.block_markov_rank = rank
    config.num_workers = 1
    config.batch_tokens = 4096
    config.perf_grad_checkpoint = False
    config.perf_liger = False
    config.perf_fused_adamw = False
    config.perf_train_no_cache = True
    return config


def _snapshot_batches(
    config_path: Path,
    head_checkpoint: Path,
    num_packs: int,
) -> tuple[list[dict[str, torch.Tensor]], str]:
    config = _config(config_path, head_checkpoint, rank=32)
    set_seed(20260719)
    tokenizer = AutoTokenizer.from_pretrained(head_checkpoint)
    _, dev_loader = build_dataloaders(config, tokenizer)
    if dev_loader is None:
        raise RuntimeError("data config has no dev manifests")

    batches: list[dict[str, torch.Tensor]] = []
    digest = hashlib.sha256()
    iterator = iter(dev_loader)
    for pack_index in range(num_packs):
        batch = next(iterator)
        frozen = {
            name: value.detach().cpu().clone()
            for name, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        required = {
            "input_ids",
            "audio_mask",
            "labels",
            "document_ids",
            "copy_tags",
            "block_ids",
            "loss_kind",
            "markov_prev_ids",
        }
        missing = sorted(required - set(frozen))
        if missing:
            raise RuntimeError(f"dev pack is missing required tensors: {missing}")
        for name in sorted(frozen):
            tensor = frozen[name].contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        batches.append(frozen)
        print(f"BLOCK_MARKOV_DEV_PACK_READY index={pack_index}", flush=True)
    del iterator, dev_loader, tokenizer
    gc.collect()
    return batches, digest.hexdigest()


def _block_offsets(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    document_ids = batch["document_ids"]
    copy_tags = batch["copy_tags"]
    block_ids = batch["block_ids"]
    offsets = torch.full_like(document_ids, -1)
    for batch_index in range(document_ids.size(0)):
        documents = torch.unique(document_ids[batch_index])
        for document in documents[documents >= 0]:
            in_document = document_ids[batch_index].eq(document)
            blocks = torch.unique(block_ids[batch_index, in_document])
            for block in blocks[blocks >= 0]:
                positions = (
                    in_document
                    & block_ids[batch_index].eq(block)
                    & copy_tags[batch_index].eq(TAG_NOISY)
                ).nonzero(as_tuple=True)[0]
                if positions.numel() > 0:
                    offsets[batch_index, positions] = torch.arange(
                        positions.numel(), dtype=offsets.dtype
                    )
    return offsets


class Metrics:
    def __init__(self, codebook_weights: list[float]) -> None:
        weights = torch.tensor(codebook_weights, dtype=torch.float64)
        self.weights = weights / weights.sum()
        self.sums = {
            group: torch.zeros(len(weights), dtype=torch.float64)
            for group in GROUPS
        }
        self.counts = {
            group: torch.zeros(len(weights), dtype=torch.int64)
            for group in GROUPS
        }
        self.pack_values = {group: [] for group in GROUPS}

    def add(
        self,
        nll: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> None:
        for group, mask in masks.items():
            codebook_means: list[torch.Tensor | None] = []
            for codebook in range(nll.size(1)):
                selected = nll[:, codebook][mask[:, codebook]]
                if selected.numel() == 0:
                    codebook_means.append(None)
                    continue
                self.sums[group][codebook] += selected.double().sum()
                self.counts[group][codebook] += selected.numel()
                codebook_means.append(selected.double().mean())
            active = [index for index, value in enumerate(codebook_means) if value is not None]
            if not active:
                self.pack_values[group].append(None)
                continue
            active_weights = self.weights[active]
            active_weights = active_weights / active_weights.sum()
            pack_value = sum(
                codebook_means[index] * active_weights[position]
                for position, index in enumerate(active)
            )
            self.pack_values[group].append(float(pack_value))

    def summary(self) -> dict:
        result = {}
        for group in GROUPS:
            active = self.counts[group].gt(0)
            if not active.any():
                raise RuntimeError(f"zero aggregate count in group {group}")
            per_codebook = torch.full_like(self.sums[group], float("nan"))
            per_codebook[active] = (
                self.sums[group][active] / self.counts[group][active]
            )
            active_weights = self.weights[active]
            active_weights = active_weights / active_weights.sum()
            result[group] = {
                "weighted_nll": float(
                    (per_codebook[active] * active_weights).sum()
                ),
                "codebook_nll": [
                    float(value) if torch.isfinite(value) else None
                    for value in per_codebook
                ],
                "counts": [int(value) for value in self.counts[group]],
                "pack_weighted_nll": self.pack_values[group],
            }
        return result


def _masks(
    batch: dict[str, torch.Tensor], *, mask_id: int, block_size: int
) -> dict[str, torch.Tensor]:
    acoustic = batch["loss_kind"].eq(KIND_ACOUSTIC)
    known_frame = batch["markov_prev_ids"].ne(mask_id).any(dim=1)
    known = acoustic & known_frame.unsqueeze(1)
    unknown = acoustic & ~known_frame.unsqueeze(1)
    offsets = _block_offsets(batch)
    split_offset = block_size // 2
    early = acoustic & offsets.lt(split_offset).unsqueeze(1)
    late = acoustic & offsets.ge(split_offset).unsqueeze(1)
    return {
        "overall": acoustic,
        "known": known,
        "unknown": unknown,
        "early": early,
        "late": late,
        "eos": batch["loss_kind"].eq(KIND_EOS),
        "void": batch["loss_kind"].eq(KIND_VOID),
    }


FORWARD_KEYS = {
    "input_ids",
    "audio_mask",
    "attention_mask",
    "document_ids",
    "position_ids",
    "copy_tags",
    "block_ids",
    "markov_prev_ids",
}


def _forward_logits(model, batch, device):
    inputs = {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if name in FORWARD_KEYS
    }
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        return model(**inputs).logits


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().permute(0, 3, 1, 2),
        labels.to(logits.device),
        reduction="none",
        ignore_index=-100,
    ).cpu()


def _load_model(config_path, checkpoint, rank, device):
    config = _config(config_path, checkpoint, rank)
    model, _ = build_model_and_tokenizer(config)
    model._split_loss = False
    model.eval().to(device)
    return model


def _evaluate_control(config_path, checkpoint, batches, device, block_size):
    model = _load_model(config_path, checkpoint, 0, device)
    metrics = Metrics(model.config.audio_codebook_weights)
    torch.cuda.reset_peak_memory_stats(device)
    for batch in batches:
        logits = _forward_logits(model, batch, device)
        metrics.add(
            _nll(logits, batch["labels"]),
            _masks(
                batch,
                mask_id=model.config.audio_mask_id,
                block_size=block_size,
            ),
        )
        del logits
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics.summary(), peak


def _evaluate_head_pair(config_path, checkpoint, batches, device, block_size):
    model = _load_model(config_path, checkpoint, 32, device)
    head = model.block_markov_head
    if head is None:
        raise RuntimeError("head checkpoint loaded without block_markov_head")
    output_nonzero = int(torch.count_nonzero(head.output.weight).item())
    output_norm = float(head.output.weight.float().norm().item())
    if output_nonzero == 0:
        raise RuntimeError("trained head output projection is still zero")

    on_metrics = Metrics(model.config.audio_codebook_weights)
    off_metrics = Metrics(model.config.audio_codebook_weights)
    structural_logits_exact = True
    structural_logprob_max_abs_delta = 0.0
    acoustic_log_partition_max_abs_delta = 0.0
    unknown_exact = True
    torch.cuda.reset_peak_memory_stats(device)
    for batch in batches:
        logits_on = _forward_logits(model, batch, device)
        model.block_markov_head = None
        logits_off = _forward_logits(model, batch, device)
        model.block_markov_head = head

        structural_logits_exact &= torch.equal(
            logits_on[..., model.config.audio_mask_id :],
            logits_off[..., model.config.audio_mask_id :],
        )
        on_fp32 = logits_on.float()
        off_fp32 = logits_off.float()
        # Structural logits are byte-identical, so their log-probability
        # drift is exactly the negative drift of the full log-partition.
        structural_partition_delta = (
            torch.logsumexp(on_fp32, dim=-1)
            - torch.logsumexp(off_fp32, dim=-1)
        )
        acoustic_partition_delta = torch.logsumexp(
            on_fp32[..., : model.config.audio_mask_id], dim=-1
        ) - torch.logsumexp(
            off_fp32[..., : model.config.audio_mask_id], dim=-1
        )
        if not torch.isfinite(structural_partition_delta).all():
            raise RuntimeError("non-finite structural log-partition delta")
        if not torch.isfinite(acoustic_partition_delta).all():
            raise RuntimeError("non-finite acoustic log-partition delta")
        structural_logprob_max_abs_delta = max(
            structural_logprob_max_abs_delta,
            float(structural_partition_delta.abs().max().item()),
        )
        acoustic_log_partition_max_abs_delta = max(
            acoustic_log_partition_max_abs_delta,
            float(acoustic_partition_delta.abs().max().item()),
        )
        unknown_frame = batch["markov_prev_ids"].eq(
            model.config.audio_mask_id
        ).all(dim=1).to(device)
        on_by_frame = logits_on.permute(0, 2, 1, 3)
        off_by_frame = logits_off.permute(0, 2, 1, 3)
        unknown_exact &= torch.equal(
            on_by_frame[unknown_frame], off_by_frame[unknown_frame]
        )

        masks = _masks(
            batch,
            mask_id=model.config.audio_mask_id,
            block_size=block_size,
        )
        on_metrics.add(_nll(logits_on, batch["labels"]), masks)
        off_metrics.add(_nll(logits_off, batch["labels"]), masks)
        del logits_on, logits_off
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del model, head
    gc.collect()
    torch.cuda.empty_cache()
    return (
        on_metrics.summary(),
        off_metrics.summary(),
        {
            "structural_logits_exact": structural_logits_exact,
            "structural_logprob_max_abs_delta": (
                structural_logprob_max_abs_delta
            ),
            "acoustic_log_partition_max_abs_delta": (
                acoustic_log_partition_max_abs_delta
            ),
            "unknown_frame_logits_exact": unknown_exact,
            "output_nonzero": output_nonzero,
            "output_l2_norm": output_norm,
        },
        peak,
    )


def _bootstrap_delta(on_values, off_values, samples=10_000):
    if len(on_values) != len(off_values) or not on_values:
        raise ValueError("paired bootstrap requires equal non-empty packs")
    pairs = [
        (on, off)
        for on, off in zip(on_values, off_values)
        if on is not None and off is not None
    ]
    if len(pairs) < 8:
        raise ValueError(
            f"paired bootstrap requires at least 8 non-empty packs, got {len(pairs)}"
        )
    rng = random.Random(20260719)
    n = len(pairs)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        delta = sum(pairs[i][0] - pairs[i][1] for i in indices) / n
        deltas.append(delta)
    deltas.sort()
    return {
        "pairs": n,
        "mean": sum(on - off for on, off in pairs) / n,
        "ci95_low": deltas[int(0.025 * samples)],
        "ci95_high": deltas[int(0.975 * samples)],
    }


def _assert_finite(value, path: str = "report") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite evaluation value at {path}: {value!r}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported evaluation value at {path}: {type(value)!r}")


def _symmetric_delta_pct(left: float, right: float) -> float:
    mean = 0.5 * (left + right)
    if mean <= 0.0:
        raise RuntimeError(
            f"NLL values must have a positive mean, got left={left} right={right}"
        )
    return 100.0 * abs(left - right) / mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--head-config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--control-replay-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-packs", type=int, default=32)
    args = parser.parse_args()
    if args.num_packs < 8:
        raise SystemExit("num-packs must be at least 8 for paired bootstrap")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for flex-attention evaluation")
    device = torch.device("cuda:0")
    eval_config = TrainingConfig.from_json(args.head_config)
    block_size = int(eval_config.block_size)
    if block_size <= 1 or block_size % 2:
        raise SystemExit(f"block_size must be an even integer > 1, got {block_size}")
    checkpoint_paths = {
        args.base_checkpoint.resolve(),
        args.control_checkpoint.resolve(),
        args.control_replay_checkpoint.resolve(),
        args.head_checkpoint.resolve(),
    }
    if len(checkpoint_paths) != 4:
        raise SystemExit("base/control/control-replay/head checkpoints must be distinct")

    batches, snapshot_hash = _snapshot_batches(
        args.head_config,
        args.head_checkpoint,
        args.num_packs,
    )
    base, base_peak = _evaluate_control(
        args.control_config,
        args.base_checkpoint,
        batches,
        device,
        block_size,
    )
    control, control_peak = _evaluate_control(
        args.control_config,
        args.control_checkpoint,
        batches,
        device,
        block_size,
    )
    control_replay, control_replay_peak = _evaluate_control(
        args.control_config,
        args.control_replay_checkpoint,
        batches,
        device,
        block_size,
    )
    head_on, head_off, invariants, head_peak = _evaluate_head_pair(
        args.head_config,
        args.head_checkpoint,
        batches,
        device,
        block_size,
    )
    for name, summary in (
        ("base", base),
        ("control", control),
        ("control_replay", control_replay),
        ("head_on", head_on),
        ("head_off", head_off),
        ("invariants", invariants),
    ):
        _assert_finite(summary, name)

    known_on = head_on["known"]["weighted_nll"]
    known_control = control["known"]["weighted_nll"]
    known_control_replay = control_replay["known"]["weighted_nll"]
    known_off = head_off["known"]["weighted_nll"]
    known_improvement_pct = 100.0 * (
        known_control - known_on
    ) / known_control
    known_replay_improvement_pct = 100.0 * (
        known_control_replay - known_on
    ) / known_control_replay
    known_control_mean = 0.5 * (known_control + known_control_replay)
    known_control_half_spread_pct = (
        50.0 * abs(known_control - known_control_replay) / known_control_mean
    )
    known_mean_improvement_pct = 100.0 * (
        known_control_mean - known_on
    ) / known_control_mean
    known_improvement_margin_over_noise_pct = (
        known_mean_improvement_pct - known_control_half_spread_pct
    )
    mechanism_known_improvement_pct = 100.0 * (
        known_off - known_on
    ) / known_off
    overall_regression_pct = 100.0 * (
        head_on["overall"]["weighted_nll"]
        - control["overall"]["weighted_nll"]
    ) / control["overall"]["weighted_nll"]
    improved_codebooks = sum(
        on is not None
        and control_value is not None
        and replay_value is not None
        and on < min(control_value, replay_value)
        for on, control_value, replay_value in zip(
            head_on["known"]["codebook_nll"],
            control["known"]["codebook_nll"],
            control_replay["known"]["codebook_nll"],
        )
    )
    late_improved = (
        head_on["late"]["weighted_nll"]
        < min(
            control["late"]["weighted_nll"],
            control_replay["late"]["weighted_nll"],
        )
    )
    bootstrap_control = _bootstrap_delta(
        head_on["known"]["pack_weighted_nll"],
        control["known"]["pack_weighted_nll"],
    )
    bootstrap_replay = _bootstrap_delta(
        head_on["known"]["pack_weighted_nll"],
        control_replay["known"]["pack_weighted_nll"],
    )
    bootstrap_control_replay = _bootstrap_delta(
        control["known"]["pack_weighted_nll"],
        control_replay["known"]["pack_weighted_nll"],
    )
    control_replay_nll_drift_pct = {
        group: _symmetric_delta_pct(
            control[group]["weighted_nll"],
            control_replay[group]["weighted_nll"],
        )
        for group in ("overall", "known", "unknown", "eos", "void")
    }
    base_overall_regression_pct = 100.0 * (
        head_on["overall"]["weighted_nll"]
        - base["overall"]["weighted_nll"]
    ) / base["overall"]["weighted_nll"]
    structural_regressions = {
        group: 100.0
        * (
            head_on[group]["weighted_nll"]
            - control[group]["weighted_nll"]
        )
        / control[group]["weighted_nll"]
        for group in ("unknown", "eos", "void")
    }
    replay_regressions = {
        group: 100.0
        * (
            head_on[group]["weighted_nll"]
            - control_replay[group]["weighted_nll"]
        )
        / control_replay[group]["weighted_nll"]
        for group in ("overall", "unknown", "eos", "void")
    }
    base_regressions = {
        group: 100.0
        * (
            head_on[group]["weighted_nll"]
            - base[group]["weighted_nll"]
        )
        / base[group]["weighted_nll"]
        for group in ("known", "unknown", "eos", "void")
    }

    failures = []
    if not invariants["structural_logits_exact"]:
        failures.append("structural logits changed")
    if (
        invariants["structural_logprob_max_abs_delta"]
        > STRUCTURAL_LOGPROB_MAX_ABS_DELTA
    ):
        failures.append(
            "structural log-probability drift exceeds "
            f"{STRUCTURAL_LOGPROB_MAX_ABS_DELTA}: "
            f"{invariants['structural_logprob_max_abs_delta']:.6g}"
        )
    if (
        invariants["acoustic_log_partition_max_abs_delta"]
        > STRUCTURAL_LOGPROB_MAX_ABS_DELTA
    ):
        failures.append(
            "acoustic log-partition drift exceeds "
            f"{STRUCTURAL_LOGPROB_MAX_ABS_DELTA}: "
            f"{invariants['acoustic_log_partition_max_abs_delta']:.6g}"
        )
    if not invariants["unknown_frame_logits_exact"]:
        failures.append("unknown-predecessor logits changed")
    if known_improvement_pct < 0.1:
        failures.append(
            f"known-predecessor NLL improvement {known_improvement_pct:.4f}% < 0.1%"
        )
    if known_replay_improvement_pct < 0.1:
        failures.append(
            "known-predecessor NLL improvement vs replay "
            f"{known_replay_improvement_pct:.4f}% < 0.1%"
        )
    if known_improvement_margin_over_noise_pct < 0.1:
        failures.append(
            "known-predecessor NLL improvement does not clear the control "
            "replay noise floor by 0.1 percentage points: "
            f"margin={known_improvement_margin_over_noise_pct:.4f}% "
            f"mean_improvement={known_mean_improvement_pct:.4f}% "
            f"half_spread={known_control_half_spread_pct:.4f}%"
        )
    if bootstrap_control["ci95_high"] >= 0:
        failures.append(
            "paired known-NLL CI vs control crosses zero: "
            f"{bootstrap_control['ci95_high']:.6g}"
        )
    if bootstrap_replay["ci95_high"] >= 0:
        failures.append(
            "paired known-NLL CI vs control replay crosses zero: "
            f"{bootstrap_replay['ci95_high']:.6g}"
        )
    for group, drift in control_replay_nll_drift_pct.items():
        if drift > 0.5:
            failures.append(
                f"control replay {group} NLL drift {drift:.4f}% > 0.5%"
            )
    num_codebooks = len(head_on["known"]["codebook_nll"])
    minimum_improved_codebooks = (num_codebooks + 1) // 2
    if improved_codebooks < minimum_improved_codebooks:
        failures.append(
            f"only {improved_codebooks}/{num_codebooks} codebooks improved; "
            f"need at least {minimum_improved_codebooks}"
        )
    if not late_improved:
        failures.append("late-block NLL did not improve")
    if overall_regression_pct > 0.5:
        failures.append(
            f"overall NLL regressed {overall_regression_pct:.4f}% vs control"
        )
    if base_overall_regression_pct > 0.5:
        failures.append(
            "overall NLL regressed "
            f"{base_overall_regression_pct:.4f}% vs frozen base"
        )
    for group, regression in structural_regressions.items():
        if regression > 0.5:
            failures.append(
                f"{group} NLL regressed {regression:.4f}% vs control"
            )
    for group, regression in replay_regressions.items():
        if regression > 0.5:
            failures.append(
                f"{group} NLL regressed {regression:.4f}% vs control replay"
            )
    for group, regression in base_regressions.items():
        if regression > 0.5:
            failures.append(
                f"{group} NLL regressed {regression:.4f}% vs frozen base"
            )

    report = {
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
        "scope": (
            "fixed deterministic packed batches from the configured held-out "
            "English dev shard; bilingual behavior is gated separately by "
            "the generation canary"
        ),
        "num_packs": args.num_packs,
        "block_size": block_size,
        "snapshot_sha256": snapshot_hash,
        "base": base,
        "control": control,
        "control_replay": control_replay,
        "head_on": head_on,
        "head_off": head_off,
        "invariants": invariants,
        "known_improvement_pct": known_improvement_pct,
        "known_improvement_pct_vs_control_replay": (
            known_replay_improvement_pct
        ),
        "known_control_replay_noise_floor": {
            "control_weighted_nll": known_control,
            "control_replay_weighted_nll": known_control_replay,
            "mean_weighted_nll": known_control_mean,
            "absolute_delta": abs(known_control - known_control_replay),
            "half_spread_pct_of_mean": known_control_half_spread_pct,
            "head_improvement_pct_from_mean": known_mean_improvement_pct,
            "head_margin_over_half_spread_pct": (
                known_improvement_margin_over_noise_pct
            ),
        },
        "mechanism_known_improvement_pct_head_on_vs_head_off": (
            mechanism_known_improvement_pct
        ),
        "overall_regression_pct_vs_control": overall_regression_pct,
        "overall_regression_pct_vs_frozen_base": base_overall_regression_pct,
        "structural_regression_pct_vs_control": structural_regressions,
        "regression_pct_vs_control_replay": replay_regressions,
        "regression_pct_vs_frozen_base": base_regressions,
        "known_improved_codebooks": improved_codebooks,
        "late_improved": late_improved,
        "paired_known_delta_vs_control": bootstrap_control,
        "paired_known_delta_vs_control_replay": bootstrap_replay,
        "paired_known_delta_control_vs_replay": bootstrap_control_replay,
        "control_replay_nll_drift_pct": control_replay_nll_drift_pct,
        "eval_peak_memory_bytes": {
            "base": base_peak,
            "control": control_peak,
            "control_replay": control_replay_peak,
            "head_pair": head_peak,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _assert_finite(report)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        "BLOCK_MARKOV_AB_VERDICT "
        + json.dumps(report, sort_keys=True, allow_nan=False)
    )


if __name__ == "__main__":
    main()
