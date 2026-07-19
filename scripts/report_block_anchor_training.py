#!/usr/bin/env python3
"""Verify seed-0 training, frozen-base integrity, replay, speed, and memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import torch
from safetensors import safe_open


HEAD_PREFIX = "block_anchor_scan_head."
STEP_RE = re.compile(
    r"Step\s+(\d+)\s+\|[^\r\n]*?train/loss:\s*([0-9.eE+-]+)"
    r"[^\r\n]*?train/steps_per_sec:\s*([0-9.eE+-]+)"
)
ERROR_RE = re.compile(
    r"\bnan\b|\binf\b|out of memory|Traceback|NCCL[^\r\n]*(?:error|failed)",
    re.IGNORECASE,
)
# The replay still reports bitwise equality as the strongest signal, while
# these explicit numeric floors keep harmless kernel-order noise from turning
# into an implementation verdict.  Both are far below the 0.05% NLL science
# gates used by the fixed-pack evaluator.
REPLAY_HEAD_MAX_ABS_TOL = 5e-6
REPLAY_LOSS_MAX_ABS_TOL = 1e-5


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _safetensor_hash(path: Path, *, include_head: bool) -> tuple[str, list[str]]:
    digest = hashlib.sha256()
    selected: list[str] = []
    with safe_open(path, framework="pt", device="cpu") as stream:
        for key in sorted(stream.keys()):
            is_head = key.startswith(HEAD_PREFIX)
            if is_head != include_head:
                continue
            tensor = stream.get_tensor(key)
            digest.update(key.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(_tensor_bytes(tensor))
            selected.append(key)
    return digest.hexdigest(), selected


def _head_state(path: Path) -> dict:
    head_hash, keys = _safetensor_hash(path, include_head=True)
    if not keys:
        raise RuntimeError(f"checkpoint contains no {HEAD_PREFIX} tensors: {path}")
    output_nonzero = 0
    output_l2_sq = 0.0
    with safe_open(path, framework="pt", device="cpu") as stream:
        output_keys = [
            key
            for key in keys
            if key.endswith("output.weight")
            or ".output." in key
            or key.endswith("out.weight")
        ]
        if not output_keys:
            raise RuntimeError(
                "cannot locate the zero-initialized anchor output projection: "
                f"keys={keys}"
            )
        for key in keys:
            tensor = stream.get_tensor(key)
            if not bool(tensor.isfinite().all()):
                raise RuntimeError(f"non-finite anchor tensor {key} in {path}")
        for key in output_keys:
            tensor = stream.get_tensor(key)
            output_nonzero += int(tensor.count_nonzero().item())
            output_l2_sq += float(tensor.float().square().sum().item())
    if output_nonzero == 0:
        raise RuntimeError("anchor output projection remained exactly zero")
    return {
        "sha256": head_hash,
        "tensor_keys": keys,
        "tensor_count": len(keys),
        "parameter_tensor_count": len(
            [key for key in keys if not key.endswith("codebook_offsets")]
        ),
        "output_keys": output_keys,
        "output_nonzero": output_nonzero,
        "output_l2_norm": math.sqrt(output_l2_sq),
    }


def _head_max_abs_delta(left: Path, right: Path) -> float:
    with safe_open(left, framework="pt", device="cpu") as left_stream, safe_open(
        right, framework="pt", device="cpu"
    ) as right_stream:
        left_keys = sorted(
            key for key in left_stream.keys() if key.startswith(HEAD_PREFIX)
        )
        right_keys = sorted(
            key for key in right_stream.keys() if key.startswith(HEAD_PREFIX)
        )
        if left_keys != right_keys:
            raise RuntimeError("replay head tensor schemas differ")
        maximum = 0.0
        for key in left_keys:
            left_tensor = left_stream.get_tensor(key)
            right_tensor = right_stream.get_tensor(key)
            if left_tensor.shape != right_tensor.shape:
                raise RuntimeError(f"replay head tensor shape differs for {key}")
            maximum = max(
                maximum,
                float((left_tensor.float() - right_tensor.float()).abs().max()),
            )
        return maximum


def _optimizer_parameter_count(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    groups = payload.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError(f"optimizer has no parameter groups: {path}")
    params = [parameter for group in groups for parameter in group.get("params", [])]
    if len(params) != len(set(params)):
        raise RuntimeError(f"optimizer repeats parameter ids: {path}")
    return len(params)


def _checkpoint(path: Path, expected_mode: str, base_hash: str) -> dict:
    required = (
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        "train_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
    )
    for name in required:
        item = path / name
        if not item.is_file() or item.stat().st_size == 0:
            raise RuntimeError(f"incomplete checkpoint: {item}")
    random_states = sorted(path.glob("random_states_*.pkl"))
    if len(random_states) != 2 or any(item.stat().st_size == 0 for item in random_states):
        raise RuntimeError(
            f"expected two non-empty random states in {path}, got {len(random_states)}"
        )
    config = json.loads((path / "config.json").read_text())
    contract = {
        "block_markov_rank": int(config.get("block_markov_rank", 0)),
        "block_anchor_scan_dim": int(config.get("block_anchor_scan_dim", 0)),
        "block_anchor_proposal_dim": int(config.get("block_anchor_proposal_dim", 0)),
        "block_anchor_stride": int(config.get("block_anchor_stride", 0)),
        "block_anchor_mode": config.get("block_anchor_mode"),
    }
    expected = {
        "block_markov_rank": 0,
        "block_anchor_scan_dim": 64,
        "block_anchor_proposal_dim": 32,
        "block_anchor_stride": 8,
        "block_anchor_mode": expected_mode,
    }
    if contract != expected:
        raise RuntimeError(
            f"checkpoint architecture mismatch: expected={expected}, actual={contract}"
        )
    backbone_hash, backbone_keys = _safetensor_hash(
        path / "model.safetensors", include_head=False
    )
    if backbone_hash != base_hash:
        raise RuntimeError(
            "frozen backbone hash drifted: "
            f"checkpoint={path} expected={base_hash} actual={backbone_hash}"
        )
    head = _head_state(path / "model.safetensors")
    optimizer_parameter_count = _optimizer_parameter_count(path / "optimizer.bin")
    if optimizer_parameter_count != head["parameter_tensor_count"]:
        raise RuntimeError(
            "optimizer is not head-only: "
            f"optimizer_params={optimizer_parameter_count} "
            f"head_parameters={head['parameter_tensor_count']} path={path}"
        )
    return {
        "architecture": contract,
        "backbone_sha256": backbone_hash,
        "backbone_tensor_count": len(backbone_keys),
        "head": head,
        "optimizer_parameter_count": optimizer_parameter_count,
        "random_state_count": len(random_states),
    }


def _log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    error = ERROR_RE.search(text)
    if error:
        raise RuntimeError(f"fatal log pattern {error.group(0)!r}: {path}")
    points = [
        (int(step), float(loss), float(rate))
        for step, loss, rate in STEP_RE.findall(text)
    ]
    points = [point for point in points if point[0] >= 100]
    if len(points) < 10:
        raise RuntimeError(f"too few post-warmup telemetry points in {path}")
    if any(not math.isfinite(value) for point in points for value in point[1:]):
        raise RuntimeError(f"non-finite training telemetry in {path}")
    return {
        "points": points,
        "telemetry_points": len(points),
        "median_steps_per_sec": statistics.median(point[2] for point in points),
        "last_step": points[-1][0],
        "last_loss": points[-1][1],
    }


def _memory(path: Path) -> dict:
    values: dict[int, list[int]] = {}
    for line in path.read_text(errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            index, memory = int(fields[1]), int(fields[2])
        except ValueError:
            continue
        values.setdefault(index, []).append(memory)
    if len(values) != 2:
        raise RuntimeError(
            f"memory sampler must cover exactly two GPUs in {path}, got {sorted(values)}"
        )
    if any(len(samples) < 10 for samples in values.values()):
        raise RuntimeError(f"too few memory samples in {path}")
    return {
        "peak_mib": max(max(samples) for samples in values.values()),
        "samples_per_gpu": {
            str(index): len(samples) for index, samples in sorted(values.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-log", type=Path, required=True)
    parser.add_argument("--stateless-log", type=Path, required=True)
    parser.add_argument("--causal-replay-log", type=Path, required=True)
    parser.add_argument("--causal-memory", type=Path, required=True)
    parser.add_argument("--stateless-memory", type=Path, required=True)
    parser.add_argument("--causal-replay-memory", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--causal-checkpoint", type=Path, required=True)
    parser.add_argument("--stateless-checkpoint", type=Path, required=True)
    parser.add_argument("--causal-replay-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_hash, base_keys = _safetensor_hash(
        args.base_checkpoint / "model.safetensors", include_head=False
    )
    causal_artifact = _checkpoint(args.causal_checkpoint, "causal", base_hash)
    stateless_artifact = _checkpoint(args.stateless_checkpoint, "stateless", base_hash)
    replay_artifact = _checkpoint(args.causal_replay_checkpoint, "causal", base_hash)
    causal = _log(args.causal_log)
    stateless = _log(args.stateless_log)
    replay = _log(args.causal_replay_log)
    causal_memory = _memory(args.causal_memory)
    stateless_memory = _memory(args.stateless_memory)
    replay_memory = _memory(args.causal_replay_memory)

    invalid: list[str] = []
    engineering: list[str] = []
    if any(report["last_step"] != 300 for report in (causal, stateless, replay)):
        invalid.append("one training arm did not report step 300")
    causal_loss_trace = [(step, loss) for step, loss, _ in causal["points"]]
    replay_loss_trace = [(step, loss) for step, loss, _ in replay["points"]]
    if [step for step, _ in causal_loss_trace] != [step for step, _ in replay_loss_trace]:
        invalid.append("causal replay telemetry step indices differ")
        replay_loss_max_abs_delta = REPLAY_LOSS_MAX_ABS_TOL + 1.0
    else:
        replay_loss_max_abs_delta = max(
            abs(left - right)
            for (_, left), (_, right) in zip(causal_loss_trace, replay_loss_trace)
        )
        if replay_loss_max_abs_delta > REPLAY_LOSS_MAX_ABS_TOL:
            invalid.append(
                "causal replay loss drift "
                f"{replay_loss_max_abs_delta:.9g} > {REPLAY_LOSS_MAX_ABS_TOL}"
            )
    replay_head_max_abs_delta = _head_max_abs_delta(
        args.causal_checkpoint / "model.safetensors",
        args.causal_replay_checkpoint / "model.safetensors",
    )
    if replay_head_max_abs_delta > REPLAY_HEAD_MAX_ABS_TOL:
        invalid.append(
            "causal replay head drift "
            f"{replay_head_max_abs_delta:.9g} > {REPLAY_HEAD_MAX_ABS_TOL}"
        )
    if causal_artifact["head"]["tensor_keys"] != stateless_artifact["head"]["tensor_keys"]:
        invalid.append("causal/stateless head tensor schemas differ")

    causal_replay_speed_drift = abs(
        causal["median_steps_per_sec"] - replay["median_steps_per_sec"]
    ) / causal["median_steps_per_sec"]
    causal_stateless_speed_ratio = (
        causal["median_steps_per_sec"] / stateless["median_steps_per_sec"]
    )
    causal_stateless_memory_delta = (
        causal_memory["peak_mib"] - stateless_memory["peak_mib"]
    )
    if causal_replay_speed_drift > 0.05:
        engineering.append(
            f"causal replay speed drift {causal_replay_speed_drift:.4f} > 0.05"
        )
    if causal_stateless_speed_ratio < 0.95:
        engineering.append(
            "causal/stateless speed ratio "
            f"{causal_stateless_speed_ratio:.4f} < 0.95"
        )
    if causal_stateless_memory_delta > 1024:
        engineering.append(
            "causal/stateless peak memory delta "
            f"{causal_stateless_memory_delta} MiB > 1024 MiB"
        )

    if invalid:
        verdict = "INVALID_IMPLEMENTATION"
    elif engineering:
        verdict = "ENGINEERING_BLOCK"
    else:
        verdict = "PASS"
    for report in (causal, stateless, replay):
        report.pop("points")
    payload = {
        "verdict": verdict,
        "invalid_failures": invalid,
        "engineering_failures": engineering,
        "frozen_base": {
            "model": str(args.base_checkpoint / "model.safetensors"),
            "backbone_sha256": base_hash,
            "backbone_tensor_count": len(base_keys),
        },
        "causal": {
            "artifact": causal_artifact,
            "timing": causal,
            "memory": causal_memory,
        },
        "stateless": {
            "artifact": stateless_artifact,
            "timing": stateless,
            "memory": stateless_memory,
        },
        "causal_replay": {
            "artifact": replay_artifact,
            "timing": replay,
            "memory": replay_memory,
        },
        "causal_replay_speed_drift": causal_replay_speed_drift,
        "causal_stateless_speed_ratio": causal_stateless_speed_ratio,
        "causal_stateless_peak_memory_delta_mib": causal_stateless_memory_delta,
        "frozen_backbone_hash_exact": all(
            artifact["backbone_sha256"] == base_hash
            for artifact in (causal_artifact, stateless_artifact, replay_artifact)
        ),
        "causal_replay_bitwise_exact": (
            causal_artifact["head"]["sha256"]
            == replay_artifact["head"]["sha256"]
        ),
        "causal_replay_loss_trace_bitwise_exact": causal_loss_trace == replay_loss_trace,
        "causal_replay_head_max_abs_delta": replay_head_max_abs_delta,
        "causal_replay_loss_max_abs_delta": replay_loss_max_abs_delta,
        "replay_numeric_noise_floor": {
            "head_max_abs": REPLAY_HEAD_MAX_ABS_TOL,
            "logged_loss_max_abs": REPLAY_LOSS_MAX_ABS_TOL,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("BLOCK_ANCHOR_TRAINING_VERDICT " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
