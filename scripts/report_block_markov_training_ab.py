#!/usr/bin/env python3
"""Summarize throughput, memory, artifacts, and learned head state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from safetensors import safe_open


STEP_RE = re.compile(
    r"Step\s+(\d+)\s+\|[^\r\n]*?train/loss:\s*([0-9.eE+-]+)"
    r"[^\r\n]*?train/steps_per_sec:\s*([0-9.eE+-]+)"
)
ERROR_RE = re.compile(
    r"\bnan\b|\binf\b|out of memory|Traceback|NCCL[^\r\n]*(?:error|failed)",
    re.IGNORECASE,
)


def _checkpoint(path: Path, expected_rank: int) -> dict:
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
    rank = int(config.get("block_markov_rank", 0))
    if rank != expected_rank:
        raise RuntimeError(
            f"checkpoint rank mismatch: expected={expected_rank}, actual={rank}"
        )
    return {"rank": rank, "random_state_count": len(random_states)}


def _log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    error = ERROR_RE.search(text)
    if error:
        raise RuntimeError(
            f"training log contains fatal pattern {error.group(0)!r}: {path}"
        )
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
            gpu_index = int(fields[1])
            memory = int(fields[2])
        except ValueError:
            continue
        values.setdefault(gpu_index, []).append(memory)
    if len(values) != 2:
        raise RuntimeError(
            f"memory sampler must cover exactly two GPU indices in {path}, "
            f"got {sorted(values)}"
        )
    sparse = {
        index: len(samples)
        for index, samples in values.items()
        if len(samples) < 10
    }
    if sparse:
        raise RuntimeError(f"too few GPU memory samples in {path}: {sparse}")
    return {
        "peak_mib": max(max(samples) for samples in values.values()),
        "samples_per_gpu": {
            str(index): len(samples) for index, samples in sorted(values.items())
        },
    }


def _head_state(model_path: Path) -> dict:
    with safe_open(model_path, framework="pt", device="cpu") as stream:
        keys = set(stream.keys())
        output_key = "block_markov_head.output.weight"
        embedding_key = "block_markov_head.prev_embeddings.weight"
        if output_key not in keys or embedding_key not in keys:
            raise RuntimeError("head checkpoint is missing Markov tensors")
        output = stream.get_tensor(output_key)
        embedding = stream.get_tensor(embedding_key).contiguous()
    if not bool(output.isfinite().all()) or not bool(embedding.isfinite().all()):
        raise RuntimeError("head checkpoint contains non-finite Markov tensors")
    nonzero = int(output.count_nonzero().item())
    if nonzero == 0:
        raise RuntimeError("head output projection remained exactly zero")
    return {
        "output_nonzero": nonzero,
        "output_l2_norm": float(output.float().norm().item()),
        "embedding_sha256": hashlib.sha256(
            embedding.numpy().tobytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--head-log", type=Path, required=True)
    parser.add_argument("--control-replay-log", type=Path, required=True)
    parser.add_argument("--control-memory", type=Path, required=True)
    parser.add_argument("--head-memory", type=Path, required=True)
    parser.add_argument("--control-replay-memory", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--control-replay-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control_artifact = _checkpoint(args.control_checkpoint, 0)
    head_artifact = _checkpoint(args.head_checkpoint, 32)
    replay_artifact = _checkpoint(args.control_replay_checkpoint, 0)
    control_timing = _log(args.control_log)
    head_timing = _log(args.head_log)
    replay_timing = _log(args.control_replay_log)
    control_memory = _memory(args.control_memory)
    head_memory = _memory(args.head_memory)
    replay_memory = _memory(args.control_replay_memory)
    memory_index_sets = {
        tuple(map(int, report["samples_per_gpu"]))
        for report in (control_memory, head_memory, replay_memory)
    }
    if len(memory_index_sets) != 1:
        raise RuntimeError(
            "GPU memory sampler index set drifted across arms: "
            f"{sorted(memory_index_sets)}"
        )
    head_state = _head_state(args.head_checkpoint / "model.safetensors")

    warm_control_rate = replay_timing["median_steps_per_sec"]
    speed_ratio = head_timing["median_steps_per_sec"] / warm_control_rate
    control_replay_drift = abs(
        warm_control_rate - control_timing["median_steps_per_sec"]
    ) / control_timing["median_steps_per_sec"]
    memory_delta = head_memory["peak_mib"] - replay_memory["peak_mib"]
    control_replay_loss_delta = abs(
        replay_timing["last_loss"] - control_timing["last_loss"]
    )
    failures = []
    if any(
        timing["last_step"] != 300
        for timing in (control_timing, head_timing, replay_timing)
    ):
        failures.append("one arm did not report step 300")
    if speed_ratio < 0.90:
        failures.append(f"head/control speed ratio {speed_ratio:.4f} < 0.90")
    if control_replay_drift > 0.05:
        failures.append(
            f"control replay speed drift {control_replay_drift:.4f} > 0.05"
        )
    if control_replay_loss_delta > 0.10:
        failures.append(
            "control replay final-loss drift "
            f"{control_replay_loss_delta:.6f} > 0.10"
        )
    if memory_delta > 2048:
        failures.append(f"head peak memory delta {memory_delta} MiB > 2048 MiB")

    report = {
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
        "control": {
            "artifact": control_artifact,
            "timing": control_timing,
            "memory": control_memory,
        },
        "head": {
            "artifact": head_artifact,
            "timing": head_timing,
            "memory": head_memory,
            "state": head_state,
        },
        "control_replay": {
            "artifact": replay_artifact,
            "timing": replay_timing,
            "memory": replay_memory,
        },
        "head_control_speed_ratio": speed_ratio,
        "control_replay_speed_drift": control_replay_drift,
        "control_replay_final_loss_abs_delta": control_replay_loss_delta,
        "head_control_peak_memory_delta_mib": memory_delta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        "BLOCK_MARKOV_TRAINING_VERDICT "
        + json.dumps(report, sort_keys=True, allow_nan=False)
    )


if __name__ == "__main__":
    main()
