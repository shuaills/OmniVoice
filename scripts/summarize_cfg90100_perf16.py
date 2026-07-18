#!/usr/bin/env python3
"""Compare the 16-GPU GC-off arm with the frozen successful RDMA baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import statistics


METRIC_RE = re.compile(r"^Step (\d+) \| (.*)$")
GLOBAL_TOKENS_PER_STEP = 125184


def parse_metrics(path: pathlib.Path) -> dict[int, dict[str, float]]:
    metrics: dict[int, dict[str, float]] = {}
    for raw in path.read_text(errors="replace").splitlines():
        match = METRIC_RE.match(raw.strip())
        if not match:
            continue
        values: dict[str, float] = {}
        for field in match.group(2).split(" | "):
            key, sep, value = field.partition(": ")
            if not sep:
                continue
            try:
                values[key] = float(value)
            except ValueError:
                continue
        metrics[int(match.group(1))] = values
    return metrics


def steady_throughput(metrics: dict[int, dict[str, float]]) -> list[float]:
    return [
        values["train/steps_per_sec"]
        for step, values in sorted(metrics.items())
        if 100 <= step < 300 and "train/steps_per_sec" in values
    ]


def parse_gpu(path: pathlib.Path) -> dict[str, float | int]:
    memory: list[float] = []
    utilization: list[float] = []
    gpu_ids: set[int] = set()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            gpu_ids.add(int(row["index"]))
            memory.append(float(row["memory_used_mib"]))
            utilization.append(float(row["utilization_gpu_percent"]))
    return {
        "rows": len(memory),
        "gpu_ids": len(gpu_ids),
        "max_memory_mib": max(memory) if memory else 0.0,
        "median_utilization_percent": statistics.median(utilization)
        if utilization
        else 0.0,
    }


def relative_delta(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-log", type=pathlib.Path, required=True)
    parser.add_argument("--gpu-csv", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output-prefix", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if len(args.gpu_csv) != 2:
        raise SystemExit("exactly two --gpu-csv paths are required")

    baseline_metrics = parse_metrics(args.baseline_log)
    candidate_metrics = parse_metrics(args.candidate_log)
    baseline_steady = steady_throughput(baseline_metrics)
    candidate_steady = steady_throughput(candidate_metrics)
    if len(baseline_steady) < 20 or len(candidate_steady) < 20:
        raise SystemExit(
            "insufficient steady throughput samples: "
            f"baseline={len(baseline_steady)} candidate={len(candidate_steady)}"
        )

    baseline_median = statistics.median(baseline_steady)
    candidate_median = statistics.median(candidate_steady)
    speedup = candidate_median / baseline_median

    finite_metrics = all(
        math.isfinite(value)
        for values in candidate_metrics.values()
        for value in values.values()
    )
    parity_fields = (
        "train/loss",
        "train/audio_loss",
        "train/eos_loss",
        "train/void_loss",
    )
    parity_deltas = []
    for step in sorted(set(baseline_metrics) & set(candidate_metrics)):
        if not 100 <= step < 300:
            continue
        for field in parity_fields:
            left = baseline_metrics[step].get(field)
            right = candidate_metrics[step].get(field)
            if left is not None and right is not None:
                parity_deltas.append(relative_delta(left, right))
    if not parity_deltas:
        raise SystemExit("baseline/candidate loss parity has no matched metrics")
    worst_parity_delta = max(parity_deltas)

    gpu = [parse_gpu(path) for path in args.gpu_csv]
    max_memory_mib = max(row["max_memory_mib"] for row in gpu)
    gates = {
        "speedup_ge_1p10": speedup >= 1.10,
        "worst_loss_relative_delta_le_0p005": worst_parity_delta <= 0.005,
        "finite_candidate_metrics": finite_metrics,
        "max_memory_lt_70gib": max_memory_mib < 70 * 1024,
        "both_nodes_have_all_8_gpu_ids": all(
            row["rows"] >= 8 and row["gpu_ids"] == 8 for row in gpu
        ),
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    rows = [
        {
            "arm": "rdma_gc_on_baseline",
            "samples": len(baseline_steady),
            "steps_per_sec_median": baseline_median,
            "tokens_per_sec_median": baseline_median * GLOBAL_TOKENS_PER_STEP,
        },
        {
            "arm": "rdma_gc_off_candidate",
            "samples": len(candidate_steady),
            "steps_per_sec_median": candidate_median,
            "tokens_per_sec_median": candidate_median * GLOBAL_TOKENS_PER_STEP,
        },
    ]
    payload = {
        "verdict": verdict,
        "speedup": speedup,
        "baseline_steps_per_sec_median": baseline_median,
        "candidate_steps_per_sec_median": candidate_median,
        "baseline_tokens_per_sec_median": baseline_median * GLOBAL_TOKENS_PER_STEP,
        "candidate_tokens_per_sec_median": candidate_median * GLOBAL_TOKENS_PER_STEP,
        "worst_relative_loss_delta": worst_parity_delta,
        "max_memory_mib": max_memory_mib,
        "gpu_telemetry": gpu,
        "gates": gates,
    }

    prefix = str(args.output_prefix)
    pathlib.Path(prefix + ".SUMMARY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with pathlib.Path(prefix + ".SUMMARY.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pathlib.Path(prefix + ".VERDICT.txt").open("w") as stream:
        stream.write(f"VERDICT={verdict}\n")
        stream.write(f"speedup={speedup:.6f}\n")
        stream.write(f"baseline_steps_per_sec_median={baseline_median:.6f}\n")
        stream.write(f"candidate_steps_per_sec_median={candidate_median:.6f}\n")
        stream.write(f"worst_relative_loss_delta={worst_parity_delta:.8f}\n")
        stream.write(f"max_memory_mib={max_memory_mib:.0f}\n")
        for gate, passed in gates.items():
            stream.write(f"{gate}={str(passed).lower()}\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
