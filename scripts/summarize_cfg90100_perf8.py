#!/usr/bin/env python3
"""Summarize and gate the four-arm CFG90100 eight-GPU performance matrix."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics


ARMS = (
    "base",
    "nogc",
    "nogc_bal8",
    "small_nogc",
    "small_nogc_bal8",
    "base_replay",
)
BATCH_TOKENS = {
    "base": 15648,
    "nogc": 15648,
    "nogc_bal8": 15648,
    "small_nogc": 7824,
    "small_nogc_bal8": 7824,
    "base_replay": 15648,
}
METRIC_RE = re.compile(r"^Step (\d+) \| (.*)$")


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


def parse_gpu(path: pathlib.Path) -> tuple[float, float]:
    memory: list[float] = []
    utilization: list[float] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            memory.append(float(row["memory_used_mib"]))
            utilization.append(float(row["utilization_gpu_percent"]))
    if not memory:
        raise ValueError(f"no GPU samples in {path}")
    return max(memory), statistics.median(utilization)


def relative_delta(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=pathlib.Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()

    rows = []
    metrics_by_arm: dict[str, dict[int, dict[str, float]]] = {}
    for arm in ARMS:
        metrics = parse_metrics(run_root / "logs" / f"{arm}.log")
        metrics_by_arm[arm] = metrics
        steady = [
            values["train/steps_per_sec"]
            for step, values in sorted(metrics.items())
            if 100 <= step < 300 and "train/steps_per_sec" in values
        ]
        if len(steady) < 20:
            raise SystemExit(
                f"arm={arm} has only {len(steady)} steady-state throughput samples"
            )
        max_memory, median_utilization = parse_gpu(
            run_root / "gpu" / f"{arm}.csv"
        )
        rows.append(
            {
                "arm": arm,
                "samples": len(steady),
                "batch_tokens_per_gpu": BATCH_TOKENS[arm],
                "global_tokens_per_step": 8 * BATCH_TOKENS[arm],
                "steps_per_sec_median": statistics.median(steady),
                "steps_per_sec_mean": statistics.fmean(steady),
                "steps_per_sec_min": min(steady),
                "steps_per_sec_max": max(steady),
                "tokens_per_sec_median": (
                    statistics.median(steady) * 8 * BATCH_TOKENS[arm]
                ),
                "max_memory_mib": max_memory,
                "median_gpu_util_percent": median_utilization,
            }
        )

    by_arm = {row["arm"]: row for row in rows}
    baseline_reference = statistics.fmean(
        (
            by_arm["base"]["steps_per_sec_median"],
            by_arm["base_replay"]["steps_per_sec_median"],
        )
    )
    baseline_drift = relative_delta(
        by_arm["base"]["steps_per_sec_median"],
        by_arm["base_replay"]["steps_per_sec_median"],
    )
    gc_speedup = by_arm["nogc"]["steps_per_sec_median"] / baseline_reference
    balanced_speedup = (
        by_arm["nogc_bal8"]["steps_per_sec_median"]
        / by_arm["nogc"]["steps_per_sec_median"]
    )
    small_balanced_speedup = (
        by_arm["small_nogc_bal8"]["steps_per_sec_median"]
        / by_arm["small_nogc"]["steps_per_sec_median"]
    )

    parity_fields = (
        "train/loss",
        "train/audio_loss",
        "train/eos_loss",
        "train/void_loss",
    )
    parity_deltas = []
    for step in sorted(
        set(metrics_by_arm["base"]) & set(metrics_by_arm["nogc"])
    ):
        if step > 300:
            continue
        for field in parity_fields:
            left = metrics_by_arm["base"][step].get(field)
            right = metrics_by_arm["nogc"][step].get(field)
            if left is not None and right is not None:
                parity_deltas.append(relative_delta(left, right))
    if not parity_deltas:
        raise SystemExit("base/nogc loss parity has no matched metrics")
    worst_parity_delta = max(parity_deltas)

    max_memory_mib = max(row["max_memory_mib"] for row in rows)
    integrity_gates = {
        "base_nogc_worst_rel_le_0p005": worst_parity_delta <= 0.005,
        "max_memory_lt_70gib": max_memory_mib < 70 * 1024,
        "baseline_replay_drift_le_0p03": baseline_drift <= 0.03,
    }
    verdict = "PASS" if all(integrity_gates.values()) else "FAIL"
    gc_candidate_pass = verdict == "PASS" and gc_speedup >= 1.10
    balanced_candidate_pass = verdict == "PASS" and balanced_speedup >= 1.05
    small_balanced_candidate_pass = (
        verdict == "PASS" and small_balanced_speedup >= 1.05
    )
    payload = {
        "verdict": verdict,
        "baseline_reference_steps_per_sec": baseline_reference,
        "baseline_replay_relative_drift": baseline_drift,
        "gc_speedup": gc_speedup,
        "balanced_speedup": balanced_speedup,
        "small_balanced_speedup": small_balanced_speedup,
        "gc_candidate_pass": gc_candidate_pass,
        "balanced_candidate_pass": balanced_candidate_pass,
        "small_balanced_candidate_pass": small_balanced_candidate_pass,
        "base_nogc_worst_relative_loss_delta": worst_parity_delta,
        "max_memory_mib": max_memory_mib,
        "integrity_gates": integrity_gates,
        "arms": rows,
    }

    (run_root / "SUMMARY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with (run_root / "SUMMARY.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with (run_root / "VERDICT.txt").open("w") as stream:
        stream.write(f"VERDICT={verdict}\n")
        stream.write(f"baseline_reference_steps_per_sec={baseline_reference:.6f}\n")
        stream.write(f"baseline_replay_relative_drift={baseline_drift:.8f}\n")
        stream.write(f"gc_speedup={gc_speedup:.6f}\n")
        stream.write(f"balanced_speedup={balanced_speedup:.6f}\n")
        stream.write(f"small_balanced_speedup={small_balanced_speedup:.6f}\n")
        stream.write(f"base_nogc_worst_relative_loss_delta={worst_parity_delta:.8f}\n")
        stream.write(f"max_memory_mib={max_memory_mib:.0f}\n")
        stream.write(f"gc_candidate_pass={str(gc_candidate_pass).lower()}\n")
        stream.write(f"balanced_candidate_pass={str(balanced_candidate_pass).lower()}\n")
        stream.write(
            "small_balanced_candidate_pass="
            f"{str(small_balanced_candidate_pass).lower()}\n"
        )
        for gate, passed in integrity_gates.items():
            stream.write(f"{gate}={str(passed).lower()}\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
