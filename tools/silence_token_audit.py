#!/usr/bin/env python3
"""Measure exact digital-silence runs in dumped generated audio tokens."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


SILENCE_FRAME_TOKENS = np.asarray(
    [244, 354, 998, 351, 433, 552, 926, 419], dtype=np.int64
)


def _longest_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _first_run(values: np.ndarray, run_frames: int) -> int | None:
    current = 0
    for index, value in enumerate(values):
        current = current + 1 if bool(value) else 0
        if current >= run_frames:
            return index - run_frames + 1
    return None


def _quantile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values), q))


def _read_meta(pattern: str) -> dict[str, dict]:
    rows = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                utt_id = row["utt_id"]
                if utt_id in rows:
                    raise ValueError(f"duplicate metadata for {utt_id}")
                rows[utt_id] = row
    return rows


def audit(
    token_dir: Path,
    metadata: dict[str, dict],
    widths: list[int],
    run_frames: int,
) -> dict:
    records = {
        width: {
            "any_match": 0,
            "run_ge_threshold": 0,
            "would_force_stop": 0,
            "longest_runs": [],
            "trimmed_frames": [],
            "would_force_stop_ids": [],
        }
        for width in widths
    }
    files = sorted(token_dir.glob("*.npy"))
    missing_meta = []
    eos_count = 0
    for path in files:
        utt_id = path.stem
        meta = metadata.get(utt_id)
        if meta is None:
            missing_meta.append(utt_id)
            continue
        tokens = np.load(path, allow_pickle=False)
        if tokens.ndim != 2 or tokens.shape[0] < max(widths):
            raise ValueError(f"bad token shape for {utt_id}: {tokens.shape}")
        if int(meta["frames"]) != tokens.shape[1]:
            raise ValueError(
                f"frame mismatch for {utt_id}: meta={meta['frames']} "
                f"tokens={tokens.shape[1]}"
            )
        start_frame = max(0, int(meta.get("min_gen_frames", 0)))
        eos_col = tokens.shape[1] if meta.get("eos") else None
        eos_count += int(eos_col is not None)
        for width in widths:
            matches = np.all(
                tokens[:width, start_frame:]
                == SILENCE_FRAME_TOKENS[:width, None],
                axis=0,
            )
            longest = _longest_run(matches)
            first = _first_run(matches, run_frames)
            record = records[width]
            record["any_match"] += int(bool(matches.any()))
            record["run_ge_threshold"] += int(longest >= run_frames)
            record["longest_runs"].append(longest)
            if first is None:
                continue
            run_start = start_frame + first
            trigger_col = run_start + run_frames - 1
            # The live decoder gives EOS priority when it appears no later
            # than the final qualifying silence frame.
            would_force = eos_col is None or trigger_col < eos_col
            if would_force:
                record["would_force_stop"] += 1
                record["trimmed_frames"].append(tokens.shape[1] - run_start)
                record["would_force_stop_ids"].append(utt_id)

    summary = {
        "token_dir": str(token_dir),
        "run_frames": run_frames,
        "files": len(files),
        "audited": len(files) - len(missing_meta),
        "eos_count": eos_count,
        "missing_meta": missing_meta,
        "widths": {},
    }
    for width, record in records.items():
        longest = record.pop("longest_runs")
        trimmed = record["trimmed_frames"]
        summary["widths"][str(width)] = {
            **record,
            "max_run": max(longest, default=0),
            "run_p95": _quantile(longest, 0.95),
            "trimmed_frames_median": _quantile(trimmed, 0.5),
            "trimmed_frames_p95": _quantile(trimmed, 0.95),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--meta-glob", required=True)
    parser.add_argument("--widths", default="1,2,3,4,8")
    parser.add_argument("--run-frames", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    widths = [int(value) for value in args.widths.split(",")]
    if args.run_frames < 1:
        raise ValueError("--run-frames must be >= 1")
    if not widths or min(widths) < 1 or max(widths) > 8:
        raise ValueError("--widths must contain values in [1, 8]")
    summary = audit(
        args.token_dir,
        _read_meta(args.meta_glob),
        widths,
        args.run_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
