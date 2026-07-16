#!/usr/bin/env python3
"""Fail-closed audit for paired Seed-TTS silence force-stop runs.

The generator is intentionally resume-safe.  That is useful for exploratory
runs, but it means a verdict job must independently prove that every waveform
and score belongs to the current, paired control/forced run.  This helper only
reads artifacts and exits non-zero on any missing, duplicate, stale, or
cross-arm ID.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


class AuditError(ValueError):
    """Raised when a run artifact violates the paired-run contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _id_diff(expected: set[str], actual: set[str]) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return f"missing={missing[:8]} extra={extra[:8]}"


def _unique_ids(ids: list[str], source: str) -> set[str]:
    _require(all(ids), f"{source}: empty ID")
    unique = set(ids)
    _require(len(unique) == len(ids), f"{source}: duplicate IDs")
    return unique


def _load_dataset(path: Path) -> tuple[list[str], dict[str, str]]:
    _require(path.is_file(), f"dataset TSV missing: {path}")
    ids: list[str] = []
    target_text: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for line_no, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not row:
                continue
            _require(
                len(row) >= 4,
                f"{path}:{line_no}: expected at least four TSV columns",
            )
            utt_id = row[0]
            ids.append(utt_id)
            target_text[utt_id] = row[3]
    _require(ids, f"dataset TSV is empty: {path}")
    _unique_ids(ids, str(path))
    return ids, target_text


def _load_test_list(
    path: Path, expected_ids: set[str], target_text: dict[str, str]
) -> None:
    _require(path.is_file(), f"test JSONL missing: {path}")
    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            _require(bool(line), f"{path}:{line_no}: blank JSONL row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(f"{path}:{line_no}: invalid JSON: {error}") from error
            utt_id = row.get("id")
            _require(isinstance(utt_id, str), f"{path}:{line_no}: invalid id")
            ids.append(utt_id)
            _require(
                row.get("text") == target_text.get(utt_id),
                f"{path}:{line_no}: target text differs from dataset for {utt_id}",
            )
            ref_audio = row.get("ref_audio")
            _require(
                isinstance(ref_audio, str) and Path(ref_audio).is_file(),
                f"{path}:{line_no}: reference audio missing for {utt_id}: {ref_audio}",
            )
    actual_ids = _unique_ids(ids, str(path))
    _require(
        actual_ids == expected_ids,
        f"test JSONL IDs differ from dataset: {_id_diff(expected_ids, actual_ids)}",
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            _require(bool(line), f"{path}:{line_no}: blank JSONL row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(f"{path}:{line_no}: invalid JSON: {error}") from error
            _require(isinstance(row, dict), f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


_FINISHED_RE = re.compile(
    r"^\[shard (?P<index>\d+)/(?P<count>\d+)\] FINISHED "
    r"total=(?P<total>\d+) done=(?P<done>\d+) "
    r"skip=(?P<skip>\d+) fail=(?P<fail>\d+)$"
)


def _audit_arm(
    *,
    arm: str,
    lang: str,
    output_dir: Path,
    logs_dir: Path,
    dataset_ids: list[str],
    num_shards: int,
    match_codebooks: int,
) -> dict[str, dict[str, Any]]:
    _require(output_dir.is_dir(), f"{arm}: output directory missing: {output_dir}")
    expected = set(dataset_ids)

    wav_paths = sorted(output_dir.glob("*.wav"))
    wav_ids = _unique_ids([path.stem for path in wav_paths], f"{arm} wavs")
    _require(
        wav_ids == expected,
        f"{arm}: wav IDs differ from dataset: {_id_diff(expected, wav_ids)}",
    )

    expected_meta_names = {f"gen_meta_shard{i}.jsonl" for i in range(num_shards)}
    actual_meta_names = {path.name for path in output_dir.glob("gen_meta_shard*.jsonl")}
    _require(
        actual_meta_names == expected_meta_names,
        f"{arm}: metadata shard files differ: "
        f"{_id_diff(expected_meta_names, actual_meta_names)}",
    )
    expected_failure_names = {
        f"failures_shard{i}.jsonl" for i in range(num_shards)
    }
    actual_failure_names = {
        path.name for path in output_dir.glob("failures_shard*.jsonl")
    }
    _require(
        actual_failure_names == expected_failure_names,
        f"{arm}: failure shard files differ: "
        f"{_id_diff(expected_failure_names, actual_failure_names)}",
    )

    rows_by_id: dict[str, dict[str, Any]] = {}
    for shard in range(num_shards):
        expected_shard = set(dataset_ids[shard::num_shards])
        meta_path = output_dir / f"gen_meta_shard{shard}.jsonl"
        failure_path = output_dir / f"failures_shard{shard}.jsonl"
        _require(
            failure_path.stat().st_size == 0,
            f"{arm}: non-empty failure JSONL: {failure_path}",
        )

        meta_rows = _load_jsonl(meta_path)
        shard_ids: list[str] = []
        for row in meta_rows:
            utt_id = row.get("utt_id")
            _require(isinstance(utt_id, str), f"{meta_path}: invalid utt_id")
            shard_ids.append(utt_id)
            _require(utt_id not in rows_by_id, f"{arm}: duplicate metadata ID {utt_id}")
            rows_by_id[utt_id] = row
        actual_shard = _unique_ids(shard_ids, str(meta_path)) if shard_ids else set()
        _require(
            actual_shard == expected_shard,
            f"{arm}: shard {shard} metadata IDs differ: "
            f"{_id_diff(expected_shard, actual_shard)}",
        )

        log_path = logs_dir / f"gen_{lang}_{arm}_shard{shard}.log"
        _require(log_path.is_file(), f"{arm}: shard log missing: {log_path}")
        finished = []
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _FINISHED_RE.fullmatch(line)
            if match:
                finished.append({key: int(value) for key, value in match.groupdict().items()})
        _require(len(finished) == 1, f"{arm}: expected one FINISHED line in {log_path}")
        record = finished[0]
        expected_count = len(expected_shard)
        _require(
            record
            == {
                "index": shard,
                "count": num_shards,
                "total": expected_count,
                "done": expected_count,
                "skip": 0,
                "fail": 0,
            },
            f"{arm}: abnormal shard completion in {log_path}: {record}",
        )

    actual_meta_ids = set(rows_by_id)
    _require(
        actual_meta_ids == expected,
        f"{arm}: metadata IDs differ from dataset: "
        f"{_id_diff(expected, actual_meta_ids)}",
    )

    for utt_id, row in rows_by_id.items():
        silence_frames = row.get("silence_run_frames")
        _require(
            isinstance(silence_frames, int),
            f"{arm}: {utt_id} has invalid silence_run_frames",
        )
        _require(
            row.get("silence_match_codebooks") == match_codebooks,
            f"{arm}: {utt_id} has unexpected silence_match_codebooks",
        )
        if arm == "control":
            _require(silence_frames == 0, f"control: detector enabled for {utt_id}")
            _require(not row.get("silence_stop"), f"control: silence stop for {utt_id}")
        else:
            _require(silence_frames > 0, f"forced: detector disabled for {utt_id}")
            _require(
                bool(row.get("silence_stop")) == (row.get("stop_reason") == "silence"),
                f"forced: inconsistent stop metadata for {utt_id}",
            )
    return rows_by_id


def _score_ids(path: Path, kind: str, output_dir: Path) -> set[str]:
    _require(path.is_file(), f"{kind} score TSV missing: {path}")
    ids: list[str] = []
    output_dir = output_dir.resolve()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle, delimiter="\t")
        try:
            header = next(rows)
        except StopIteration as error:
            raise AuditError(f"empty score TSV: {path}") from error
        if kind == "wer":
            _require(header[:2] == ["Name", "WER"], f"invalid WER header: {path}")
        else:
            _require(
                header[:2] == ["Prompt-path", "Eval-path"],
                f"invalid SIM header: {path}",
            )

        for row in rows:
            if kind == "wer":
                eval_path = row[0] if row and row[0].endswith(".wav") else None
            elif len(row) == 3 and row[1].endswith(".wav"):
                eval_path = row[1]
            elif len(row) == 4 and row[2].endswith(".wav"):
                eval_path = row[2]
            else:
                eval_path = None
            if eval_path is None:
                continue
            resolved = Path(eval_path).resolve()
            _require(
                resolved.parent == output_dir,
                f"{kind}: score path belongs to another output directory: {eval_path}",
            )
            ids.append(resolved.stem)
    return _unique_ids(ids, str(path))


def _bucket_frames(frames: int) -> str:
    if frames <= 64:
        return "le64"
    if frames <= 128:
        return "65_128"
    if frames <= 256:
        return "129_256"
    return "gt256"


def _bucket_chars(text: str) -> str:
    length = len(text)
    if length <= 20:
        return "le20"
    if length <= 50:
        return "21_50"
    if length <= 100:
        return "51_100"
    return "gt100"


def _stratify(
    dataset_ids: list[str],
    target_text: dict[str, str],
    control: dict[str, dict[str, Any]],
    forced: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    strata: dict[str, dict[str, list[bool]]] = {
        "control_stop_reason": defaultdict(list),
        "control_frames": defaultdict(list),
        "target_chars": defaultdict(list),
    }
    deltas: list[int] = []
    triggered_deltas: list[int] = []
    trigger_onsets: list[int] = []
    trigger_observed: list[int] = []
    triggered_ids: list[str] = []
    for utt_id in dataset_ids:
        control_row = control[utt_id]
        forced_row = forced[utt_id]
        for field in ("min_gen_frames", "seed_frames"):
            _require(
                control_row.get(field) == forced_row.get(field),
                f"paired metadata differs for {utt_id}: {field}",
            )
        control_frames = control_row.get("frames")
        forced_frames = forced_row.get("frames")
        _require(
            isinstance(control_frames, int) and isinstance(forced_frames, int),
            f"paired metadata has invalid frame count for {utt_id}",
        )
        triggered = bool(forced_row.get("silence_stop"))
        delta = control_frames - forced_frames
        deltas.append(delta)
        if triggered:
            triggered_ids.append(utt_id)
            triggered_deltas.append(delta)
            onset = forced_row.get("silence_col")
            observed = forced_row.get("silence_trigger_col")
            _require(
                isinstance(onset, int) and isinstance(observed, int) and observed >= onset,
                f"forced: invalid trigger columns for {utt_id}",
            )
            trigger_onsets.append(onset)
            trigger_observed.append(observed)
        strata["control_stop_reason"][str(control_row.get("stop_reason"))].append(
            triggered
        )
        strata["control_frames"][_bucket_frames(control_frames)].append(triggered)
        strata["target_chars"][_bucket_chars(target_text[utt_id])].append(triggered)

    summary_strata: dict[str, dict[str, dict[str, float | int]]] = {}
    for dimension, values in strata.items():
        summary_strata[dimension] = {}
        for name, flags in sorted(values.items()):
            count = len(flags)
            hits = sum(flags)
            summary_strata[dimension][name] = {
                "n": count,
                "triggered": hits,
                "rate": hits / count,
            }

    return {
        "triggered": len(triggered_ids),
        "trigger_rate": len(triggered_ids) / len(dataset_ids),
        "triggered_ids": triggered_ids,
        "frame_delta_median": statistics.median(deltas),
        "triggered_frame_delta_median": (
            statistics.median(triggered_deltas) if triggered_deltas else None
        ),
        "trigger_onset_median": (
            statistics.median(trigger_onsets) if trigger_onsets else None
        ),
        "trigger_observed_median": (
            statistics.median(trigger_observed) if trigger_observed else None
        ),
        "strata": summary_strata,
    }


def audit_run(
    *,
    lang: str,
    dataset_tsv: Path,
    test_jsonl: Path,
    control_dir: Path,
    forced_dir: Path,
    logs_dir: Path,
    num_shards: int,
    match_codebooks: int,
    score_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    _require(num_shards > 0, "num_shards must be positive")
    dataset_ids, target_text = _load_dataset(dataset_tsv)
    expected = set(dataset_ids)
    _load_test_list(test_jsonl, expected, target_text)
    control = _audit_arm(
        arm="control",
        lang=lang,
        output_dir=control_dir,
        logs_dir=logs_dir,
        dataset_ids=dataset_ids,
        num_shards=num_shards,
        match_codebooks=match_codebooks,
    )
    forced = _audit_arm(
        arm="forced",
        lang=lang,
        output_dir=forced_dir,
        logs_dir=logs_dir,
        dataset_ids=dataset_ids,
        num_shards=num_shards,
        match_codebooks=match_codebooks,
    )

    summary: dict[str, Any] = {
        "lang": lang,
        "expected_ids": len(dataset_ids),
        "control_wavs": len(dataset_ids),
        "forced_wavs": len(dataset_ids),
        "control_meta": len(control),
        "forced_meta": len(forced),
        "trigger": _stratify(dataset_ids, target_text, control, forced),
    }
    if score_paths is not None:
        expected_keys = {"wer_control", "wer_forced", "sim_control", "sim_forced"}
        _require(
            set(score_paths) == expected_keys,
            f"score paths must be exactly {sorted(expected_keys)}",
        )
        score_summary: dict[str, int] = {}
        for arm, output_dir in (("control", control_dir), ("forced", forced_dir)):
            for kind in ("wer", "sim"):
                key = f"{kind}_{arm}"
                actual = _score_ids(score_paths[key], kind, output_dir)
                _require(
                    actual == expected,
                    f"{key}: score IDs differ from dataset: "
                    f"{_id_diff(expected, actual)}",
                )
                score_summary[key] = len(actual)
        summary["scores"] = score_summary
    return summary


def _print_summary(summary: dict[str, Any], complete: bool) -> None:
    trigger = summary["trigger"]
    score_text = ""
    if complete:
        score_text = " " + " ".join(
            f"{name}={count}" for name, count in sorted(summary["scores"].items())
        )
    print(
        f"AUDIT_{'COMPLETE' if complete else 'GENERATION'} "
        f"lang={summary['lang']} ids={summary['expected_ids']} "
        f"control_wavs={summary['control_wavs']} forced_wavs={summary['forced_wavs']} "
        f"control_meta={summary['control_meta']} forced_meta={summary['forced_meta']} "
        f"triggered={trigger['triggered']} rate={trigger['trigger_rate']:.6f}"
        f"{score_text}"
    )
    for dimension, values in trigger["strata"].items():
        for name, row in values.items():
            print(
                f"TRIGGER_STRATUM lang={summary['lang']} dimension={dimension} "
                f"value={name} n={row['n']} triggered={row['triggered']} "
                f"rate={row['rate']:.6f}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, choices=("zh", "en"))
    parser.add_argument("--dataset-tsv", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--forced-dir", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--match-codebooks", type=int, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--wer-control", type=Path)
    parser.add_argument("--wer-forced", type=Path)
    parser.add_argument("--sim-control", type=Path)
    parser.add_argument("--sim-forced", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    optional_scores = {
        "wer_control": args.wer_control,
        "wer_forced": args.wer_forced,
        "sim_control": args.sim_control,
        "sim_forced": args.sim_forced,
    }
    present = [value is not None for value in optional_scores.values()]
    if any(present) and not all(present):
        print("AUDIT_FAILED: provide all four score TSV paths or none", file=sys.stderr)
        return 2
    score_paths = optional_scores if all(present) else None
    try:
        summary = audit_run(
            lang=args.lang,
            dataset_tsv=args.dataset_tsv,
            test_jsonl=args.test_jsonl,
            control_dir=args.control_dir,
            forced_dir=args.forced_dir,
            logs_dir=args.logs_dir,
            num_shards=args.num_shards,
            match_codebooks=args.match_codebooks,
            score_paths=score_paths,
        )
    except (AuditError, OSError) as error:
        print(f"AUDIT_FAILED: {error}", file=sys.stderr)
        return 1
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_summary(summary, score_paths is not None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
