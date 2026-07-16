#!/usr/bin/env python3
"""Validate and summarize paired Seed-TTS first-300 decode sweeps."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import itertools
import json
import math
import re
import statistics
import sys
import wave
from pathlib import Path
from typing import Any, Iterable


def fail(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return sorted(repeated)


def require_unique(name: str, values: list[str]) -> None:
    repeated = duplicates(values)
    if repeated:
        fail(f"{name} has duplicate IDs: {repeated[:10]}")


def diff_ids(expected: list[str], actual: list[str]) -> str:
    expected_set = set(expected)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    return f"missing={missing[:10]} extra={extra[:10]}"


def require_exact_ids(name: str, expected: list[str], actual: list[str]) -> None:
    require_unique(name, actual)
    if set(expected) != set(actual) or len(expected) != len(actual):
        fail(f"{name} ID mismatch: {diff_ids(expected, actual)}")


def read_source_rows(
    tsv_path: Path,
    jsonl_path: Path,
    expected_count: int,
    check_prompt_wavs: bool = False,
    check_ref_audio: bool = False,
) -> tuple[list[str], list[list[str]], list[dict[str, Any]]]:
    with tsv_path.open(newline="", encoding="utf-8") as handle:
        tsv_rows = [row for row in csv.reader(handle, delimiter="\t") if row]
    if any(len(row) < 4 for row in tsv_rows):
        fail(f"{tsv_path} contains rows with fewer than four columns")
    tsv_ids = [row[0] for row in tsv_rows]
    require_unique(f"source TSV {tsv_path}", tsv_ids)

    json_rows = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{jsonl_path}:{line_number}: invalid JSON: {exc}")
            if not isinstance(row, dict) or not row.get("id"):
                fail(f"{jsonl_path}:{line_number}: missing id")
            json_rows.append(row)
    json_ids = [str(row["id"]) for row in json_rows]
    require_unique(f"source JSONL {jsonl_path}", json_ids)

    if len(tsv_ids) != expected_count or len(json_ids) != expected_count:
        fail(
            "source count mismatch: "
            f"expected={expected_count} tsv={len(tsv_ids)} jsonl={len(json_ids)}"
        )
    if tsv_ids != json_ids:
        if set(tsv_ids) == set(json_ids):
            first = next(
                index
                for index, pair in enumerate(zip(tsv_ids, json_ids, strict=True))
                if pair[0] != pair[1]
            )
            fail(
                "source TSV/JSONL order mismatch at "
                f"index={first}: tsv={tsv_ids[first]!r} jsonl={json_ids[first]!r}"
            )
        fail(f"source TSV/JSONL ID mismatch: {diff_ids(tsv_ids, json_ids)}")

    if check_prompt_wavs:
        prompt_dir = tsv_path.parent / "prompt_wavs"
        missing_prompts = [
            str(prompt_dir / Path(row[2]).name)
            for row in tsv_rows
            if not (prompt_dir / Path(row[2]).name).is_file()
        ]
        if missing_prompts:
            fail(f"missing prompt wavs: {missing_prompts[:10]}")

    if check_ref_audio:
        missing_refs = [
            str(row.get("ref_audio"))
            for row in json_rows
            if not row.get("ref_audio") or not Path(row["ref_audio"]).is_file()
        ]
        if missing_refs:
            fail(f"missing JSONL reference audio: {missing_refs[:10]}")

    return tsv_ids, tsv_rows, json_rows


def validate_inputs(args: argparse.Namespace) -> None:
    tsv = Path(args.tsv).resolve()
    jsonl = Path(args.jsonl).resolve()
    ids, _, _ = read_source_rows(
        tsv,
        jsonl,
        args.expected_count,
        check_prompt_wavs=args.check_prompt_wavs,
        check_ref_audio=args.check_ref_audio,
    )
    ids_out = Path(args.ids_out)
    manifest_out = Path(args.manifest_out)
    ids_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    ids_out.write_text("".join(f"{utt_id}\n" for utt_id in ids), encoding="utf-8")
    manifest = {
        "count": len(ids),
        "first_id": ids[0],
        "last_id": ids[-1],
        "tsv": str(tsv),
        "tsv_sha256": sha256(tsv),
        "jsonl": str(jsonl),
        "jsonl_sha256": sha256(jsonl),
        "ordered_ids_sha256": hashlib.sha256(
            "".join(f"{utt_id}\n" for utt_id in ids).encode()
        ).hexdigest(),
    }
    manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"INPUT_OK count={len(ids)} ids_sha256={manifest['ordered_ids_sha256']} "
        f"tsv={tsv} jsonl={jsonl}"
    )


def parse_wer(path: Path) -> tuple[dict[str, dict[str, Any]], float]:
    results: dict[str, dict[str, Any]] = {}
    text = path.read_text(encoding="utf-8")
    official_match = re.search(
        r"Seed-TTS WER \(Avg of WERs\):\s*([0-9.]+)%", text
    )
    if not official_match:
        fail(f"{path} lacks the official Seed-TTS WER footer")
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 7 or Path(row[0]).suffix.lower() != ".wav":
                continue
            utt_id = Path(row[0]).stem
            if utt_id in results:
                fail(f"{path} has duplicate WER ID {utt_id}")
            try:
                wer = float(row[1])
                insertions, deletions, substitutions = map(float, row[-3:])
            except ValueError as exc:
                fail(f"{path}: invalid WER row for {utt_id}: {exc}")
            results[utt_id] = {
                "wer": wer,
                "truth": row[2],
                "hypothesis": row[3],
                "insertions": insertions,
                "deletions": deletions,
                "substitutions": substitutions,
            }
    if not results:
        fail(f"{path} has no per-utterance WER rows")
    computed = statistics.mean(row["wer"] for row in results.values()) * 100
    official = float(official_match.group(1))
    if not math.isclose(computed, official, abs_tol=0.011):
        fail(
            f"{path} WER footer mismatch: computed={computed:.6f} official={official:.6f}"
        )
    return results, official


def parse_sim(path: Path) -> tuple[dict[str, float], float, float]:
    results: dict[str, float] = {}
    text = path.read_text(encoding="utf-8")
    official_match = re.search(r"Average SIM-o:\s*([-+0-9.eE]+)", text)
    if not official_match:
        fail(f"{path} lacks the official Average SIM-o footer")
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 3 or Path(row[-2]).suffix.lower() != ".wav":
                continue
            utt_id = Path(row[-2]).stem
            if utt_id in results:
                fail(f"{path} has duplicate SIM ID {utt_id}")
            try:
                results[utt_id] = float(row[-1])
            except ValueError as exc:
                fail(f"{path}: invalid SIM row for {utt_id}: {exc}")
    if not results:
        fail(f"{path} has no per-utterance SIM rows")
    rounded_mean = statistics.mean(results.values())
    official = float(official_match.group(1))
    if not math.isclose(rounded_mean, official, abs_tol=0.011):
        fail(
            f"{path} SIM footer mismatch: "
            f"rounded_per_utt_mean={rounded_mean:.6f} official={official:.6f}"
        )
    return results, official, rounded_mean


def parse_meta(meta_pattern: str, failure_pattern: str) -> dict[str, dict[str, Any]]:
    failure_paths = [Path(path) for path in sorted(glob.glob(failure_pattern))]
    nonempty_failures = [str(path) for path in failure_paths if path.read_text().strip()]
    if nonempty_failures:
        fail(f"generation failure ledgers are nonempty: {nonempty_failures}")

    meta_paths = [Path(path) for path in sorted(glob.glob(meta_pattern))]
    if not meta_paths:
        fail(f"no generation metadata matched {meta_pattern!r}")
    results: dict[str, dict[str, Any]] = {}
    for path in meta_paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    fail(f"{path}:{line_number}: invalid JSON: {exc}")
                utt_id = str(row.get("utt_id", ""))
                if not utt_id:
                    fail(f"{path}:{line_number}: missing utt_id")
                if utt_id in results:
                    fail(f"generation metadata has duplicate ID {utt_id}")
                if row.get("silence_stop"):
                    fail(f"silence-stop fired while fixed off: id={utt_id}")
                if row.get("silence_run_frames") != 0:
                    fail(
                        "silence-stop was not disabled in generation metadata: "
                        f"id={utt_id} silence_run_frames={row.get('silence_run_frames')!r}"
                    )
                results[utt_id] = row
    return results


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        duration = handle.getnframes() / handle.getframerate()
    if not math.isfinite(duration) or duration <= 0:
        fail(f"invalid wav duration: {path} -> {duration}")
    return duration


def percentile95_like_campaign(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def summarize_arm(args: argparse.Namespace) -> dict[str, Any]:
    expected_ids, _, _ = read_source_rows(
        Path(args.tsv), Path(args.jsonl), args.expected_count
    )
    wav_dir = Path(args.wav_dir).resolve()
    anchor_dir = Path(args.anchor_dir).resolve()
    wav_ids = sorted(path.stem for path in wav_dir.glob("*.wav"))
    require_exact_ids("generated wavs", expected_ids, wav_ids)
    missing_anchors = [
        utt_id for utt_id in expected_ids if not (anchor_dir / f"{utt_id}.wav").is_file()
    ]
    if missing_anchors:
        fail(f"duration anchor missing IDs: {missing_anchors[:10]}")

    wer_rows, official_wer = parse_wer(Path(args.wer_tsv))
    sim_rows, official_sim, rounded_sim_mean = parse_sim(Path(args.sim_tsv))
    meta_rows = parse_meta(args.meta_glob, args.failures_glob)
    require_exact_ids("WER rows", expected_ids, list(wer_rows))
    require_exact_ids("SIM rows", expected_ids, list(sim_rows))
    require_exact_ids("generation metadata", expected_ids, list(meta_rows))

    output_rows: list[dict[str, Any]] = []
    ratios: list[float] = []
    for utt_id in expected_ids:
        duration = wav_duration(wav_dir / f"{utt_id}.wav")
        anchor_duration = wav_duration(anchor_dir / f"{utt_id}.wav")
        ratio = duration / anchor_duration
        ratios.append(ratio)
        wer = wer_rows[utt_id]
        meta = meta_rows[utt_id]
        output_rows.append(
            {
                "id": utt_id,
                "lang": args.lang,
                "guidance_scale": args.guidance_scale,
                "steps_per_block": args.steps_per_block,
                "wer": wer["wer"],
                "sim": sim_rows[utt_id],
                "duration_seconds": duration,
                "anchor_duration_seconds": anchor_duration,
                "duration_ratio": ratio,
                "runaway": int(wer["wer"] > 0.5),
                "lt_0_6": int(ratio < 0.6),
                "gt_2": int(ratio > 2.0),
                "frames": meta.get("frames"),
                "eos": int(bool(meta.get("eos"))),
                "silence_stop": int(bool(meta.get("silence_stop"))),
                "stop_reason": meta.get("stop_reason"),
                "n_blocks": meta.get("n_blocks"),
                "truth": wer["truth"],
                "hypothesis": wer["hypothesis"],
                "insertions": wer["insertions"],
                "deletions": wer["deletions"],
                "substitutions": wer["substitutions"],
            }
        )

    per_utt_path = Path(args.per_utt_out).resolve()
    per_utt_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with per_utt_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "arm": args.arm,
        "lang": args.lang,
        "guidance_scale": args.guidance_scale,
        "steps_per_block": args.steps_per_block,
        "count": len(output_rows),
        "wer_percent": official_wer,
        "sim": official_sim,
        "sim_per_utt_rounded_mean": rounded_sim_mean,
        "duration_ratio_mean": statistics.mean(ratios),
        "duration_ratio_median": statistics.median(ratios),
        "duration_ratio_p95": percentile95_like_campaign(ratios),
        "runaway": sum(row["runaway"] for row in output_rows),
        "lt_0_6": sum(row["lt_0_6"] for row in output_rows),
        "gt_2": sum(row["gt_2"] for row in output_rows),
        "eos_count": sum(row["eos"] for row in output_rows),
        "max_blocks_count": sum(
            row["stop_reason"] == "max_blocks" for row in output_rows
        ),
        "silence_stop_count": sum(row["silence_stop"] for row in output_rows),
        "per_utt_tsv": str(per_utt_path),
        "wav_dir": str(wav_dir),
        "anchor_dir": str(anchor_dir),
    }
    summary_path = Path(args.summary_out).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"ARM_OK arm={args.arm} lang={args.lang} n={summary['count']} "
        f"WER={summary['wer_percent']:.2f}% SIM={summary['sim']:.3f} "
        f"dur_mean={summary['duration_ratio_mean']:.3f} "
        f"dur_median={summary['duration_ratio_median']:.3f} "
        f"dur_p95={summary['duration_ratio_p95']:.3f} "
        f"runaway={summary['runaway']} lt0.6={summary['lt_0_6']} "
        f"gt2={summary['gt_2']}"
    )
    return summary


SUMMARY_FIELDS = [
    "sim_rank",
    "arm",
    "lang",
    "guidance_scale",
    "steps_per_block",
    "count",
    "wer_percent",
    "sim",
    "duration_ratio_mean",
    "duration_ratio_median",
    "duration_ratio_p95",
    "runaway",
    "lt_0_6",
    "gt_2",
    "eos_count",
    "max_blocks_count",
    "silence_stop_count",
    "per_utt_tsv",
]


def parse_csv_values(raw: str, transform: Any = str) -> list[Any]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        fail(f"empty expected value list: {raw!r}")
    return [transform(value) for value in values]


def aggregate(args: argparse.Namespace) -> None:
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.summaries]
    languages = parse_csv_values(args.expected_languages)
    guidances = parse_csv_values(args.expected_guidance, float)
    steps_values = parse_csv_values(args.expected_steps, int)
    expected = set(itertools.product(languages, guidances, steps_values))
    actual_list = [
        (row["lang"], float(row["guidance_scale"]), int(row["steps_per_block"]))
        for row in rows
    ]
    repeated = duplicates([repr(item) for item in actual_list])
    if repeated:
        fail(f"duplicate arm summaries: {repeated}")
    actual = set(actual_list)
    if actual != expected:
        fail(
            "aggregate arm matrix mismatch: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    if any(row["count"] != args.expected_count for row in rows):
        bad = [(row["arm"], row["lang"], row["count"]) for row in rows]
        fail(f"aggregate count mismatch: expected={args.expected_count} rows={bad}")
    if any(row["silence_stop_count"] != 0 for row in rows):
        fail("aggregate contains a silence-stop event although the sweep fixes it off")

    for lang in languages:
        ranked = sorted(
            (row for row in rows if row["lang"] == lang),
            key=lambda row: (-row["sim"], row["wer_percent"], row["steps_per_block"]),
        )
        for rank, row in enumerate(ranked, 1):
            row["sim_rank"] = rank
    rows.sort(key=lambda row: (languages.index(row["lang"]), row["sim_rank"]))

    output_tsv = Path(args.output_tsv).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    for path in (output_tsv, output_json, output_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in SUMMARY_FIELDS} for row in rows)

    best = {
        lang: next(row for row in rows if row["lang"] == lang and row["sim_rank"] == 1)
        for lang in languages
    }
    output_json.write_text(
        json.dumps({"rows": rows, "best_sim_by_language": best}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Decode sweep first-300 summary",
        "",
        "Silence force-stop is fixed off for every arm. SIM rank is within language.",
        "",
        "| SIM rank | lang | arm | guidance | steps | WER % | SIM | dur mean | dur median | dur p95 | runaway | <0.6 | >2 |",
        "|---:|:---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sim_rank']} | {row['lang']} | {row['arm']} | "
            f"{row['guidance_scale']:g} | {row['steps_per_block']} | "
            f"{row['wer_percent']:.2f} | {row['sim']:.3f} | "
            f"{row['duration_ratio_mean']:.3f} | "
            f"{row['duration_ratio_median']:.3f} | "
            f"{row['duration_ratio_p95']:.3f} | {row['runaway']} | "
            f"{row['lt_0_6']} | {row['gt_2']} |"
        )
    lines.extend(
        [
            "",
            *[
                f"- Best {lang} SIM: {best[lang]['arm']} "
                f"(SIM {best[lang]['sim']:.3f}, WER {best[lang]['wer_percent']:.2f}%)."
                for lang in languages
            ],
            "",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"SWEEP_OK arms={len(rows)} tsv={output_tsv} json={output_json} md={output_md}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-inputs")
    validate.add_argument("--tsv", required=True)
    validate.add_argument("--jsonl", required=True)
    validate.add_argument("--expected-count", required=True, type=int)
    validate.add_argument("--ids-out", required=True)
    validate.add_argument("--manifest-out", required=True)
    validate.add_argument("--check-prompt-wavs", action="store_true")
    validate.add_argument("--check-ref-audio", action="store_true")
    validate.set_defaults(function=validate_inputs)

    summarize = subparsers.add_parser("summarize-arm")
    summarize.add_argument("--arm", required=True)
    summarize.add_argument("--lang", required=True, choices=("zh", "en"))
    summarize.add_argument("--guidance-scale", required=True, type=float)
    summarize.add_argument("--steps-per-block", required=True, type=int)
    summarize.add_argument("--expected-count", required=True, type=int)
    summarize.add_argument("--tsv", required=True)
    summarize.add_argument("--jsonl", required=True)
    summarize.add_argument("--wav-dir", required=True)
    summarize.add_argument("--anchor-dir", required=True)
    summarize.add_argument("--wer-tsv", required=True)
    summarize.add_argument("--sim-tsv", required=True)
    summarize.add_argument("--meta-glob", required=True)
    summarize.add_argument("--failures-glob", required=True)
    summarize.add_argument("--per-utt-out", required=True)
    summarize.add_argument("--summary-out", required=True)
    summarize.set_defaults(function=summarize_arm)

    combine = subparsers.add_parser("aggregate")
    combine.add_argument("--summaries", nargs="+", required=True)
    combine.add_argument("--expected-languages", default="zh,en")
    combine.add_argument("--expected-guidance", default="1.5,2.0,3.0")
    combine.add_argument("--expected-steps", default="16,32")
    combine.add_argument("--expected-count", type=int, default=300)
    combine.add_argument("--output-tsv", required=True)
    combine.add_argument("--output-json", required=True)
    combine.add_argument("--output-md", required=True)
    combine.set_defaults(function=aggregate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.function(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
