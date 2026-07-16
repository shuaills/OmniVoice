#!/usr/bin/env python3
"""Prepare, validate, and report paired training-contract decode probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_decode_reporter() -> Any:
    path = Path(__file__).with_name("decode_sweep_report.py")
    spec = importlib.util.spec_from_file_location("decode_sweep_report_common", path)
    if spec is None or spec.loader is None:
        fail(f"cannot load decode reporter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load_decode_reporter()

TOKEN_DECODE_TIMING_FIELDS = {
    "token_decode_seconds",
    "token_decode_rtf",
    "timing_warmup",
}
TOKEN_DECODE_TIMING_SUMMARY_FIELDS = [
    "timed_count",
    "token_decode_seconds_mean",
    "token_decode_seconds_median",
    "token_decode_seconds_p95",
    "token_decode_rtf_mean",
    "token_decode_rtf_median",
    "token_decode_rtf_p95",
]


def summarize_token_decode_timing(
    meta: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate optional timing metadata and summarize non-warmup rows."""
    field_sets = [TOKEN_DECODE_TIMING_FIELDS.intersection(row) for row in meta.values()]
    if not any(field_sets):
        return None
    partial = {
        utt_id: sorted(TOKEN_DECODE_TIMING_FIELDS - row.keys())
        for utt_id, row in meta.items()
        if TOKEN_DECODE_TIMING_FIELDS - row.keys()
    }
    if partial:
        fail(
            "partial token decode timing metadata: "
            f"{dict(list(partial.items())[:5])}"
        )

    for utt_id, row in meta.items():
        for field in ("token_decode_seconds", "token_decode_rtf"):
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                fail(
                    "invalid token decode timing: "
                    f"id={utt_id} field={field} value={value!r}"
                )
        if not isinstance(row["timing_warmup"], bool):
            fail(
                "invalid token decode timing: "
                f"id={utt_id} field=timing_warmup "
                f"value={row['timing_warmup']!r}"
            )

    timed = [row for row in meta.values() if not row["timing_warmup"]]
    if not timed:
        fail("token decode timing has no post-warmup rows")
    seconds = [float(row["token_decode_seconds"]) for row in timed]
    rtfs = [float(row["token_decode_rtf"]) for row in timed]
    return {
        "timed_count": len(timed),
        "token_decode_seconds_mean": statistics.mean(seconds),
        "token_decode_seconds_median": statistics.median(seconds),
        "token_decode_seconds_p95": COMMON.percentile95_like_campaign(seconds),
        "token_decode_rtf_mean": statistics.mean(rtfs),
        "token_decode_rtf_median": statistics.median(rtfs),
        "token_decode_rtf_p95": COMMON.percentile95_like_campaign(rtfs),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{path}:{line_number}: invalid JSON: {exc}")
            if not isinstance(row, dict) or not row.get("id"):
                fail(f"{path}:{line_number}: missing id")
            rows.append(row)
    return rows


def prepare_inputs(args: argparse.Namespace) -> None:
    source_tsv = Path(args.source_tsv).resolve()
    source_jsonl = Path(args.source_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        fail(f"refusing to reuse input directory: {output_dir}")
    if args.count <= 0:
        fail(f"count must be positive, got {args.count}")

    with source_tsv.open(newline="", encoding="utf-8") as handle:
        tsv_rows = [row for row in csv.reader(handle, delimiter="\t") if row]
    json_rows = read_jsonl(source_jsonl)
    if len(tsv_rows) < args.count or len(json_rows) < args.count:
        fail(
            "source is shorter than requested subset: "
            f"count={args.count} tsv={len(tsv_rows)} jsonl={len(json_rows)}"
        )

    tsv_rows = tsv_rows[: args.count]
    json_rows = json_rows[: args.count]
    if any(len(row) < 4 for row in tsv_rows):
        fail("selected TSV contains rows with fewer than four columns")
    tsv_ids = [row[0] for row in tsv_rows]
    json_ids = [str(row["id"]) for row in json_rows]
    COMMON.require_unique("selected TSV", tsv_ids)
    COMMON.require_unique("selected JSONL", json_ids)
    if tsv_ids != json_ids:
        fail(
            "selected TSV/JSONL order mismatch: "
            f"{COMMON.diff_ids(tsv_ids, json_ids)}"
        )

    prompt_dir = source_tsv.parent / "prompt_wavs"
    if not prompt_dir.is_dir():
        fail(f"prompt directory missing: {prompt_dir}")
    missing_prompts = [
        str(prompt_dir / Path(row[2]).name)
        for row in tsv_rows
        if not (prompt_dir / Path(row[2]).name).is_file()
    ]
    if missing_prompts:
        fail(f"selected prompt wavs are missing: {missing_prompts[:10]}")
    missing_refs = [
        str(row.get("ref_audio"))
        for row in json_rows
        if not row.get("ref_audio") or not Path(row["ref_audio"]).is_file()
    ]
    if args.check_ref_audio and missing_refs:
        fail(f"selected JSONL reference audio is missing: {missing_refs[:10]}")
    anchor_dir = Path(args.anchor_dir).resolve()
    if not anchor_dir.is_dir():
        fail(f"duration anchor directory missing: {anchor_dir}")
    missing_anchors = [
        str(anchor_dir / f"{utt_id}.wav")
        for utt_id in tsv_ids
        if not (anchor_dir / f"{utt_id}.wav").is_file()
    ]
    if missing_anchors:
        fail(f"selected duration anchors are missing: {missing_anchors[:10]}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    output_tsv = output_dir / "test.tsv"
    output_jsonl = output_dir / "test.jsonl"
    with output_tsv.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(tsv_rows)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in json_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "prompt_wavs").symlink_to(prompt_dir.resolve(), target_is_directory=True)

    ordered_ids_sha256 = hashlib.sha256(
        "".join(f"{utt_id}\n" for utt_id in tsv_ids).encode()
    ).hexdigest()
    manifest = {
        "count": args.count,
        "first_id": tsv_ids[0],
        "last_id": tsv_ids[-1],
        "ordered_ids_sha256": ordered_ids_sha256,
        "source_tsv": str(source_tsv),
        "source_tsv_sha256": sha256(source_tsv),
        "source_jsonl": str(source_jsonl),
        "source_jsonl_sha256": sha256(source_jsonl),
        "prompt_dir": str(prompt_dir.resolve()),
        "prompt_wavs": [
            {
                "id": row[0],
                "path": str((prompt_dir / Path(row[2]).name).resolve()),
                "sha256": sha256(prompt_dir / Path(row[2]).name),
            }
            for row in tsv_rows
        ],
        "reference_audio": [
            {
                "id": str(row["id"]),
                "path": str(Path(row["ref_audio"]).resolve()),
                "sha256": sha256(Path(row["ref_audio"])),
            }
            for row in json_rows
        ],
        "duration_anchors": [
            {
                "id": utt_id,
                "path": str((anchor_dir / f"{utt_id}.wav").resolve()),
                "sha256": sha256(anchor_dir / f"{utt_id}.wav"),
            }
            for utt_id in tsv_ids
        ],
        "subset_tsv": str(output_tsv),
        "subset_tsv_sha256": sha256(output_tsv),
        "subset_jsonl": str(output_jsonl),
        "subset_jsonl_sha256": sha256(output_jsonl),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"INPUTS_READY count={args.count} first={tsv_ids[0]} last={tsv_ids[-1]} "
        f"ids_sha256={ordered_ids_sha256} output={output_dir}"
    )


def expected_shard_paths(wav_dir: Path, stem: str, count: int) -> set[Path]:
    return {wav_dir / f"{stem}{index}.jsonl" for index in range(count)}


def validate_generation(args: argparse.Namespace) -> None:
    if args.num_shards <= 0:
        fail(f"num_shards must be positive, got {args.num_shards}")
    expected_ids, _, _ = COMMON.read_source_rows(
        Path(args.tsv), Path(args.jsonl), args.expected_count
    )
    expected_seed_indices = {
        utt_id: index for index, utt_id in enumerate(expected_ids)
    }
    seed_index_map_path = None
    seed_index_map_arg = getattr(args, "seed_index_map", None)
    if seed_index_map_arg is not None:
        seed_index_map_path = Path(seed_index_map_arg).resolve()
        try:
            seed_index_map = json.loads(
                seed_index_map_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"invalid seed index map {seed_index_map_path}: {exc}")
        if not isinstance(seed_index_map, dict):
            fail(f"seed index map must be a JSON object: {seed_index_map_path}")
        COMMON.require_exact_ids(
            "seed index map", expected_ids, list(seed_index_map)
        )
        invalid_seed_indices = {
            key: value
            for key, value in seed_index_map.items()
            if not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        }
        if invalid_seed_indices:
            fail(
                "seed index map values must be non-negative integers: "
                f"{dict(list(invalid_seed_indices.items())[:5])}"
            )
        expected_seed_indices = seed_index_map
    wav_dir = Path(args.wav_dir).resolve()
    wav_ids = sorted(path.stem for path in wav_dir.glob("*.wav"))
    COMMON.require_exact_ids("generated wavs", expected_ids, wav_ids)

    expected_meta = expected_shard_paths(wav_dir, "gen_meta_shard", args.num_shards)
    expected_failures = expected_shard_paths(
        wav_dir, "failures_shard", args.num_shards
    )
    actual_meta = set(wav_dir.glob("gen_meta_shard*.jsonl"))
    actual_failures = set(wav_dir.glob("failures_shard*.jsonl"))
    if actual_meta != expected_meta:
        fail(
            "generation metadata shard mismatch: "
            f"missing={sorted(map(str, expected_meta - actual_meta))} "
            f"extra={sorted(map(str, actual_meta - expected_meta))}"
        )
    if actual_failures != expected_failures:
        fail(
            "failure ledger shard mismatch: "
            f"missing={sorted(map(str, expected_failures - actual_failures))} "
            f"extra={sorted(map(str, actual_failures - expected_failures))}"
        )

    meta = COMMON.parse_meta(
        str(wav_dir / "gen_meta_shard*.jsonl"),
        str(wav_dir / "failures_shard*.jsonl"),
    )
    COMMON.require_exact_ids("generation metadata", expected_ids, list(meta))
    timing_summary = summarize_token_decode_timing(meta)
    if timing_summary is not None:
        expected_warmup_ids = set(expected_ids[: args.num_shards])
        actual_warmup_ids = {
            utt_id for utt_id, row in meta.items() if row["timing_warmup"]
        }
        if actual_warmup_ids != expected_warmup_ids:
            fail(
                "timing warmup IDs mismatch: "
                f"expected={sorted(expected_warmup_ids)} "
                f"actual={sorted(actual_warmup_ids)}"
            )
    if seed_index_map_path is not None:
        for utt_id in expected_ids:
            expected_index = expected_seed_indices[utt_id]
            row = meta[utt_id]
            actual_index = row.get("generation_seed_index")
            actual_value = row.get("generation_seed_value")
            if actual_index != expected_index or actual_value != (
                args.seed_base + expected_index
            ):
                fail(
                    "canonical generation seed mismatch: "
                    f"id={utt_id} expected_index={expected_index} "
                    f"actual_index={actual_index} "
                    f"expected_seed={args.seed_base + expected_index} "
                    f"actual_seed={actual_value}"
                )
    contract_fields = {
        "prompt_contract",
        "language",
        "ref_text_punctuation",
        "cfg_unconditional_seed_policy",
        "guidance_scale",
        "generation_seed",
    }
    if args.is_baseline:
        leaked = {
            utt_id: sorted(contract_fields.intersection(row))
            for utt_id, row in meta.items()
            if contract_fields.intersection(row)
        }
        if leaked:
            fail(
                "baseline metadata shape changed; contract fields must remain absent: "
                f"{dict(list(leaked.items())[:5])}"
            )
    else:
        expected_language = None if args.lang_policy == "none" else args.lang
        expected_punctuation = (
            "preserve" if args.prompt_contract == "official-emilia" else "add"
        )
        for index, utt_id in enumerate(expected_ids):
            row = meta[utt_id]
            missing = sorted(contract_fields - row.keys())
            if missing:
                fail(f"metadata contract fields missing: id={utt_id} fields={missing}")
            expected_values = {
                "prompt_contract": args.prompt_contract,
                "language": expected_language,
                "ref_text_punctuation": expected_punctuation,
                "cfg_unconditional_seed_policy": args.cfg_unconditional_seed_policy,
                "generation_seed": (
                    args.seed_base + expected_seed_indices[utt_id]
                ),
            }
            mismatches = {
                key: {"expected": value, "actual": row.get(key)}
                for key, value in expected_values.items()
                if row.get(key) != value
            }
            try:
                guidance_matches = math.isclose(
                    float(row["guidance_scale"]),
                    args.guidance_scale,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError):
                guidance_matches = False
            if not guidance_matches:
                mismatches["guidance_scale"] = {
                    "expected": args.guidance_scale,
                    "actual": row.get("guidance_scale"),
                }
            if mismatches:
                fail(f"metadata contract mismatch: id={utt_id} {mismatches}")
            official_fields = {
                "ref_rms",
                "ref_truncated_samples",
                "output_ref_rms_restored",
            }
            if args.prompt_contract == "official-emilia":
                missing_official = sorted(official_fields - row.keys())
                if missing_official:
                    fail(
                        f"official prompt metadata missing: id={utt_id} "
                        f"fields={missing_official}"
                    )
            elif official_fields.intersection(row):
                fail(
                    "current prompt unexpectedly contains official-only metadata: "
                    f"id={utt_id} fields={sorted(official_fields.intersection(row))}"
                )
    stop_reasons: dict[str, int] = {}
    for row in meta.values():
        reason = str(row.get("stop_reason", "unknown"))
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    payload = {
        "arm": args.arm,
        "count": len(expected_ids),
        "lang": args.lang,
        "lang_policy": args.lang_policy,
        "num_shards": args.num_shards,
        "prompt_contract": args.prompt_contract,
        "cfg_unconditional_seed_policy": args.cfg_unconditional_seed_policy,
        "guidance_scale": args.guidance_scale,
        "seed_base": args.seed_base,
        "seed_index_map": (
            str(seed_index_map_path) if seed_index_map_path is not None else None
        ),
        "stop_reasons": stop_reasons,
        "wav_dir": str(wav_dir),
    }
    if timing_summary is not None:
        payload["token_decode_timing"] = timing_summary
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"GENERATION_OK count={len(expected_ids)} shards={args.num_shards} "
        f"stop_reasons={json.dumps(stop_reasons, sort_keys=True)}"
    )


def summarize_arm(args: argparse.Namespace) -> None:
    common_args = argparse.Namespace(
        arm=args.arm,
        lang=args.lang,
        guidance_scale=args.guidance_scale,
        steps_per_block=args.steps_per_block,
        expected_count=args.expected_count,
        tsv=args.tsv,
        jsonl=args.jsonl,
        wav_dir=args.wav_dir,
        anchor_dir=args.anchor_dir,
        wer_tsv=args.wer_tsv,
        sim_tsv=args.sim_tsv,
        meta_glob=args.meta_glob,
        failures_glob=args.failures_glob,
        per_utt_out=args.per_utt_out,
        summary_out=args.summary_out,
    )
    summary = COMMON.summarize_arm(common_args)
    timing_summary = summarize_token_decode_timing(
        COMMON.parse_meta(args.meta_glob, args.failures_glob)
    )
    if timing_summary is not None:
        summary.update(timing_summary)
    summary.update(
        {
            "lang_policy": args.lang_policy,
            "prompt_contract": args.prompt_contract,
            "cfg_unconditional_seed_policy": args.cfg_unconditional_seed_policy,
            "is_baseline": args.is_baseline,
        }
    )
    Path(args.summary_out).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_per_utt(path: Path, expected_count: int) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != expected_count:
        fail(f"{path} row count mismatch: expected={expected_count} actual={len(rows)}")
    if any(not row.get("id") or not row.get("sim") for row in rows):
        fail(f"{path} lacks id or sim values")
    ids = [row["id"] for row in rows]
    COMMON.require_unique(f"per-utterance report {path}", ids)
    return {row["id"]: row for row in rows}


def add_paired_sim_delta(
    row: dict[str, Any],
    baseline: dict[str, dict[str, str]],
    expected_count: int,
) -> None:
    current = read_per_utt(Path(row["per_utt_tsv"]), expected_count)
    if current.keys() != baseline.keys():
        fail(
            f"paired SIM ID mismatch for arm={row['arm']} lang={row['lang']}: "
            f"{COMMON.diff_ids(list(baseline), list(current))}"
        )
    deltas = [float(current[key]["sim"]) - float(baseline[key]["sim"]) for key in baseline]
    row.update(
        {
            "paired_sim_delta_mean": statistics.mean(deltas),
            "paired_sim_delta_median": statistics.median(deltas),
            "paired_sim_improved": sum(delta > 0 for delta in deltas),
            "paired_sim_equal": sum(delta == 0 for delta in deltas),
            "paired_sim_worse": sum(delta < 0 for delta in deltas),
        }
    )


SUMMARY_FIELDS = [
    "lang",
    "arm",
    "lang_policy",
    "prompt_contract",
    "cfg_unconditional_seed_policy",
    "guidance_scale",
    "steps_per_block",
    "count",
    "wer_percent",
    "sim",
    "paired_sim_delta_mean",
    "paired_sim_delta_median",
    "paired_sim_improved",
    "paired_sim_equal",
    "paired_sim_worse",
    "duration_ratio_mean",
    "duration_ratio_median",
    "duration_ratio_p95",
    "runaway",
    "lt_0_6",
    "gt_2",
    "eos_count",
    "max_blocks_count",
    "per_utt_tsv",
]


def aggregate(args: argparse.Namespace) -> None:
    arms = COMMON.parse_csv_values(args.expected_arms)
    languages = COMMON.parse_csv_values(args.expected_languages)
    if args.baseline_arm not in arms:
        fail(f"baseline arm is absent from expected arms: {args.baseline_arm}")
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.summaries]
    expected = set(itertools.product(languages, arms))
    actual_list = [(row.get("lang"), row.get("arm")) for row in rows]
    repeated = COMMON.duplicates([repr(value) for value in actual_list])
    if repeated:
        fail(f"duplicate summaries: {repeated}")
    actual = set(actual_list)
    if actual != expected:
        fail(
            "summary matrix mismatch: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    for row in rows:
        if row.get("count") != args.expected_count:
            fail(
                f"arm count mismatch: arm={row.get('arm')} lang={row.get('lang')} "
                f"expected={args.expected_count} actual={row.get('count')}"
            )
        for field in (
            "lang_policy",
            "prompt_contract",
            "cfg_unconditional_seed_policy",
        ):
            if field not in row:
                fail(f"summary lacks {field}: arm={row['arm']} lang={row['lang']}")

    timing_field_set = set(TOKEN_DECODE_TIMING_SUMMARY_FIELDS)
    timing_sets = [timing_field_set.intersection(row) for row in rows]
    timing_enabled = bool(any(timing_sets))
    if timing_enabled and any(fields != timing_field_set for fields in timing_sets):
        fail("mixed token decode timing contract across summaries")
    if timing_enabled:
        for row in rows:
            timed_count = row["timed_count"]
            if (
                isinstance(timed_count, bool)
                or not isinstance(timed_count, int)
                or timed_count <= 0
            ):
                fail(
                    "invalid token decode timing summary: "
                    f"arm={row['arm']} lang={row['lang']} "
                    f"timed_count={timed_count!r}"
                )
            for field in TOKEN_DECODE_TIMING_SUMMARY_FIELDS[1:]:
                value = row[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    fail(
                        "invalid token decode timing summary: "
                        f"arm={row['arm']} lang={row['lang']} "
                        f"field={field} value={value!r}"
                    )

    for lang in languages:
        baseline_row = next(
            row
            for row in rows
            if row["lang"] == lang and row["arm"] == args.baseline_arm
        )
        baseline = read_per_utt(
            Path(baseline_row["per_utt_tsv"]), args.expected_count
        )
        for row in rows:
            if row["lang"] == lang:
                add_paired_sim_delta(row, baseline, args.expected_count)

    rows.sort(key=lambda row: (languages.index(row["lang"]), arms.index(row["arm"])))
    outputs = [Path(args.output_tsv), Path(args.output_json), Path(args.output_md)]
    for output in outputs:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            fail(f"refusing to overwrite report: {output}")
    output_fields = list(SUMMARY_FIELDS)
    if timing_enabled:
        output_fields.extend(TOKEN_DECODE_TIMING_SUMMARY_FIELDS)
    with Path(args.output_tsv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=output_fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in output_fields} for row in rows)

    payload = {
        "baseline_arm": args.baseline_arm,
        "expected_count": args.expected_count,
        "languages": languages,
        "arms": arms,
        "rows": rows,
    }
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        f"# Training-contract first-{args.expected_count} paired probe",
        "",
        f"Baseline arm: `{args.baseline_arm}`. Duration columns are ratios to the fixed R1 anchors.",
        "",
    ]
    if timing_enabled:
        lines.extend(
            [
                "| lang | arm | lang policy | prompt | CFG seed | gs | WER % | SIM | paired SIM Δ mean | Δ median | +/=/- | dur mean | dur median | dur p95 | runaway | <0.6 | >2 | timed count | token decode s mean | median | p95 | token decode RTF mean | median | token decode RTF p95 |",
                "|:---:|:---|:---|:---|:---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| lang | arm | lang policy | prompt | CFG seed | gs | WER % | SIM | paired SIM Δ mean | Δ median | +/=/- | dur mean | dur median | dur p95 | runaway | <0.6 | >2 |",
                "|:---:|:---|:---|:---|:---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    for row in rows:
        line = (
            f"| {row['lang']} | {row['arm']} | {row['lang_policy']} | "
            f"{row['prompt_contract']} | {row['cfg_unconditional_seed_policy']} | "
            f"{row['guidance_scale']:g} | {row['wer_percent']:.2f} | "
            f"{row['sim']:.3f} | {row['paired_sim_delta_mean']:+.6f} | "
            f"{row['paired_sim_delta_median']:+.6f} | "
            f"{row['paired_sim_improved']}/{row['paired_sim_equal']}/{row['paired_sim_worse']} | "
            f"{row['duration_ratio_mean']:.3f} | {row['duration_ratio_median']:.3f} | "
            f"{row['duration_ratio_p95']:.3f} | {row['runaway']} | "
            f"{row['lt_0_6']} | {row['gt_2']} |"
        )
        if timing_enabled:
            line = (
                line[:-1]
                + f" {row['timed_count']} | "
                f"{row['token_decode_seconds_mean']:.4f} | "
                f"{row['token_decode_seconds_median']:.4f} | "
                f"{row['token_decode_seconds_p95']:.4f} | "
                f"{row['token_decode_rtf_mean']:.4f} | "
                f"{row['token_decode_rtf_median']:.4f} | "
                f"{row['token_decode_rtf_p95']:.4f} |"
            )
        lines.append(line)
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"CONTRACT_REPORT_OK rows={len(rows)} baseline={args.baseline_arm} "
        f"json={Path(args.output_json).resolve()} md={Path(args.output_md).resolve()}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-inputs")
    prepare.add_argument("--source-tsv", required=True)
    prepare.add_argument("--source-jsonl", required=True)
    prepare.add_argument("--count", required=True, type=int)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--anchor-dir", required=True)
    prepare.add_argument("--check-ref-audio", action="store_true")
    prepare.set_defaults(function=prepare_inputs)

    validate = subparsers.add_parser("validate-generation")
    validate.add_argument("--tsv", required=True)
    validate.add_argument("--jsonl", required=True)
    validate.add_argument("--expected-count", required=True, type=int)
    validate.add_argument("--num-shards", required=True, type=int)
    validate.add_argument("--arm", required=True)
    validate.add_argument("--lang", required=True, choices=("zh", "en"))
    validate.add_argument("--lang-policy", required=True, choices=("dataset", "none"))
    validate.add_argument(
        "--prompt-contract", required=True, choices=("current", "official-emilia")
    )
    validate.add_argument(
        "--cfg-unconditional-seed-policy",
        required=True,
        choices=("shared", "drop_ref"),
    )
    validate.add_argument("--guidance-scale", required=True, type=float)
    validate.add_argument("--seed-base", default=20260707, type=int)
    validate.add_argument("--seed-index-map")
    validate.add_argument("--is-baseline", action="store_true")
    validate.add_argument("--wav-dir", required=True)
    validate.add_argument("--output", required=True)
    validate.set_defaults(function=validate_generation)

    summarize = subparsers.add_parser("summarize-arm")
    summarize.add_argument("--arm", required=True)
    summarize.add_argument("--lang", required=True, choices=("zh", "en"))
    summarize.add_argument("--lang-policy", required=True, choices=("dataset", "none"))
    summarize.add_argument(
        "--prompt-contract", required=True, choices=("current", "official-emilia")
    )
    summarize.add_argument(
        "--cfg-unconditional-seed-policy",
        required=True,
        choices=("shared", "drop_ref"),
    )
    summarize.add_argument("--is-baseline", action="store_true")
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
    combine.add_argument("--summaries", required=True, nargs="+")
    combine.add_argument("--expected-arms", required=True)
    combine.add_argument("--expected-languages", default="zh,en")
    combine.add_argument("--baseline-arm", default="baseline")
    combine.add_argument("--expected-count", required=True, type=int)
    combine.add_argument("--output-tsv", required=True)
    combine.add_argument("--output-json", required=True)
    combine.add_argument("--output-md", required=True)
    combine.set_defaults(function=aggregate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.function(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
