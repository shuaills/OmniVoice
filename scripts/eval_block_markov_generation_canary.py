#!/usr/bin/env python3
"""Gate fixed bilingual first-5 generation canaries against frozen base."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ABSOLUTE_THRESHOLDS = {
    "wer_percent_max": 50.0,
    "sim_min": 0.20,
    "duration_ratio_mean_min": 0.50,
    "duration_ratio_mean_max": 2.00,
    "duration_ratio_p95_max": 2.50,
}
RELATIVE_THRESHOLDS = {
    "wer_delta_points_max": 10.0,
    "sim_delta_min": -0.05,
    "duration_ratio_mean_abs_delta_max": 0.25,
}
EXPECTED_ARMS = (
    "shared_g0",
    "shared_g0p25",
    "shared_g0p5",
    "shared_g1",
    "shared_g2",
)
EXPECTED_KEYS = {
    (language, arm) for language in ("zh", "en") for arm in EXPECTED_ARMS
}


def _load(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _rows(payload: dict) -> dict[tuple[str, str], dict]:
    rows = {}
    for row in payload.get("rows", []):
        key = (row.get("lang"), row.get("arm"))
        if key in rows:
            raise RuntimeError(f"duplicate canary row: {key}")
        rows[key] = row
    if not rows:
        raise RuntimeError("canary summary has no rows")
    return rows


def _finite(row: dict, field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"invalid {field}: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {field}: {value!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--head-summary", type=Path, required=True)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--head-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = {
        "base": args.base_summary,
        "control": args.control_summary,
        "head": args.head_summary,
    }
    results = {
        "base": args.base_result,
        "control": args.control_result,
        "head": args.head_result,
    }
    matrices = {name: _rows(_load(path)) for name, path in summaries.items()}
    expected_keys = set(matrices["base"])
    if expected_keys != EXPECTED_KEYS:
        raise RuntimeError(
            "generation canary does not contain the registered bilingual matrix: "
            f"expected={sorted(EXPECTED_KEYS)} actual={sorted(expected_keys)}"
        )
    for name, matrix in matrices.items():
        if set(matrix) != expected_keys:
            raise RuntimeError(
                "generation canary matrix mismatch: "
                f"base={sorted(expected_keys)} {name}={sorted(matrix)}"
            )
    for language in ("zh", "en"):
        input_hashes = {
            name: _load(result / "inputs" / language / "manifest.json").get(
                "ordered_ids_sha256"
            )
            for name, result in results.items()
        }
        if not input_hashes["base"] or len(set(input_hashes.values())) != 1:
            raise RuntimeError(
                f"{language} canary input IDs differ: {input_hashes}"
            )

    failures: list[str] = []
    absolute_checks = []
    relative_comparisons = []
    timing_comparisons = []
    for key in sorted(expected_keys):
        rows = {name: matrix[key] for name, matrix in matrices.items()}
        guidance = {
            name: _finite(row, "guidance_scale") for name, row in rows.items()
        }
        if len(set(guidance.values())) != 1:
            raise RuntimeError(f"guidance mismatch for {key}: {guidance}")
        for name, row in rows.items():
            runaway = _finite(row, "runaway")
            max_blocks = _finite(row, "max_blocks_count")
            if runaway != 0.0:
                failures.append(f"{name} {key} has runaway={row.get('runaway')}")
            if max_blocks != 0.0:
                failures.append(
                    f"{name} {key} hit max_blocks={row.get('max_blocks_count')}"
                )
            wer = _finite(row, "wer_percent")
            sim = _finite(row, "sim")
            duration_mean = _finite(row, "duration_ratio_mean")
            duration_p95 = _finite(row, "duration_ratio_p95")
            rtf = _finite(row, "token_decode_rtf_median")
            if not 0.0 <= wer <= ABSOLUTE_THRESHOLDS["wer_percent_max"]:
                failures.append(f"{name} {key} absolute WER disaster={wer:.4f}%")
            if not ABSOLUTE_THRESHOLDS["sim_min"] <= sim <= 1.0:
                failures.append(f"{name} {key} absolute SIM disaster={sim:.6f}")
            if not (
                ABSOLUTE_THRESHOLDS["duration_ratio_mean_min"]
                <= duration_mean
                <= ABSOLUTE_THRESHOLDS["duration_ratio_mean_max"]
            ):
                failures.append(
                    f"{name} {key} absolute duration-mean disaster="
                    f"{duration_mean:.6f}"
                )
            if not (
                0.0
                < duration_p95
                <= ABSOLUTE_THRESHOLDS["duration_ratio_p95_max"]
            ):
                failures.append(
                    f"{name} {key} absolute duration-p95 disaster="
                    f"{duration_p95:.6f}"
                )
            if rtf <= 0.0:
                raise RuntimeError(
                    f"{name} {key} invalid token-decode RTF={rtf:.6f}"
                )
            absolute_checks.append(
                {
                    "variant": name,
                    "lang_arm": list(key),
                    "guidance_scale": row["guidance_scale"],
                    "wer_percent": wer,
                    "sim": sim,
                    "duration_ratio_mean": duration_mean,
                    "duration_ratio_p95": duration_p95,
                    "token_decode_rtf_median": rtf,
                }
            )

        candidate = rows["head"]
        head_rtf = _finite(candidate, "token_decode_rtf_median")
        for reference_name in ("base", "control"):
            reference = rows[reference_name]
            wer_delta = _finite(candidate, "wer_percent") - _finite(
                reference, "wer_percent"
            )
            sim_delta = _finite(candidate, "sim") - _finite(reference, "sim")
            duration_delta = abs(
                _finite(candidate, "duration_ratio_mean")
                - _finite(reference, "duration_ratio_mean")
            )
            if wer_delta > RELATIVE_THRESHOLDS["wer_delta_points_max"]:
                failures.append(
                    f"head {key} WER disaster vs {reference_name}: "
                    f"delta={wer_delta:.4f} points"
                )
            if sim_delta < RELATIVE_THRESHOLDS["sim_delta_min"]:
                failures.append(
                    f"head {key} SIM disaster vs {reference_name}: "
                    f"delta={sim_delta:.6f}"
                )
            if (
                duration_delta
                > RELATIVE_THRESHOLDS["duration_ratio_mean_abs_delta_max"]
            ):
                failures.append(
                    f"head {key} duration-ratio disaster vs {reference_name}: "
                    f"delta={duration_delta:.6f}"
                )
            relative_comparisons.append(
                {
                    "reference": reference_name,
                    "lang_arm": list(key),
                    "guidance_scale": candidate["guidance_scale"],
                    "wer_delta_points": wer_delta,
                    "sim_delta": sim_delta,
                    "duration_ratio_mean_abs_delta": duration_delta,
                }
            )
            reference_rtf = _finite(reference, "token_decode_rtf_median")
            timing_comparisons.append(
                {
                    "reference": reference_name,
                    "lang_arm": list(key),
                    "head_reference_token_decode_rtf_ratio": (
                        head_rtf / reference_rtf
                    ),
                }
            )

    report = {
        "verdict": "PASS" if not failures else "FAIL",
        "scope": (
            "bilingual first5 absolute-and-relative generation disaster canary; "
            "sequential timing is descriptive only and is not a quality claim"
        ),
        "failures": failures,
        "absolute_thresholds": ABSOLUTE_THRESHOLDS,
        "relative_thresholds": RELATIVE_THRESHOLDS,
        "absolute_checks": absolute_checks,
        "relative_comparisons": relative_comparisons,
        "timing_comparisons_descriptive_only": timing_comparisons,
        "summaries": {
            name: str(path.resolve()) for name, path in summaries.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        "BLOCK_MARKOV_GENERATION_CANARY "
        + json.dumps(report, sort_keys=True, allow_nan=False)
    )


if __name__ == "__main__":
    main()
