#!/usr/bin/env python3
"""Finalize the seed-0 anchor proof with fail-closed artifact hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
VERDICTS = (
    "PROMOTE_TO_3SEED",
    "SCIENTIFIC_KILL",
    "INCONCLUSIVE_1K",
    "INVALID_IMPLEMENTATION",
    "ENGINEERING_BLOCK",
)
TRAINING_VERDICTS = frozenset({"PASS", "INVALID_IMPLEMENTATION", "ENGINEERING_BLOCK"})
ENVIRONMENT_KEYS = frozenset(
    {
        "accelerate_version",
        "cuda_version",
        "cudnn_version",
        "gpu_inventory",
        "liger_kernel_file_count",
        "liger_kernel_source_root",
        "liger_kernel_tree_sha256",
        "nvidia_driver_version",
        "python_executable",
        "python_version",
        "source_commit",
        "torch_version",
        "transformers_version",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"artifact must be NAME=PATH, got {specification!r}")
    name, raw_path = specification.split("=", 1)
    if not KEY_RE.fullmatch(name):
        raise ValueError(f"invalid artifact name: {name!r}")
    path = Path(raw_path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"artifact is missing or empty: {path}")
    return name, path


def resolve_final_verdict(training_verdict: str, nll_verdict: str) -> str:
    """Resolve with infrastructure/implementation severity ahead of science."""
    if training_verdict not in TRAINING_VERDICTS:
        raise ValueError(f"invalid training verdict: {training_verdict!r}")
    if nll_verdict not in VERDICTS:
        raise ValueError(f"invalid NLL verdict: {nll_verdict!r}")
    if "INVALID_IMPLEMENTATION" in (training_verdict, nll_verdict):
        return "INVALID_IMPLEMENTATION"
    if "ENGINEERING_BLOCK" in (training_verdict, nll_verdict):
        return "ENGINEERING_BLOCK"
    return nll_verdict


def _checkpoint_inventory(specification: str) -> tuple[str, Path, dict]:
    if "=" not in specification:
        raise ValueError(f"checkpoint must be NAME=PATH, got {specification!r}")
    name, raw_path = specification.split("=", 1)
    if not KEY_RE.fullmatch(name):
        raise ValueError(f"invalid checkpoint name: {name!r}")
    root = Path(raw_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory is missing: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint inventory forbids symlinks: {path}")
        if not path.is_file():
            continue
        if path.stat().st_size == 0:
            raise ValueError(f"checkpoint inventory contains empty file: {path}")
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise ValueError(f"checkpoint inventory is empty: {root}")
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    return name, root, {
        "path": str(root),
        "file_count": len(files),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scientific-verdict", choices=VERDICTS, required=True)
    parser.add_argument("--verdict-report", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--nll-report", type=Path, required=True)
    parser.add_argument("--environment-report", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--english-only-mechanism-probe", action="store_true")
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    output = args.output.resolve()
    temporary = output.with_name(output.name + ".tmp")
    if not manifest.is_file() or manifest.stat().st_size == 0:
        raise SystemExit(f"manifest is missing or empty: {manifest}")
    if output.exists() or temporary.exists():
        raise SystemExit(f"refusing to reuse receipt output: {output}")
    manifest_text = manifest.read_text()
    if "artifact_receipt=" in manifest_text or "scientific_verdict=" in manifest_text:
        raise SystemExit("manifest is already finalized")
    verdict_report = args.verdict_report.resolve()
    if not verdict_report.is_file() or verdict_report.stat().st_size == 0:
        raise SystemExit(f"verdict report is missing or empty: {verdict_report}")
    verdict_payload = json.loads(verdict_report.read_text())
    training_report = args.training_report.resolve()
    nll_report = args.nll_report.resolve()
    for name, path in (
        ("training", training_report),
        ("NLL", nll_report),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"{name} report is missing or empty: {path}")
    training_payload = json.loads(training_report.read_text())
    nll_payload = json.loads(nll_report.read_text())
    recomputed_verdict = resolve_final_verdict(
        training_payload.get("verdict"), nll_payload.get("verdict")
    )
    reported_verdict = verdict_payload.get("verdict")
    if (
        reported_verdict != args.scientific_verdict
        or recomputed_verdict != args.scientific_verdict
    ):
        raise SystemExit(
            "scientific verdict mismatch: "
            f"argument={args.scientific_verdict!r} "
            f"report={reported_verdict!r} recomputed={recomputed_verdict!r}"
        )
    expected_report_fields = {
        "training_verdict": training_payload["verdict"],
        "nll_verdict": nll_payload["verdict"],
        "generation_status": "NEEDS_GENERATION",
        "automatic_followup_submitted": False,
    }
    actual_report_fields = {
        key: verdict_payload.get(key) for key in expected_report_fields
    }
    if actual_report_fields != expected_report_fields:
        raise SystemExit(
            "verdict report contradicts source reports: "
            f"expected={expected_report_fields} actual={actual_report_fields}"
        )
    if (args.seed_index, args.train_seed, args.eval_seed) != (0, 42, 20260719):
        raise SystemExit(
            "seed-0 proof requires seed_index=0 train_seed=42 eval_seed=20260719"
        )
    if not args.english_only_mechanism_probe:
        raise SystemExit("English-only mechanism-probe acknowledgement is required")
    expected_verdict_contract = {
        "seed_index": 0,
        "train_seed": 42,
        "eval_seed": 20260719,
        "english_only_mechanism_probe": True,
    }
    verdict_contract = {
        key: verdict_payload.get(key) for key in expected_verdict_contract
    }
    if verdict_contract != expected_verdict_contract:
        raise SystemExit(
            "verdict report seed/probe contract mismatch: "
            f"expected={expected_verdict_contract} actual={verdict_contract}"
        )

    environment_path = args.environment_report.resolve()
    if not environment_path.is_file() or environment_path.stat().st_size == 0:
        raise SystemExit(f"environment report is missing or empty: {environment_path}")
    environment = json.loads(environment_path.read_text())
    missing_environment = sorted(
        key for key in ENVIRONMENT_KEYS if environment.get(key) in (None, "", [])
    )
    if missing_environment:
        raise SystemExit(f"environment report is incomplete: {missing_environment}")
    manifest_values = {}
    for line in manifest_text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            manifest_values[key] = value
    if environment["source_commit"] != manifest_values.get("source_commit"):
        raise SystemExit(
            "environment source commit does not match manifest: "
            f"environment={environment['source_commit']!r} "
            f"manifest={manifest_values.get('source_commit')!r}"
        )

    artifacts: dict[str, dict[str, str | int]] = {}
    for specification in args.artifact:
        name, path = _artifact(specification)
        if name in artifacts:
            raise SystemExit(f"duplicate artifact name: {name}")
        artifacts[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    if not artifacts:
        raise SystemExit("at least one --artifact is required")

    checkpoints: dict[str, dict] = {}
    for specification in args.checkpoint:
        name, _, inventory = _checkpoint_inventory(specification)
        if name in checkpoints:
            raise SystemExit(f"duplicate checkpoint name: {name}")
        checkpoints[name] = inventory
    if not checkpoints:
        raise SystemExit("at least one complete --checkpoint inventory is required")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "schema": "block_anchor_seed0_300proof_receipt_v2",
        "generated_utc": generated,
        "scientific_verdict": args.scientific_verdict,
        "verdict_report": str(verdict_report),
        "training_report": str(training_report),
        "nll_report": str(nll_report),
        "generation_status": "NEEDS_GENERATION",
        "automatic_followup_submitted": False,
        "seed_index": args.seed_index,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "english_only_mechanism_probe": True,
        "environment": environment,
        "checkpoint_inventories": dict(sorted(checkpoints.items())),
        "manifest": str(manifest),
        "manifest_before_receipt_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
        "artifacts": dict(sorted(artifacts.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    receipt_sha = _sha256(output)
    with manifest.open("a") as stream:
        stream.write(f"artifact_receipt={output}\n")
        stream.write(f"artifact_receipt_sha256={receipt_sha}\n")
        stream.write(f"artifact_count={len(artifacts)}\n")
        stream.write(f"scientific_verdict={args.scientific_verdict}\n")
        stream.write(f"seed_index={args.seed_index}\n")
        stream.write(f"train_seed={args.train_seed}\n")
        stream.write(f"eval_seed={args.eval_seed}\n")
        stream.write("english_only_mechanism_probe=1\n")
        stream.write(f"checkpoint_inventory_count={len(checkpoints)}\n")
        stream.write("generation_status=NEEDS_GENERATION\n")
        stream.write("automatic_followup_submitted=0\n")
    print(
        "BLOCK_ANCHOR_RECEIPT_FINALIZED "
        + json.dumps(
            {
                "artifact_count": len(artifacts),
                "receipt": str(output),
                "receipt_sha256": receipt_sha,
                "scientific_verdict": args.scientific_verdict,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
