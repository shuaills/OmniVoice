#!/usr/bin/env python3
"""Finalize the seed-0 anchor proof with fail-closed artifact hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
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
RECOVERY_SOURCE_CHANGED_PATHS = (
    "dspark_anchor_scan_recover_eval.sh",
    "scripts/check_block_anchor_checkpoint_attach.py",
    "scripts/eval_block_anchor_nll.py",
    "scripts/finalize_block_anchor_receipt.py",
    "tests/test_block_anchor_campaign.py",
    "tests/test_block_anchor_recovery.py",
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


def _json_file(path: Path, *, label: str) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return payload


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        output = getattr(error, "output", "")
        raise ValueError(
            f"failed to inspect frozen recovery source with git {arguments!r}: {output}"
        ) from error


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


def _unique_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            raise ValueError(f"malformed manifest line in {path}: {line!r}")
        key, value = line.split("=", 1)
        if key in values:
            raise ValueError(f"duplicate manifest key in {path}: {key!r}")
        values[key] = value
    return values


def _model_sha(inventory: dict) -> str:
    models = [
        item["sha256"]
        for item in inventory["files"]
        if item["relative_path"] == "model.safetensors"
    ]
    if len(models) != 1:
        raise ValueError(
            f"checkpoint must contain one model.safetensors: {inventory['path']}"
        )
    return models[0]


def _require_training_checkpoint(inventory: dict, *, random_state_count: int) -> None:
    names = {item["relative_path"] for item in inventory["files"]}
    required = {
        "config.json",
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "train_config.json",
    }
    missing = sorted(required - names)
    random_states = sorted(
        name
        for name in names
        if name.startswith("random_states_") and name.endswith(".pkl")
    )
    expected_random_states = [
        f"random_states_{index}.pkl" for index in range(random_state_count)
    ]
    if missing or random_states != expected_random_states:
        raise ValueError(
            "incomplete checkpoint inventory: "
            f"path={inventory['path']} missing={missing} "
            f"random_states={random_states} expected={expected_random_states}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scientific-verdict", choices=VERDICTS, required=True)
    parser.add_argument("--verdict-report", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--nll-report", type=Path, required=True)
    parser.add_argument("--environment-report", type=Path, required=True)
    parser.add_argument(
        "--receipt-mode",
        choices=("standard", "recovery_eval_only"),
        default="standard",
    )
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--parent-failure-log", type=Path)
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
    try:
        manifest_values = _unique_manifest(manifest)
    except ValueError as error:
        raise SystemExit(str(error)) from error
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
    input_report_sha256 = {
        "verdict_report": _sha256(verdict_report),
        "training_report": _sha256(training_report),
        "nll_report": _sha256(nll_report),
        "environment_report": _sha256(environment_path),
    }
    missing_environment = sorted(
        key for key in ENVIRONMENT_KEYS if environment.get(key) in (None, "", [])
    )
    if missing_environment:
        raise SystemExit(f"environment report is incomplete: {missing_environment}")
    expected_manifest_contract = {
        "seed_index": str(args.seed_index),
        "train_seed": str(args.train_seed),
        "eval_seed": str(args.eval_seed),
        "english_only_mechanism_probe": "1",
        "automatic_followup_submitted": "0",
    }
    for key, expected in expected_manifest_contract.items():
        if manifest_values.get(key) != expected:
            raise SystemExit(
                f"manifest proof contract mismatch for {key}: "
                f"expected={expected!r} actual={manifest_values.get(key)!r}"
            )
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

    recovery = None
    recovery_arguments = (args.parent_manifest, args.parent_failure_log)
    if args.receipt_mode == "standard":
        if any(value is not None for value in recovery_arguments):
            raise SystemExit(
                "parent recovery arguments require --receipt-mode recovery_eval_only"
            )
    else:
        if any(value is None for value in recovery_arguments):
            raise SystemExit(
                "recovery_eval_only requires --parent-manifest and "
                "--parent-failure-log"
            )
        parent_manifest = args.parent_manifest.resolve()
        parent_failure_log = args.parent_failure_log.resolve()
        for name, path in (
            ("parent manifest", parent_manifest),
            ("parent failure log", parent_failure_log),
        ):
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"{name} is missing or empty: {path}")
        try:
            parent_values = _unique_manifest(parent_manifest)
            recovery_values = _unique_manifest(manifest)
        except ValueError as error:
            raise SystemExit(str(error)) from error

        parent_root = parent_manifest.parent.resolve()
        parent_run_id = parent_values.get("run_id")
        exact_parent = {
            "scope": "seed0_300step_only",
            "seed_index": "0",
            "train_seed": "42",
            "eval_seed": "20260719",
            "english_only_mechanism_probe": "1",
            "automatic_followup_submitted": "0",
            "failed_phase": "nll",
            "infrastructure_rc": "1",
        }
        for key, expected in exact_parent.items():
            if parent_values.get(key) != expected:
                raise SystemExit(
                    f"parent manifest recovery mismatch for {key}: "
                    f"expected={expected!r} actual={parent_values.get(key)!r}"
                )
        if not parent_run_id or not re.fullmatch(r"shuai-[A-Za-z0-9._-]+", parent_run_id):
            raise SystemExit(f"invalid parent run id: {parent_run_id!r}")
        parent_source_commit = parent_values.get("source_commit")
        if not parent_source_commit or not re.fullmatch(
            r"[0-9a-f]{40}", parent_source_commit
        ):
            raise SystemExit(
                f"invalid parent source commit: {parent_source_commit!r}"
            )
        for forbidden in (
            "artifact_receipt",
            "artifact_receipt_sha256",
            "proof_complete",
            "scientific_verdict",
            "train_rc",
            "tee_rc",
            "rc",
        ):
            if forbidden in parent_values:
                raise SystemExit(
                    f"parent proof is not an unfinalized NLL failure: {forbidden}"
                )

        parent_manifest_sha = _sha256(parent_manifest)
        parent_training = (
            parent_root / "reports" / f"{parent_run_id}.training.json"
        ).resolve()
        if not parent_training.is_file() or parent_training.stat().st_size == 0:
            raise SystemExit(
                f"parent training report is missing or empty: {parent_training}"
            )
        parent_training_sha = _sha256(parent_training)
        parent_training_payload = _json_file(
            parent_training, label="parent training report"
        )
        actual_training_proofs = {
            "verdict": parent_training_payload.get("verdict"),
            "frozen_backbone_hash_exact": parent_training_payload.get(
                "frozen_backbone_hash_exact"
            ),
            "causal_replay_bitwise_exact": parent_training_payload.get(
                "causal_replay_bitwise_exact"
            ),
        }
        if (
            actual_training_proofs["verdict"] != "PASS"
            or actual_training_proofs["frozen_backbone_hash_exact"] is not True
            or actual_training_proofs["causal_replay_bitwise_exact"] is not True
        ):
            raise SystemExit(
                "parent training report is not reusable: "
                f"actual={actual_training_proofs}"
            )
        exact_recovery = {
            "scope": "seed0_300step_eval_only_recovery",
            "recovery_scope": "eval_only",
            "training_reused": "1",
            "training_reexecuted": "0",
            "seed_index": "0",
            "train_seed": "42",
            "eval_seed": "20260719",
            "english_only_mechanism_probe": "1",
            "parent_root": str(parent_root),
            "parent_run_id": parent_run_id,
            "parent_source_commit": parent_source_commit,
            "parent_manifest": str(parent_manifest),
            "parent_manifest_sha256": parent_manifest_sha,
            "parent_training_report": str(parent_training),
            "parent_training_report_sha256": parent_training_sha,
            "parent_failed_phase": "nll",
            "parent_infrastructure_rc": "1",
            "parent_failure_log": str(parent_failure_log),
            "parent_failure_log_sha256": _sha256(parent_failure_log),
            "automatic_followup_submitted": "0",
        }
        for key, expected in exact_recovery.items():
            if recovery_values.get(key) != expected:
                raise SystemExit(
                    f"recovery manifest lineage mismatch for {key}: "
                    f"expected={expected!r} actual={recovery_values.get(key)!r}"
                )
        if manifest.parent.resolve() == parent_root:
            raise SystemExit("recovery manifest must not share the parent result root")
        if environment["source_commit"] == parent_source_commit:
            raise SystemExit("recovery source commit must differ from parent source")
        if environment.get("parent_source_commit") != parent_source_commit:
            raise SystemExit("environment parent source commit mismatch")
        if environment.get("recovery_scope") != "eval_only":
            raise SystemExit("environment does not identify eval-only recovery")
        if (
            verdict_payload.get("recovery_scope") != "eval_only"
            or verdict_payload.get("training_reused") is not True
            or verdict_payload.get("training_reexecuted") is not False
        ):
            raise SystemExit(
                "verdict report does not identify strict reused eval-only recovery"
            )
        if training_payload.get("verdict") != "PASS":
            raise SystemExit("eval-only recovery requires a PASS training report")
        if _sha256(training_report) != parent_training_sha:
            raise SystemExit(
                "regenerated training report is not byte-identical to parent"
            )

        expected_checkpoint_paths = {
            "base": Path(parent_values["base_checkpoint"]).resolve(),
            "causal": Path(parent_values["causal_output"]).resolve()
            / "checkpoint-300",
            "stateless": Path(parent_values["stateless_output"]).resolve()
            / "checkpoint-300",
            "causal_replay": Path(parent_values["causal_replay_output"]).resolve()
            / "checkpoint-300",
        }
        if set(checkpoints) != set(expected_checkpoint_paths):
            raise SystemExit(
                "recovery checkpoint set must be exactly "
                f"{sorted(expected_checkpoint_paths)}, got {sorted(checkpoints)}"
            )
        for name, expected_path in expected_checkpoint_paths.items():
            actual_path = Path(checkpoints[name]["path"])
            if actual_path != expected_path:
                raise SystemExit(
                    f"recovery checkpoint path mismatch for {name}: "
                    f"expected={expected_path} actual={actual_path}"
                )
            _require_training_checkpoint(
                checkpoints[name], random_state_count=8 if name == "base" else 2
            )

        failure_text = parent_failure_log.read_text(errors="replace")
        failure_fragments = (
            "_benchmark_head_on_off",
            "_forward_device",
            "ValueError: Expected query, key, and value to have the same dtype",
            "query.dtype: torch.float32",
            "key.dtype: torch.float32",
            "value.dtype: torch.bfloat16",
        )
        missing_failure = [
            fragment for fragment in failure_fragments if fragment not in failure_text
        ]
        if missing_failure:
            raise SystemExit(
                f"parent failure evidence lacks dtype signature: {missing_failure}"
            )

        required_artifacts = {
            "parent_manifest": parent_manifest,
            "parent_training_report": parent_training,
            "parent_failure_log": parent_failure_log,
            "regenerated_training_report": training_report,
            "nll_report": nll_report,
            "verdict_report": verdict_report,
            "environment_report": environment_path,
        }
        for name, expected_path in required_artifacts.items():
            actual = artifacts.get(name, {}).get("path")
            if actual != str(expected_path):
                raise SystemExit(
                    f"recovery artifact binding mismatch for {name}: "
                    f"expected={expected_path} actual={actual!r}"
                )

        provenance = nll_payload.get("provenance")
        if not isinstance(provenance, dict):
            raise SystemExit("NLL report lacks evaluation-input provenance")
        if nll_payload.get("num_packs") != 32 or not re.fullmatch(
            r"[0-9a-f]{64}", str(nll_payload.get("snapshot_sha256", ""))
        ):
            raise SystemExit("NLL report lacks the fixed 32-pack snapshot receipt")
        expected_nll_contract = {
            "seed_index": 0,
            "train_seed": 42,
            "eval_seed": 20260719,
            "english_only_mechanism_probe": True,
        }
        actual_nll_contract = {
            key: nll_payload.get(key) for key in expected_nll_contract
        }
        if (
            actual_nll_contract["seed_index"] != 0
            or actual_nll_contract["train_seed"] != 42
            or actual_nll_contract["eval_seed"] != 20260719
            or actual_nll_contract["english_only_mechanism_probe"] is not True
        ):
            raise SystemExit(
                "NLL seed/probe contract mismatch: "
                f"expected={expected_nll_contract} actual={actual_nll_contract}"
            )
        if provenance.get("source_commit") != environment["source_commit"]:
            raise SystemExit("NLL source commit differs from recovery environment")
        if provenance.get("source_root") != recovery_values.get("source_root"):
            raise SystemExit("NLL source root differs from recovery manifest")
        expected_configs = {
            "causal": Path(recovery_values["source_root"])
            / parent_values["causal_config"],
            "stateless": Path(recovery_values["source_root"])
            / parent_values["stateless_config"],
        }
        actual_configs = provenance.get("configs")
        if not isinstance(actual_configs, dict):
            raise SystemExit("NLL report lacks config provenance")
        for name, expected_path in expected_configs.items():
            expected_path = expected_path.resolve()
            actual = actual_configs.get(name)
            if actual != {
                "path": str(expected_path),
                "sha256": _sha256(expected_path),
            }:
                raise SystemExit(f"NLL config provenance mismatch for {name}")
        actual_checkpoints = provenance.get("checkpoints")
        if not isinstance(actual_checkpoints, dict):
            raise SystemExit("NLL report lacks checkpoint provenance")
        for name in ("base", "causal", "stateless"):
            expected = {
                "path": checkpoints[name]["path"],
                "model_sha256": _model_sha(checkpoints[name]),
            }
            if actual_checkpoints.get(name) != expected:
                raise SystemExit(f"NLL checkpoint provenance mismatch for {name}")

        recovery_run_id = recovery_values.get("run_id")
        if not recovery_run_id or not re.fullmatch(
            r"shuai-[A-Za-z0-9._-]+", recovery_run_id
        ):
            raise SystemExit(f"invalid recovery run id: {recovery_run_id!r}")
        recovery_report_dir = manifest.parent.resolve() / "reports"
        expected_lock_report_paths = {
            "parent_preflight_report": (
                recovery_report_dir / f"{recovery_run_id}.parent_preflight.json"
            ).resolve(),
            "parent_postflight_report": (
                recovery_report_dir / f"{recovery_run_id}.parent_postflight.json"
            ).resolve(),
            "source_preflight_report": (
                recovery_report_dir / f"{recovery_run_id}.source_preflight.json"
            ).resolve(),
            "source_postflight_report": (
                recovery_report_dir / f"{recovery_run_id}.source_postflight.json"
            ).resolve(),
        }
        for name, expected_path in expected_lock_report_paths.items():
            actual_path = artifacts.get(name, {}).get("path")
            if actual_path != str(expected_path):
                raise SystemExit(
                    f"recovery lock artifact path mismatch for {name}: "
                    f"expected={expected_path} actual={actual_path!r}"
                )

        parent_evidence_paths = {
            "causal_log": parent_root / "logs" / f"causal_{parent_run_id}.log",
            "stateless_log": (
                parent_root / "logs" / f"stateless_{parent_run_id}.log"
            ),
            "causal_replay_log": (
                parent_root / "logs" / f"causal_replay_{parent_run_id}.log"
            ),
            "causal_memory": (
                parent_root / "logs" / f"causal_{parent_run_id}.memory.csv"
            ),
            "stateless_memory": (
                parent_root / "logs" / f"stateless_{parent_run_id}.memory.csv"
            ),
            "causal_replay_memory": (
                parent_root / "logs" / f"causal_replay_{parent_run_id}.memory.csv"
            ),
            "failure_log": parent_failure_log,
        }
        receipt_artifact_names = {
            name: f"parent_{name}" for name in parent_evidence_paths
        }
        expected_locked_artifacts = {}
        for lock_name, expected_path in parent_evidence_paths.items():
            expected_path = expected_path.resolve()
            receipt_name = receipt_artifact_names[lock_name]
            actual = artifacts.get(receipt_name)
            if actual is None or actual.get("path") != str(expected_path):
                raise SystemExit(
                    f"parent evidence artifact binding mismatch for {lock_name}: "
                    f"expected={expected_path} actual={actual}"
                )
            expected_locked_artifacts[lock_name] = {
                "path": str(expected_path),
                "sha256": _sha256(expected_path),
            }

        parent_preflight_path = expected_lock_report_paths[
            "parent_preflight_report"
        ]
        parent_postflight_path = expected_lock_report_paths[
            "parent_postflight_report"
        ]
        if parent_preflight_path.read_bytes() != parent_postflight_path.read_bytes():
            raise SystemExit("parent evidence changed during eval-only recovery")
        try:
            parent_lock = _json_file(
                parent_preflight_path, label="parent preflight lock"
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        expected_parent_lock_header = {
            "schema": "block_anchor_eval_recovery_parent_lock_v1",
            "parent_root": str(parent_root),
            "parent_run_id": parent_run_id,
            "parent_source_root": parent_values.get("source_root"),
            "parent_source_commit": parent_source_commit,
            "parent_manifest": str(parent_manifest),
            "parent_manifest_sha256": parent_manifest_sha,
            "parent_training_report": str(parent_training),
            "parent_training_report_sha256": parent_training_sha,
            "parent_failure_log": str(parent_failure_log),
            "parent_failure_log_sha256": _sha256(parent_failure_log),
            "parent_failed_phase": "nll",
            "parent_infrastructure_rc": 1,
            "parent_automatic_followup_submitted": False,
            "training_reused": True,
            "training_reexecuted": False,
        }
        actual_parent_lock_header = {
            key: parent_lock.get(key) for key in expected_parent_lock_header
        }
        for key, expected in (
            ("parent_automatic_followup_submitted", False),
            ("training_reused", True),
            ("training_reexecuted", False),
        ):
            if parent_lock.get(key) is not expected:
                raise SystemExit(f"parent lock boolean is not strict for {key}")
        if actual_parent_lock_header != expected_parent_lock_header:
            raise SystemExit(
                "parent lock header mismatch: "
                f"expected={expected_parent_lock_header} "
                f"actual={actual_parent_lock_header}"
            )
        if parent_lock.get("artifacts") != dict(
            sorted(expected_locked_artifacts.items())
        ):
            raise SystemExit("parent lock artifact inventory mismatch")
        expected_parent_checkpoint_locks = {
            name: checkpoints[name]
            for name in ("causal", "stateless", "causal_replay")
        }
        actual_parent_checkpoint_locks = parent_lock.get("checkpoint_inventories")
        if not isinstance(actual_parent_checkpoint_locks, dict):
            raise SystemExit("parent lock lacks checkpoint inventories")
        if set(actual_parent_checkpoint_locks) != set(
            expected_parent_checkpoint_locks
        ):
            raise SystemExit(
                "parent lock checkpoint set mismatch: "
                f"expected={sorted(expected_parent_checkpoint_locks)} "
                f"actual={sorted(actual_parent_checkpoint_locks)}"
            )
        for name, expected_inventory in expected_parent_checkpoint_locks.items():
            if actual_parent_checkpoint_locks.get(name) != expected_inventory:
                raise SystemExit(
                    f"parent lock checkpoint inventory mismatch for {name}"
                )
        expected_parent_lock = {
            **expected_parent_lock_header,
            "artifacts": dict(sorted(expected_locked_artifacts.items())),
            "checkpoint_inventories": dict(
                sorted(expected_parent_checkpoint_locks.items())
            ),
        }
        if parent_lock != expected_parent_lock:
            raise SystemExit("parent lock contains unexpected fields")

        source_preflight_path = expected_lock_report_paths[
            "source_preflight_report"
        ]
        source_postflight_path = expected_lock_report_paths[
            "source_postflight_report"
        ]
        if source_preflight_path.read_bytes() != source_postflight_path.read_bytes():
            raise SystemExit("source evidence changed during eval-only recovery")
        try:
            source_lock = _json_file(
                source_preflight_path, label="source preflight lock"
            )
            source_root = Path(recovery_values["source_root"]).resolve(strict=True)
            source_head = _git(source_root, "rev-parse", "HEAD")
            source_tree = _git(source_root, "rev-parse", "HEAD^{tree}")
            source_author = _git(
                source_root, "show", "-s", "--format=%an <%ae>", "HEAD"
            )
            source_status = _git(source_root, "status", "--porcelain")
            changed_paths = sorted(
                line
                for line in _git(
                    source_root,
                    "diff",
                    "--name-only",
                    parent_source_commit,
                    environment["source_commit"],
                ).splitlines()
                if line
            )
        except (KeyError, OSError, ValueError) as error:
            raise SystemExit(f"invalid frozen recovery source: {error}") from error
        if source_status:
            raise SystemExit("frozen recovery source is dirty at receipt finalization")
        if source_head != environment["source_commit"]:
            raise SystemExit(
                "frozen recovery source HEAD differs from environment: "
                f"head={source_head} environment={environment['source_commit']}"
            )
        if recovery_values.get("source_author") != source_author:
            raise SystemExit(
                "recovery manifest source author differs from Git: "
                f"manifest={recovery_values.get('source_author')!r} "
                f"git={source_author!r}"
            )
        expected_changed_paths = sorted(RECOVERY_SOURCE_CHANGED_PATHS)
        if changed_paths != expected_changed_paths:
            raise SystemExit(
                "frozen recovery source diff escaped allowlist: "
                f"expected={expected_changed_paths} actual={changed_paths}"
            )
        expected_source_lock = {
            "schema": "block_anchor_eval_recovery_source_lock_v1",
            "source_root": str(source_root),
            "source_commit": source_head,
            "source_tree": source_tree,
            "source_author": source_author,
            "parent_source_commit": parent_source_commit,
            "changed_paths": expected_changed_paths,
            "working_tree_clean": True,
        }
        if source_lock.get("working_tree_clean") is not True:
            raise SystemExit("source lock working_tree_clean is not strict true")
        if source_lock != expected_source_lock:
            raise SystemExit(
                "source lock does not match the frozen recovery repository: "
                f"expected={expected_source_lock} actual={source_lock}"
            )

        recovery = {
            "mode": "eval_only",
            "training_reused": True,
            "training_reexecuted": False,
            "parent": {
                "job": parent_run_id,
                "root": str(parent_root),
                "source_root": parent_values.get("source_root"),
                "source_commit": parent_source_commit,
                "manifest": {
                    "path": str(parent_manifest),
                    "sha256": parent_manifest_sha,
                },
                "training_report": {
                    "path": str(parent_training),
                    "sha256": parent_training_sha,
                },
            },
            "failure_signature": {
                "id": "flex_attention_qkv_dtype_mismatch_v1",
                "failed_phase": "nll",
                "infrastructure_rc": 1,
                "exception_type": "ValueError",
                "query_dtype": "float32",
                "key_dtype": "float32",
                "value_dtype": "bfloat16",
                "evidence": {
                    "path": str(parent_failure_log),
                    "sha256": _sha256(parent_failure_log),
                },
            },
            "source": {
                "root": recovery_values["source_root"],
                "commit": environment["source_commit"],
                "tree": source_tree,
                "author": source_author,
                "changed_paths": expected_changed_paths,
                "preflight_lock_sha256": _sha256(source_preflight_path),
                "postflight_lock_sha256": _sha256(source_postflight_path),
            },
            "parent_preflight_lock_sha256": _sha256(parent_preflight_path),
            "parent_postflight_lock_sha256": _sha256(parent_postflight_path),
        }

    final_manifest_fields = {
        "artifact_receipt",
        "artifact_receipt_sha256",
        "artifact_count",
        "scientific_verdict",
        "checkpoint_inventory_count",
        "generation_status",
    }
    conflicting_manifest_fields = sorted(
        final_manifest_fields.intersection(manifest_values)
    )
    if conflicting_manifest_fields:
        raise SystemExit(
            "manifest already contains receipt-finalization fields: "
            f"{conflicting_manifest_fields}"
        )

    # Re-read every signed input immediately before writing the receipt. This
    # closes the preflight/finalizer window for manifests, reports, artifacts,
    # and full checkpoint inventories instead of trusting cached hashes.
    if manifest.read_text() != manifest_text:
        raise SystemExit("manifest changed during receipt finalization")
    current_report_sha256 = {
        "verdict_report": _sha256(verdict_report),
        "training_report": _sha256(training_report),
        "nll_report": _sha256(nll_report),
        "environment_report": _sha256(environment_path),
    }
    if current_report_sha256 != input_report_sha256:
        raise SystemExit(
            "input report changed during receipt finalization: "
            f"before={input_report_sha256} after={current_report_sha256}"
        )
    for name, signed in artifacts.items():
        path = Path(str(signed["path"]))
        current = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if current != signed:
            raise SystemExit(
                f"artifact changed during receipt finalization for {name}: "
                f"before={signed} after={current}"
            )
    for name, signed_inventory in checkpoints.items():
        _, _, current_inventory = _checkpoint_inventory(
            f"{name}={signed_inventory['path']}"
        )
        if current_inventory != signed_inventory:
            raise SystemExit(
                f"checkpoint changed during receipt finalization for {name}"
            )
    if recovery is not None:
        final_source_root = Path(recovery["source"]["root"]).resolve(strict=True)
        final_source_state = {
            "commit": _git(final_source_root, "rev-parse", "HEAD"),
            "tree": _git(final_source_root, "rev-parse", "HEAD^{tree}"),
            "status": _git(final_source_root, "status", "--porcelain"),
            "changed_paths": sorted(
                line
                for line in _git(
                    final_source_root,
                    "diff",
                    "--name-only",
                    recovery["parent"]["source_commit"],
                    recovery["source"]["commit"],
                ).splitlines()
                if line
            ),
        }
        expected_final_source_state = {
            "commit": recovery["source"]["commit"],
            "tree": recovery["source"]["tree"],
            "status": "",
            "changed_paths": recovery["source"]["changed_paths"],
        }
        if final_source_state != expected_final_source_state:
            raise SystemExit(
                "frozen recovery source changed before receipt write: "
                f"expected={expected_final_source_state} "
                f"actual={final_source_state}"
            )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "schema": (
            "block_anchor_seed0_300proof_recovery_receipt_v1"
            if recovery is not None
            else "block_anchor_seed0_300proof_receipt_v2"
        ),
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
    if recovery is not None:
        payload["recovery"] = recovery
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    receipt_sha = _sha256(output)
    with manifest.open("a") as stream:
        stream.write(f"artifact_receipt={output}\n")
        stream.write(f"artifact_receipt_sha256={receipt_sha}\n")
        stream.write(f"artifact_count={len(artifacts)}\n")
        stream.write(f"scientific_verdict={args.scientific_verdict}\n")
        stream.write(f"checkpoint_inventory_count={len(checkpoints)}\n")
        stream.write("generation_status=NEEDS_GENERATION\n")
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
