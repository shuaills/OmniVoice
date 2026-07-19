#!/usr/bin/env python3
"""Fail-closed contract for the seed-0 causal/stateless anchor proof."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from omnivoice.training.config import TrainingConfig


CAMPAIGN_OVERRIDE_ALLOWLIST = frozenset(
    {
        "block_anchor_freeze_base",
        "block_anchor_mode",
        "block_anchor_proposal_dim",
        "block_anchor_scan_dim",
        "block_anchor_stride",
        "eval_steps",
        "init_from_checkpoint",
        "keep_last_n_checkpoints",
        "learning_rate",
        "logging_steps",
        "lr_scheduler_type",
        "output_dir",
        "perf_grad_checkpoint",
        "save_steps",
        "steps",
        "warmup_ratio",
        "warmup_steps",
        "warmup_type",
    }
)
FROZEN_CHECKPOINT_ALLOWLIST = frozenset({"output_dir"})


def _json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expanded(path: Path) -> dict:
    return asdict(TrainingConfig.from_json(str(path)))


def _data_manifest_inventory(data_recipe: dict) -> tuple[list[dict], str]:
    files: list[dict] = []
    for split in ("train", "dev"):
        for entry_index, entry in enumerate(data_recipe.get(split, [])):
            paths = entry.get("manifest_path", [])
            if isinstance(paths, str):
                paths = [paths]
            for path_index, raw_path in enumerate(paths):
                path = Path(raw_path).resolve()
                if not path.is_file() or path.stat().st_size == 0:
                    raise SystemExit(f"data manifest is missing or empty: {path}")
                files.append(
                    {
                        "split": split,
                        "entry_index": entry_index,
                        "path_index": path_index,
                        "language_id": entry.get("language_id"),
                        "repeat": entry.get("repeat", 1),
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    if not files:
        raise SystemExit("data recipe contains no manifest files")
    canonical = json.dumps(
        files, separators=(",", ":"), sort_keys=True
    ).encode()
    return files, hashlib.sha256(canonical).hexdigest()


def _drift(left: dict, right: dict, allowlist: frozenset[str]) -> dict:
    keys = set(left) | set(right)
    return {
        key: {"expected": left.get(key), "actual": right.get(key)}
        for key in sorted(keys - allowlist)
        if left.get(key) != right.get(key)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-config", type=Path, required=True)
    parser.add_argument("--stateless-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-train-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--expected-main-source-commit", required=True)
    args = parser.parse_args()

    causal = _json(args.causal_config)
    stateless = _json(args.stateless_config)
    known = set(TrainingConfig.__annotations__)
    for name, config in (("causal", causal), ("stateless", stateless)):
        unknown = sorted(set(config) - known)
        if unknown:
            raise SystemExit(f"{name} has unknown TrainingConfig keys: {unknown}")
        if "block_anchor_topk" in config:
            raise SystemExit(
                "block_anchor_topk is forbidden: this proof uses the full "
                "acoustic softmax expectation"
            )

    differences = {
        key
        for key in set(causal) | set(stateless)
        if causal.get(key) != stateless.get(key)
    }
    expected_differences = {"block_anchor_mode", "output_dir"}
    if differences != expected_differences:
        raise SystemExit(
            "causal/stateless config drift: expected only "
            f"{sorted(expected_differences)}, got {sorted(differences)}"
        )
    if causal.get("block_anchor_mode") != "causal":
        raise SystemExit("causal arm must use block_anchor_mode='causal'")
    if stateless.get("block_anchor_mode") != "stateless":
        raise SystemExit("stateless arm must use block_anchor_mode='stateless'")

    checkpoint = args.checkpoint.resolve()
    expected_common = {
        "init_from_checkpoint": str(checkpoint),
        "resume_from_checkpoint": None,
        "steps": 300,
        "learning_rate": 3e-5,
        "warmup_type": "steps",
        "warmup_ratio": 0.0,
        "warmup_steps": 50,
        "lr_scheduler_type": "constant",
        "batch_tokens": 15648,
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
        "seed": 42,
        "cfg_branch_training": True,
        "cfg_branch_cond_ratio": 0.90,
        "cfg_branch_shared_ratio": 0.10,
        "cfg_branch_drop_ref_ratio": 0.0,
        "block_training": True,
        "block_size": 32,
        "block_scheme": "dual",
        "block_markov_rank": 0,
        "block_anchor_scan_dim": 64,
        "block_anchor_proposal_dim": 32,
        "block_anchor_stride": 8,
        "block_anchor_freeze_base": True,
        "eos_band_k": 4,
        "split_loss": True,
        "split_gamma": 0.8718,
        "lambda_eos": 0.03728,
        "lambda_void": 0.14516,
        "perf_grad_checkpoint": False,
        "save_steps": 300,
    }
    for key, expected in expected_common.items():
        actual = causal.get(key)
        if actual != expected:
            raise SystemExit(
                f"common contract mismatch: {key} expected={expected!r} "
                f"actual={actual!r}"
            )

    frozen_expanded = _expanded(args.base_train_config)
    causal_expanded = _expanded(args.causal_config)
    campaign_drift = _drift(
        frozen_expanded,
        causal_expanded,
        CAMPAIGN_OVERRIDE_ALLOWLIST,
    )
    if campaign_drift:
        raise SystemExit(
            "campaign config drift outside explicit allowlist: "
            + json.dumps(campaign_drift, sort_keys=True)
        )

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
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"incomplete frozen checkpoint: {path}")
    random_states = sorted(checkpoint.glob("random_states_*.pkl"))
    if len(random_states) != 8 or any(path.stat().st_size == 0 for path in random_states):
        raise SystemExit(
            "frozen checkpoint must contain eight non-empty random states, got "
            f"{len(random_states)}"
        )
    model_config = _json(checkpoint / "config.json")
    if int(model_config.get("block_markov_rank", 0)) != 0:
        raise SystemExit("frozen checkpoint unexpectedly contains a Markov head")
    if int(model_config.get("block_anchor_scan_dim", 0)) != 0:
        raise SystemExit("frozen checkpoint unexpectedly contains an anchor head")

    manifest = _manifest(args.manifest)
    for key in ("train_rc", "tee_rc", "rc"):
        if manifest.get(key) != "0":
            raise SystemExit(
                f"main training manifest is incomplete: {key}={manifest.get(key)!r}"
            )
    if Path(manifest.get("output", "")).resolve() != checkpoint.parent:
        raise SystemExit("main manifest output does not own the frozen checkpoint")
    if manifest.get("source_commit") != args.expected_main_source_commit:
        raise SystemExit(
            "main source commit mismatch: "
            f"expected={args.expected_main_source_commit!r} "
            f"actual={manifest.get('source_commit')!r}"
        )
    if manifest.get("config_sha256") != _sha256(args.base_train_config):
        raise SystemExit("main train-config SHA does not match frozen receipt")
    if manifest.get("data_config_sha256") != _sha256(args.data_config):
        raise SystemExit("main data-config SHA does not match frozen receipt")

    saved_expanded = _expanded(checkpoint / "train_config.json")
    checkpoint_drift = _drift(
        frozen_expanded,
        saved_expanded,
        FROZEN_CHECKPOINT_ALLOWLIST,
    )
    if checkpoint_drift:
        raise SystemExit(
            "frozen checkpoint recipe drift: "
            + json.dumps(checkpoint_drift, sort_keys=True)
        )

    data_recipe = _json(args.data_config)
    data_manifest_files, data_manifest_inventory_sha256 = (
        _data_manifest_inventory(data_recipe)
    )
    dev_languages = {
        entry.get("language_id") for entry in data_recipe.get("dev", [])
    }
    if dev_languages != {"en"}:
        raise SystemExit(
            "mechanism probe must be explicitly English-only, got dev languages "
            f"{sorted(str(value) for value in dev_languages)}"
        )

    receipt = {
        "checkpoint": str(checkpoint),
        "model_sha256": _sha256(checkpoint / "model.safetensors"),
        "causal_config_sha256": _sha256(args.causal_config),
        "stateless_config_sha256": _sha256(args.stateless_config),
        "random_state_count": len(random_states),
        "main_source_commit": manifest["source_commit"],
        "main_config_sha256": manifest["config_sha256"],
        "data_config_sha256": manifest["data_config_sha256"],
        "data_manifest_file_count": len(data_manifest_files),
        "data_manifest_inventory_sha256": data_manifest_inventory_sha256,
        "data_manifest_files": data_manifest_files,
        "only_config_differences": sorted(differences),
        "proposal_contract": "full_acoustic_softmax_expectation",
        "seed_scope": "seed0_only",
        "seed_index": 0,
        "train_seed": 42,
        "eval_seed": 20260719,
        "english_only_mechanism_probe": True,
        "campaign_override_allowlist": sorted(CAMPAIGN_OVERRIDE_ALLOWLIST),
        "frozen_checkpoint_allowlist": sorted(FROZEN_CHECKPOINT_ALLOWLIST),
    }
    print("BLOCK_ANCHOR_CONTRACT_OK " + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
