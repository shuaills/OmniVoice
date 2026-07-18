#!/usr/bin/env python3
"""Fail-closed receipt check for the 300-step Markov-head A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omnivoice.training.config import TrainingConfig


def _load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--head-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-train-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--expected-main-source-commit", required=True)
    args = parser.parse_args()

    control = _load_json(args.control_config)
    head = _load_json(args.head_config)
    known = set(TrainingConfig.__annotations__)
    for name, config in (("control", control), ("head", head)):
        unknown = sorted(set(config) - known)
        if unknown:
            raise SystemExit(f"{name} has unknown TrainingConfig keys: {unknown}")

    differences = {
        key
        for key in set(control) | set(head)
        if control.get(key) != head.get(key)
    }
    expected_differences = {"block_markov_rank", "output_dir"}
    if differences != expected_differences:
        raise SystemExit(
            "A/B config drift: expected only "
            f"{sorted(expected_differences)}, got {sorted(differences)}"
        )
    if control["block_markov_rank"] != 0:
        raise SystemExit("control block_markov_rank must be 0")
    if head["block_markov_rank"] != 32:
        raise SystemExit("head block_markov_rank must be 32")

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
        "eos_band_k": 4,
        "split_loss": True,
        "split_gamma": 0.8718,
        "lambda_eos": 0.03728,
        "lambda_void": 0.14516,
        "save_steps": 300,
    }
    for key, expected in expected_common.items():
        actual = control.get(key)
        if actual != expected:
            raise SystemExit(
                f"common contract mismatch: {key} expected={expected!r} "
                f"actual={actual!r}"
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
            raise SystemExit(f"incomplete final checkpoint: {path}")
    random_states = sorted(checkpoint.glob("random_states_*.pkl"))
    if len(random_states) != 8 or any(path.stat().st_size == 0 for path in random_states):
        raise SystemExit(
            "final checkpoint must contain eight non-empty random states, got "
            f"{len(random_states)}"
        )
    model_config = _load_json(checkpoint / "config.json")
    if int(model_config.get("block_markov_rank", 0)) != 0:
        raise SystemExit("base checkpoint unexpectedly contains a Markov head")

    manifest = _manifest(args.manifest)
    for key in ("train_rc", "tee_rc", "rc"):
        if manifest.get(key) != "0":
            raise SystemExit(
                f"main training manifest is not complete: {key}={manifest.get(key)!r}"
            )
    if Path(manifest.get("output", "")).resolve() != checkpoint.parent:
        raise SystemExit("main manifest output does not own the final checkpoint")
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

    frozen_recipe = _load_json(args.base_train_config)
    saved_recipe = _load_json(checkpoint / "train_config.json")
    recipe_drift = {
        key: {"expected": expected, "actual": saved_recipe.get(key)}
        for key, expected in frozen_recipe.items()
        if key != "output_dir" and saved_recipe.get(key) != expected
    }
    if recipe_drift:
        raise SystemExit(
            "final checkpoint recipe drift: "
            + json.dumps(recipe_drift, sort_keys=True)
        )

    receipt = {
        "checkpoint": str(checkpoint),
        "model_sha256": _sha256(checkpoint / "model.safetensors"),
        "control_config_sha256": _sha256(args.control_config),
        "head_config_sha256": _sha256(args.head_config),
        "random_state_count": len(random_states),
        "main_source_commit": manifest["source_commit"],
        "main_config_sha256": manifest["config_sha256"],
        "data_config_sha256": manifest["data_config_sha256"],
        "only_config_differences": sorted(differences),
    }
    print("BLOCK_MARKOV_AB_CONTRACT_OK " + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
