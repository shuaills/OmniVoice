#!/usr/bin/env python3
"""Fail fast unless a config is an approved plain correct-case EOS run."""

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_300 = ROOT / "examples/config/train_config_correct_case_eos_300.json"
CHECKPOINT_300 = (
    "/opt/gpfs/users/shuai/experiments/eos-correct-case-20260721/"
    "ce300-v1/train/checkpoint-300"
)
FORBIDDEN = (
    "generated_prefix",
    "teacher_kl",
    "terminal_recovery",
    "lambda_eos",
    "lambda_void",
    "reward",
)


def _training_config_fields() -> set[str]:
    source = ROOT / "omnivoice/training/config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TrainingConfig":
            return {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
            }
    raise RuntimeError("TrainingConfig class not found")


def _approved_configs() -> list[dict]:
    step_300 = json.loads(CANONICAL_300.read_text(encoding="utf-8"))
    step_600 = dict(step_300)
    step_600.update(
        {
            "resume_from_checkpoint": CHECKPOINT_300,
            "init_from_checkpoint": CHECKPOINT_300,
            "force_lr_from_config_on_resume": False,
            "steps": 600,
        }
    )
    return [step_300, step_600]


def _trajectory(config: dict) -> tuple:
    return (
        config.get("resume_from_checkpoint"),
        config.get("init_from_checkpoint"),
        config.get("steps"),
    )


def validate(config: dict) -> list[str]:
    failures = []
    unknown = sorted(set(config) - _training_config_fields())
    if unknown:
        failures.append(f"unknown config keys: {unknown}")
    approved = _approved_configs()
    if config not in approved:
        matching = [candidate for candidate in approved if _trajectory(candidate) == _trajectory(config)]
        if matching:
            expected = matching[0]
            mismatches = {
                key: {"expected": expected.get(key), "actual": config.get(key)}
                for key in sorted(set(expected) | set(config))
                if expected.get(key) != config.get(key) or (key in expected) != (key in config)
            }
            failures.append(f"config differs from canonical trajectory: {mismatches}")
        else:
            failures.append(
                "resume/init checkpoints and steps must identify the approved 0->300 "
                f"or 300->600 trajectory, got {_trajectory(config)!r}"
            )
    present = sorted(
        key for key in config if any(token in key for token in FORBIDDEN)
    )
    if present:
        failures.append(f"forbidden objective keys: {present}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    failures = validate(config)
    if failures:
        raise SystemExit("CONFIG_PREFLIGHT_FAILED: " + "; ".join(failures))
    print("CORRECT_CASE_EOS_CONFIG_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
