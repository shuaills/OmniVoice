#!/usr/bin/env python3
"""Fail fast unless a config is an approved plain correct-case EOS run."""

import argparse
import ast
import json
from pathlib import Path

EXPECTED = {
    "audio_vocab_size": 1026,
    "audio_mask_id": 1024,
    "learning_rate": 0.00003,
    "lr_scheduler_type": "constant",
    "warmup_steps": 50,
    "save_steps": 300,
    "block_training": True,
    "block_size": 32,
    "block_scheme": "dual",
    "eos_decouple_silence": False,
    "split_loss": False,
}
ALLOWED_TRAJECTORIES = {
    (
        None,
        "/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block",
        300,
    ),
    (
        "/opt/gpfs/users/shuai/experiments/eos-correct-case-20260721/ce300-v1/train/checkpoint-300",
        "/opt/gpfs/users/shuai/experiments/eos-correct-case-20260721/ce300-v1/train/checkpoint-300",
        600,
    ),
}
FORBIDDEN = (
    "generated_prefix",
    "teacher_kl",
    "terminal_recovery",
    "lambda_eos",
    "lambda_void",
    "reward",
)


def _training_config_fields() -> set[str]:
    source = Path(__file__).resolve().parents[1] / "omnivoice/training/config.py"
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


def validate(config: dict) -> list[str]:
    failures = []
    unknown = sorted(set(config) - _training_config_fields())
    if unknown:
        failures.append(f"unknown config keys: {unknown}")
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            failures.append(
                f"{key} must be {expected!r}, got {config.get(key)!r}"
            )
    if config.get("force_lr_from_config_on_resume", False) is not False:
        failures.append("force_lr_from_config_on_resume must remain false")
    trajectory = (
        config.get("resume_from_checkpoint"),
        config.get("init_from_checkpoint"),
        config.get("steps"),
    )
    if trajectory not in ALLOWED_TRAJECTORIES:
        failures.append(
            "resume/init checkpoints and steps must identify the approved 0->300 or "
            f"300->600 trajectory, got {trajectory!r}"
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
