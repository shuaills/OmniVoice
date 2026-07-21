#!/usr/bin/env python3
"""Fail fast unless a config is the plain 300-step correct-case EOS run."""

import argparse
import ast
import json
from pathlib import Path

EXPECTED = {
    "audio_vocab_size": 1026,
    "audio_mask_id": 1024,
    "resume_from_checkpoint": None,
    "init_from_checkpoint": (
        "/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block"
    ),
    "steps": 300,
    "block_training": True,
    "block_size": 32,
    "block_scheme": "dual",
    "eos_decouple_silence": False,
    "split_loss": False,
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
