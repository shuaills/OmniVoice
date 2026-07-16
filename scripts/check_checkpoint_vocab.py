#!/usr/bin/env python3
"""Fail-fast preflight for OmniVoice checkpoint text-vocabulary integrity."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from omnivoice.training.builder import (
    TextVocabContractError,
    inspect_checkpoint_text_vocab_contract,
)


def _checkpoint_from_train_config(train_config: Path) -> str:
    with train_config.open() as f:
        config = json.load(f)
    checkpoint = config.get("init_from_checkpoint")
    if not checkpoint:
        raise TextVocabContractError(
            f"{train_config} has no non-empty init_from_checkpoint"
        )
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=str)
    source.add_argument("--train-config", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        checkpoint = (
            args.checkpoint
            if args.checkpoint is not None
            else _checkpoint_from_train_config(args.train_config)
        )
        if not Path(checkpoint).is_dir():
            raise TextVocabContractError(
                "preflight requires an existing local checkpoint directory: "
                f"{checkpoint}"
            )
        contract = inspect_checkpoint_text_vocab_contract(checkpoint)
    except (OSError, ValueError, TextVocabContractError) as error:
        print(f"VOCAB_PREFLIGHT_FAILED: {error}", file=sys.stderr)
        return 1

    result = asdict(contract)
    result["requires_config_shim"] = contract.requires_config_shim
    print("VOCAB_PREFLIGHT_OK " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
