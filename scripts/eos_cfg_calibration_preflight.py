#!/usr/bin/env python3
"""Dependency-free Torch preflight for the EOS/CFG calibration OMS job."""

import runpy
from pathlib import Path
from types import SimpleNamespace

from omnivoice.models.omnivoice import (
    OmniVoice,
    OmniVoiceGenerationConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def run_calibration_numeric_gates() -> None:
    namespace = runpy.run_path(
        str(ROOT / "tests" / "test_block_eos_cfg_calibration.py")
    )
    names = sorted(
        name
        for name, value in namespace.items()
        if name.startswith("test_") and callable(value)
    )
    if not names:
        raise RuntimeError("no EOS calibration numeric gates discovered")
    for name in names:
        namespace[name]()
        print(f"PASS {name}", flush=True)


def run_generation_routing_gates() -> None:
    official = SimpleNamespace(
        config=SimpleNamespace(audio_mask_id=1024, audio_vocab_size=1025)
    )
    block = SimpleNamespace(
        config=SimpleNamespace(audio_mask_id=1024, audio_vocab_size=1026)
    )

    legacy = OmniVoiceGenerationConfig()
    OmniVoice._assert_block_only_generation_options(official, legacy)
    OmniVoice._assert_legacy_fixed_canvas_allowed(official, legacy)

    try:
        OmniVoice._assert_legacy_fixed_canvas_allowed(block, legacy)
    except RuntimeError as exc:
        if "fixed-canvas" not in str(exc):
            raise
    else:
        raise AssertionError("block checkpoint legacy routing did not fail")

    allowed = OmniVoiceGenerationConfig(
        allow_block_checkpoint_fixed_canvas=True
    )
    OmniVoice._assert_legacy_fixed_canvas_allowed(block, allowed)

    block_only = OmniVoiceGenerationConfig(eos_cfg_calibration="renorm")
    try:
        OmniVoice._assert_block_only_generation_options(official, block_only)
    except ValueError as exc:
        if "block-causal decoder option" not in str(exc):
            raise
    else:
        raise AssertionError("fixed-canvas decoder accepted block-only option")

    print("PASS generation routing gates", flush=True)


def main() -> None:
    run_calibration_numeric_gates()
    run_generation_routing_gates()
    print("ALL EOS CFG PREFLIGHT GATES PASSED", flush=True)


if __name__ == "__main__":
    main()
