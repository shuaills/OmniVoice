#!/usr/bin/env python3
"""Dependency-light preflight for the 90/10/0 CFG + Band-4 contract."""

import types

import torch

from omnivoice.blockdiff_dual import KIND_EOS, OmniVoiceBlockDualSampleProcessor


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, _text, return_tensors="pt"):
        return types.SimpleNamespace(
            input_ids=torch.tensor([[11, 12, 13]], dtype=torch.long)
        )


def _processor(**overrides):
    kwargs = {
        "text_tokenizer": _Tokenizer(),
        "num_channels": 8,
        "audio_mask_id": 1024,
        "prompt_ratio_range": (0.3, 0.3),
        "mask_ratio_range": (0.0, 1.0),
        "drop_cond_ratio": 0.1,
        "language_ratio": 0.0,
        "use_pinyin_ratio": 0.0,
        "instruct_ratio": 0.0,
        "only_instruct_ratio": 0.0,
        "block_size": 32,
        "eos_decouple_silence": True,
        "eos_band_k": 4,
        "silence_void_window": 32,
        "cfg_branch_training": True,
        "cfg_branch_cond_ratio": 0.90,
        "cfg_branch_shared_ratio": 0.10,
        "cfg_branch_drop_ref_ratio": 0.0,
        "cfg_branch_seed": 42,
    }
    kwargs.update(overrides)
    return OmniVoiceBlockDualSampleProcessor(**kwargs)


def _sample(sample_id, frames=128):
    audio = torch.arange(8 * frames, dtype=torch.long).reshape(8, frames) % 1024
    return {"audio_tokens": audio, "label": {"id": sample_id, "text": "preflight"}}


def main():
    processor = _processor()
    counts = {"C": 0, "U_shared": 0, "U_drop_ref": 0}
    for index in range(10_000):
        rng = processor._cfg_rng(_sample(f"branch-{index}"))
        counts[processor._choose_cfg_branch(rng)] += 1
    expected_counts = {"C": 9021, "U_shared": 979, "U_drop_ref": 0}
    if counts != expected_counts:
        raise SystemExit(f"deterministic branch count mismatch: expected={expected_counts} actual={counts}")
    ratios = {key: value / 10_000 for key, value in counts.items()}

    shared = _processor(
        cfg_branch_cond_ratio=0.0,
        cfg_branch_shared_ratio=1.0,
        cfg_branch_drop_ref_ratio=0.0,
    )(_sample("shared"))
    if shared["cfg_branch"] != "U_shared" or shared["cfg_reference_frames"] <= 0:
        raise SystemExit("U_shared did not preserve a reference prefix")
    if shared["cfg_target_frames"] <= 0:
        raise SystemExit("U_shared did not preserve a target suffix")
    if shared["cfg_requested_q"] is not None or shared["cfg_actual_q"] is not None:
        raise SystemExit("q geometry must remain inactive for U_shared")
    eos_cells = int((shared["loss_kind"][0] == KIND_EOS).sum().item())
    if not 1 <= eos_cells <= 4:
        raise SystemExit(f"bad Band-4 width: {eos_cells}")

    print(f"CFG90100_PREFLIGHT_OK counts={counts} ratios={ratios} band_width={eos_cells}")


if __name__ == "__main__":
    main()
