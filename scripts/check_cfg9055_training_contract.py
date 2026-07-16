#!/usr/bin/env python3
"""Dependency-light runtime preflight for the 90/5/5 + Band-4 contract."""

import types

import torch

from omnivoice.blockdiff_dual import (
    KIND_EOS,
    OmniVoiceBlockDualSampleProcessor,
)


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
    ratios = {key: value / 10_000 for key, value in counts.items()}
    if not (0.88 <= ratios["C"] <= 0.92):
        raise SystemExit(f"bad C ratio: {ratios}")
    if not (0.04 <= ratios["U_shared"] <= 0.06):
        raise SystemExit(f"bad U_shared ratio: {ratios}")
    if not (0.04 <= ratios["U_drop_ref"] <= 0.06):
        raise SystemExit(f"bad U_drop_ref ratio: {ratios}")

    for q in range(1, 33):
        output = _processor(
            cfg_branch_cond_ratio=0.0,
            cfg_branch_shared_ratio=0.0,
            cfg_branch_drop_ref_ratio=1.0,
            cfg_drop_ref_q_min=q,
            cfg_drop_ref_q_max=q,
        )(_sample(f"q-{q}"))
        if output["cfg_actual_q"] != q or output["cfg_reference_frames"] != 0:
            raise SystemExit(f"bad drop-ref geometry at q={q}")
        eos_cells = int((output["loss_kind"][0] == KIND_EOS).sum().item())
        if not 1 <= eos_cells <= 4:
            raise SystemExit(f"bad Band-4 width at q={q}: {eos_cells}")

    shared = _processor(
        cfg_branch_cond_ratio=0.0,
        cfg_branch_shared_ratio=1.0,
        cfg_branch_drop_ref_ratio=0.0,
    )(_sample("shared"))
    if shared["cfg_branch"] != "U_shared" or shared["cfg_reference_frames"] <= 0:
        raise SystemExit("U_shared did not preserve a reference prefix")

    print(f"CFG9055_PREFLIGHT_OK counts={counts} ratios={ratios} q=1..32")


if __name__ == "__main__":
    main()
