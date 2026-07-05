#!/usr/bin/env python3
"""GPU smoke for the elastic canvas (single GPU).

1) Equivalence: official ckpt vs migrated ckpt in OFFICIAL (fixed-canvas)
   mode with identical seeds must produce identical tokens.
2) Elastic mechanics: elastic mode on the migrated (not yet fine-tuned) ckpt
   must execute without errors and emit only valid codec ids. Quality is NOT
   asserted — specials are untrained until E1.

Usage:
  python tests/gpu_smoke_elastic.py --official <dir> --migrated <dir>
"""

import argparse

import torch

from omnivoice.models.omnivoice import (
    GenerationTask,
    OmniVoice,
    OmniVoiceGenerationConfig,
)

TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "今天天气不错，我们一起去公园散步吧。",
]
LANGS = ["en", "zh"]
TARGET_LENS = [140, 90]


def make_task(target_lens):
    return GenerationTask(
        batch_size=len(TEXTS),
        texts=list(TEXTS),
        target_lens=list(target_lens),
        langs=list(LANGS),
        instructs=[None] * len(TEXTS),
        ref_texts=[None] * len(TEXTS),
        ref_audio_tokens=[None] * len(TEXTS),
        ref_rms=[None] * len(TEXTS),
    )


def load(path):
    model = OmniVoice.from_pretrained(path, device_map="cuda:0", dtype=torch.float16)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--migrated", required=True)
    args = ap.parse_args()

    cfg = OmniVoiceGenerationConfig(num_step=16)

    print("== loading official ==")
    m_off = load(args.official)
    torch.manual_seed(1234)
    toks_off = m_off._generate_iterative(make_task(TARGET_LENS), cfg)
    del m_off
    torch.cuda.empty_cache()

    print("== loading migrated ==")
    m_mig = load(args.migrated)
    torch.manual_seed(1234)
    toks_mig = m_mig._generate_iterative(make_task(TARGET_LENS), cfg)

    for i, (a, b) in enumerate(zip(toks_off, toks_mig)):
        assert a.shape == b.shape, f"[{i}] shape {a.shape} vs {b.shape}"
        n_diff = int((a != b).sum())
        print(f"sample {i}: shape={tuple(a.shape)} diff_cells={n_diff}")
        assert n_diff == 0, f"[{i}] official-mode outputs differ after migration"
    print("EQUIVALENCE OK: migrated ckpt is byte-identical in official mode")

    print("== elastic mechanics (untrained specials; mechanics only) ==")
    e_cfg = OmniVoiceGenerationConfig(num_step=16, elastic=True)
    for scale, tag in ((0.6, "short"), (1.0, "exact"), (1.4, "long")):
        lens = [max(8, int(t * scale)) for t in TARGET_LENS]
        torch.manual_seed(1234)
        outs = m_mig._generate_iterative(make_task(lens), e_cfg)
        for i, t in enumerate(outs):
            assert (t < m_mig.config.audio_mask_id).all(), (
                f"invalid ids in elastic output: max={int(t.max())}"
            )
            print(
                f"elastic[{tag}] sample {i}: init={lens[i]} final={t.shape[1]}"
            )
    print("ELASTIC MECHANICS OK")
    print("ALL GPU SMOKE PASSED")


if __name__ == "__main__":
    main()
