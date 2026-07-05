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


def load(path, dtype=torch.float16):
    model = OmniVoice.from_pretrained(path, device_map="cuda:0", dtype=dtype)
    model.eval()
    return model


@torch.no_grad()
def forward_logits(model, seed=7):
    """One deterministic forward on a synthetic batch; returns [B,C,S,V]."""
    g = torch.Generator().manual_seed(seed)
    C = model.config.num_audio_codebook
    S, B = 64, 2
    ids = torch.randint(0, model.config.audio_mask_id, (B, C, S), generator=g)
    ids[:, :, S // 2 :] = model.config.audio_mask_id  # half masked
    audio_mask = torch.ones(B, S, dtype=torch.bool)
    audio_mask[:, :6] = False  # a fake text prefix region
    return model(
        input_ids=ids.to(model.device), audio_mask=audio_mask.to(model.device)
    ).logits.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--migrated", required=True)
    args = ap.parse_args()

    # Equivalence gate in fp32: identical logits on the shared vocab slice.
    # (Token-level identity across full sampling is chaotic under fp16 GEMM
    # shape differences and is NOT the right assertion.)
    print("== equivalence (fp32 single forward) ==")
    m_off = load(args.official, dtype=torch.float32)
    lo = forward_logits(m_off)
    del m_off
    torch.cuda.empty_cache()
    m_mig32 = load(args.migrated, dtype=torch.float32)
    lm = forward_logits(m_mig32)
    del m_mig32
    torch.cuda.empty_cache()
    V = lo.shape[-1]
    diff = (lo - lm[..., :V]).abs().max().item()
    print(f"max |dlogit| on shared vocab slice: {diff:.3e}")
    assert diff < 1e-3, f"migrated ckpt logits diverge: {diff}"
    print("EQUIVALENCE OK: migrated ckpt matches official logits (fp32)")

    print("== loading migrated (fp16) ==")
    m_mig = load(args.migrated)

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
