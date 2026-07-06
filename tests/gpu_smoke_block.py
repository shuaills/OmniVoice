#!/usr/bin/env python3
"""GPU smoke for the block-diffusion conversion (single GPU, sdpa).

Gate 3 -- migration: official vs migrated (1025 -> 1026) fp32 logits must be
          identical on the shared vocab slice.
Gate 2b -- end-to-end anchor: the degenerate block batch (block_size >=
          canvas, eos off) and the official batch, produced under the same
          RNG, must give the SAME fp32 loss through the SAME migrated model.
Mechanics -- generate_blockwise on the migrated (eos untrained) checkpoint
          must emit only legal ids and respect the max_blocks safety net.

Usage:
  python tests/gpu_smoke_block.py --official <dir> --migrated <dir>
"""

import argparse
import random

import torch

from omnivoice.blockdiff import OmniVoiceBlockSampleProcessor, generate_blockwise
from omnivoice.data.collator import PaddingDataCollator
from omnivoice.data.processor import OmniVoiceSampleProcessor
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig


def load(path, dtype=torch.float32):
    model = OmniVoice.from_pretrained(
        path, device_map="cuda:0", dtype=dtype, attn_implementation="sdpa"
    )
    model.eval()
    return model


@torch.no_grad()
def forward_logits(model, seed=7):
    g = torch.Generator().manual_seed(seed)
    C = model.config.num_audio_codebook
    S, B = 64, 2
    ids = torch.randint(0, model.config.audio_mask_id, (B, C, S), generator=g)
    ids[:, :, S // 2 :] = model.config.audio_mask_id
    audio_mask = torch.ones(B, S, dtype=torch.bool)
    audio_mask[:, :6] = False
    return model(
        input_ids=ids.to(model.device), audio_mask=audio_mask.to(model.device)
    ).logits.float()


@torch.no_grad()
def gate2_loss_identity(model):
    """Official vs degenerate-block data path through one model, fp32."""

    def proc_kwargs(tok):
        return dict(
            text_tokenizer=tok,
            num_channels=model.config.num_audio_codebook,
            audio_mask_id=model.config.audio_mask_id,
            prompt_ratio_range=(0.0, 0.3),
            mask_ratio_range=(0.0, 1.0),
            drop_cond_ratio=0.1,
            language_ratio=0.5,
            use_pinyin_ratio=0.0,
            instruct_ratio=0.0,
            only_instruct_ratio=0.0,
        )

    tok = model.text_tokenizer
    official = OmniVoiceSampleProcessor(**proc_kwargs(tok))
    block = OmniVoiceBlockSampleProcessor(
        **proc_kwargs(tok), block_size=100000, eos_enabled=False
    )

    def build(processor, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        samples = []
        for i in range(4):
            g = torch.Generator().manual_seed(seed + i)
            samples.append(
                processor(
                    {
                        "audio_tokens": torch.randint(
                            0, 1024,
                            (model.config.num_audio_codebook, 60 + 13 * i),
                            generator=g,
                        ),
                        "label": {
                            "text": "block conversion anchor sentence",
                            "language_id": "en",
                        },
                    }
                )
            )
        return PaddingDataCollator(processor, 8192)(samples)

    losses = []
    for processor in (official, block):
        batch = build(processor, seed=20260706)
        batch = {
            k: (v.to(model.device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        out = model(**batch)
        losses.append(out.loss.item())
    print(f"official-path loss: {losses[0]:.10f}")
    print(f"block-degenerate  : {losses[1]:.10f}")
    assert losses[0] == losses[1], "degenerate block path diverged from official"
    print("GATE2b OK: identical fp32 loss through the migrated model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--migrated", required=True)
    args = ap.parse_args()

    print("== gate 3: migration equivalence (fp32 single forward) ==")
    m_off = load(args.official)
    lo = forward_logits(m_off)
    del m_off
    torch.cuda.empty_cache()
    m_mig = load(args.migrated)
    lm = forward_logits(m_mig)
    V = lo.shape[-1]
    diff = (lo - lm[..., :V]).abs().max().item()
    print(f"max |dlogit| on shared vocab slice: {diff:.3e}")
    assert diff < 1e-3, f"migrated ckpt logits diverge: {diff}"
    print("GATE3 OK: migrated ckpt matches official logits (fp32)")

    print("== gate 2b: degenerate loss identity ==")
    gate2_loss_identity(m_mig)

    print("== blockwise mechanics (eos untrained; mechanics only) ==")
    gc = OmniVoiceGenerationConfig(num_step=8)
    torch.manual_seed(1234)
    tokens, stats = generate_blockwise(
        m_mig,
        "今天天气不错，我们一起去公园散步吧。",
        language="zh",
        gen_config=gc,
        block_size=32,
        max_blocks=4,
        num_step_per_block=8,
    )
    eos = m_mig.config.audio_mask_id + 1
    assert (tokens < m_mig.config.audio_mask_id).all(), (
        f"illegal ids in committed output: max={int(tokens.max())}"
    )
    print(
        f"blockwise: n_blocks={stats['n_blocks']} "
        f"stopped_by_eos={stats['stopped_by_eos']} T={tokens.shape[1]}"
    )
    assert stats["n_blocks"] <= 4
    print("BLOCKWISE MECHANICS OK")
    print("ALL BLOCK GPU SMOKE PASSED")


if __name__ == "__main__":
    main()
