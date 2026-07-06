#!/usr/bin/env python3
"""B2 GPU gates on the real migrated checkpoint (single 4090, sdpa, fp32).

  gate 3g -- cache vs recompute with the 0.8B migrated ckpt: identical
             tokens (greedy), max |dlogit| reported (fp32 tolerance gate)
  gate 5  -- end-to-end training forward: dual processor -> packing
             collator -> model loss under the dense block-causal mask;
             finite loss, supervision counted, and a corruption check that
             breaking the mask (full bidirectional) changes the loss
             (i.e. the mask is actually load-bearing)

Usage: python tests/gpu_gate_b2.py --ckpt <migrated dir>
"""

import argparse
import random

import torch

from omnivoice.blockdiff_dual import (
    OmniVoiceBlockDualSampleProcessor,
    _decode_block_causal,
    build_block_causal_attn_mask,
)
from omnivoice.data.collator import PackingDataCollator
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    model = OmniVoice.from_pretrained(
        args.ckpt, device_map="cuda:0", dtype=torch.float32,
        attn_implementation="sdpa",
    )
    model.eval()
    tok = model.text_tokenizer if hasattr(model, "text_tokenizer") else None
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.ckpt)

    # ---- gate 3g: cache equivalence on the real model ----
    gen = OmniVoiceGenerationConfig()
    gen.class_temperature = 0.0
    gen.position_temperature = 0.0
    C = model.config.num_audio_codebook

    text_ids = tok("<|text_start|>你好世界<|text_end|>", return_tensors="pt")
    prefix = text_ids.input_ids.repeat(C, 1)

    for scale, tag in ((0.0, "cond"), (2.0, "cfg")):
        gen.guidance_scale = scale
        toks, traces = {}, {}
        for use_cache in (True, False):
            torch.manual_seed(20260706)
            trace = []
            out, stats = _decode_block_causal(
                model, prefix, gen, block_size=32, max_blocks=2,
                num_step_per_block=4, use_kv_cache=use_cache,
                logit_trace=trace,
            )
            toks[use_cache] = out.cpu()
            traces[use_cache] = trace
        same = torch.equal(toks[True], toks[False])
        dmax = max(
            (a - b).abs().max().item()
            for a, b in zip(traces[True], traces[False])
        )
        legal = int(toks[True].max()) <= model.config.audio_vocab_size - 1
        print(f"gate3g [{tag}]: tokens_equal={same} shape={tuple(toks[True].shape)} "
              f"max|dlogit|={dmax:.3e} legal_ids={legal}")
        assert same and legal

    # ---- gate 5: end-to-end training forward under the dense mask ----
    proc = OmniVoiceBlockDualSampleProcessor(
        text_tokenizer=tok, num_channels=C,
        audio_mask_id=model.config.audio_mask_id,
        prompt_ratio_range=(0.0, 0.3), mask_ratio_range=(0.0, 1.0),
        drop_cond_ratio=0.1, language_ratio=0.5, use_pinyin_ratio=0.0,
        instruct_ratio=0.0, only_instruct_ratio=0.0, block_size=32,
    )
    random.seed(42)
    torch.manual_seed(42)
    samples = []
    for i, (txt, T) in enumerate(
        [("今天天气不错我们出去走走", 90), ("block causal attention works", 61)]
    ):
        samples.append(
            proc(
                {
                    "audio_tokens": torch.randint(0, 1024, (C, T)),
                    "label": {"text": txt, "language_id": "zh" if i == 0 else "en"},
                }
            )
        )
    total = sum(s["length"] for s in samples)
    coll = PackingDataCollator(proc, batch_tokens=total)
    batch = coll(samples)
    attn = build_block_causal_attn_mask(
        batch["document_ids"][0], batch["copy_tags"][0], batch["block_ids"][0]
    ).cuda()
    dev = {
        k: v.cuda()
        for k, v in batch.items()
        if k in ("input_ids", "labels", "audio_mask", "position_ids")
    }
    with torch.no_grad():
        out = model(**dev, attention_mask=attn)
        full = torch.ones_like(attn)
        out_bidir = model(**dev, attention_mask=full)
    n_sup = int((batch["labels"] != -100).sum())
    dloss = abs(out.loss.item() - out_bidir.loss.item())
    print(f"gate5: loss={out.loss.item():.6f} finite={torch.isfinite(out.loss).item()} "
          f"supervised_cells={n_sup} |loss-bidir_loss|={dloss:.6f} (must be >0)")
    assert torch.isfinite(out.loss) and n_sup > 0 and dloss > 1e-6

    print("ALL B2 GPU GATES PASSED")


if __name__ == "__main__":
    main()
