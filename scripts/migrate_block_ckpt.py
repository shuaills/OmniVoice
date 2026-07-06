#!/usr/bin/env python3
"""Migrate an OmniVoice checkpoint to the block-diffusion vocabulary.

Grows the fused audio embedding/head tables from audio_vocab_size (1025) to
audio_vocab_size + 1 (1026), preserving the ``row = c * vocab + v`` block
layout so all original logits are bit-identical. The new [eos] row is
initialised from each codebook block's weight statistics.

Deliberately migrates from the OFFICIAL checkpoint, not OmniVoice-elastic:
the block arm and the elastic arm stay single-variable independent.

Usage:
    python scripts/migrate_block_ckpt.py \
        --src <official ckpt dir> --dst pretrained_models/OmniVoice-block
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from omnivoice.elastic import migrate_state_dict

NUM_BLOCK_CLASSES = 1  # [eos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=20260706)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    old_vocab = cfg["audio_vocab_size"]
    num_cb = cfg["num_audio_codebook"]
    print(f"vocab {old_vocab} -> {old_vocab + NUM_BLOCK_CLASSES}, C={num_cb}")

    st_files = sorted(src.glob("*.safetensors"))
    if st_files:
        from safetensors.torch import load_file, save_file

        assert len(st_files) == 1, "sharded checkpoints not supported yet"
        sd = load_file(st_files[0])
        sd = migrate_state_dict(
            sd, num_cb, old_vocab, num_new_classes=NUM_BLOCK_CLASSES
        )
        save_file(sd, dst / st_files[0].name, metadata={"format": "pt"})
        migrated_name = st_files[0].name
    else:
        bin_files = sorted(src.glob("pytorch_model*.bin"))
        assert len(bin_files) == 1, "expected exactly one weight file"
        sd = torch.load(bin_files[0], map_location="cpu", weights_only=True)
        sd = migrate_state_dict(
            sd, num_cb, old_vocab, num_new_classes=NUM_BLOCK_CLASSES
        )
        torch.save(sd, dst / bin_files[0].name)
        migrated_name = bin_files[0].name

    cfg["audio_vocab_size"] = old_vocab + NUM_BLOCK_CLASSES
    cfg["block_migrated_from"] = str(src)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    for f in src.iterdir():
        if f.name in ("config.json", migrated_name):
            continue
        if f.is_file():
            shutil.copy2(f, dst / f.name)
        elif f.is_dir():
            shutil.copytree(f, dst / f.name, dirs_exist_ok=True)
    print(f"migrated checkpoint written to {dst}")


if __name__ == "__main__":
    main()
