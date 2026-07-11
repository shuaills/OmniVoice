#!/usr/bin/env python3
"""Per-pack beta statistics for legacy-moment-matched lambda init (contract v2 SS4).

Runs the REAL processor -> filter -> packer -> collator pipeline of a given
train config and records, per consumed pack:
  N_audio[c], N_eos, N_void[c], beta_a, beta_e, beta_v
plus overlong-drop counts and a config/RNG fingerprint. Consumes loss_kind
(contract SS2) -- never classifies by token value.
"""
import argparse
import hashlib
import json
import random
import subprocess

import numpy as np
import torch

from omnivoice.blockdiff_dual import KIND_ACOUSTIC, KIND_EOS, KIND_VOID
from omnivoice.training.config import TrainingConfig
from omnivoice.training.builder import build_dataloaders, _resolve_model_path
from transformers import AutoTokenizer

SPECIALS = ["<|denoise|>", "<|lang_start|>", "<|lang_end|>", "<|instruct_start|>",
            "<|instruct_end|>", "<|text_start|>", "<|text_end|>"]


def build_tokenizer(config):
    tok = AutoTokenizer.from_pretrained(_resolve_model_path(config.llm_name_or_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    add = [t for t in SPECIALS if t not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", required=True)
    ap.add_argument("--num_packs", type=int, default=2000)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    config = TrainingConfig.from_json(args.train_config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    tok = build_tokenizer(config)
    train_loader, _ = build_dataloaders(config, tok)
    # rebuild single-process (builder hardcodes prefetch_factor=4, illegal at
    # num_workers=0): counters stay in-process, iteration order deterministic
    from torch.utils.data import DataLoader
    dataset = train_loader.dataset
    train_loader = DataLoader(
        dataset, batch_size=None, num_workers=0,
        collate_fn=train_loader.collate_fn,
    )

    w = torch.tensor(config.audio_codebook_weights, dtype=torch.float64)
    wbar = (w / w.sum()).numpy()
    C = config.num_audio_codebook

    fingerprint = {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "train_config": args.train_config,
        "config_md5": hashlib.md5(open(args.train_config, "rb").read()).hexdigest(),
        "seed": config.seed,
        "batch_tokens": config.batch_tokens,
        "torch": torch.__version__,
    }

    packs = []
    betas_a, betas_e, betas_v = [], [], []
    fout = open(f"{args.out_prefix}.packs.jsonl", "w")
    it = iter(train_loader)
    for k in range(args.num_packs):
        batch = next(it)
        lk = batch["loss_kind"][0]          # [C, L] uint8
        doc = batch["document_ids"][0]      # [L]
        n_audio = [(lk[c] == KIND_ACOUSTIC).sum().item() for c in range(C)]
        n_eos = int((lk[0] == KIND_EOS).sum().item())
        n_void = [(lk[c] == KIND_VOID).sum().item() for c in range(C)]
        # per-document EOS event check (docs with any supervision carry exactly one)
        n_docs = 0
        for d in doc.unique():
            if d.item() < 0:
                continue
            m = doc == d
            if (lk[:, m] != 0).any():
                n_docs += 1
                assert (lk[0, m] == KIND_EOS).sum().item() == 1, \
                    f"pack {k} doc {d.item()}: cb0 EOS != 1"
        beta_a = beta_e = beta_v = 0.0
        for c in range(C):
            D_c = n_audio[c] + (n_eos if c == 0 else 0) + n_void[c]
            if D_c == 0:
                continue
            beta_a += wbar[c] * n_audio[c] / D_c
            beta_v += wbar[c] * n_void[c] / D_c
            if c == 0:
                beta_e = wbar[0] * n_eos / D_c
        rec = {"pack": k, "n_docs": n_docs, "N_audio": n_audio, "N_eos": n_eos,
               "N_void": n_void, "beta_a": beta_a, "beta_e": beta_e, "beta_v": beta_v}
        fout.write(json.dumps(rec) + "\n")
        betas_a.append(beta_a); betas_e.append(beta_e); betas_v.append(beta_v)
        if (k + 1) % 100 == 0:
            print(f"BETA_HB pack={k+1}/{args.num_packs}", flush=True)
    fout.close()

    ds = getattr(train_loader, "dataset", None)
    dropped = getattr(ds, "dropped_overlong", 0) if ds is not None else "n/a"
    Ea, Ee, Ev = (float(np.mean(x)) for x in (betas_a, betas_e, betas_v))
    summary = {
        "fingerprint": fingerprint,
        "num_packs": len(betas_a),
        "E_beta_a": Ea, "E_beta_e": Ee, "E_beta_v": Ev,
        "std_beta_a": float(np.std(betas_a)), "std_beta_e": float(np.std(betas_e)),
        "std_beta_v": float(np.std(betas_v)),
        "gamma": Ea, "lambda_eos": Ee / Ea, "lambda_void": Ev / Ea,
        "dropped_overlong": dropped,
    }
    json.dump(summary, open(f"{args.out_prefix}.summary.json", "w"), indent=2)
    print("BETA_SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
