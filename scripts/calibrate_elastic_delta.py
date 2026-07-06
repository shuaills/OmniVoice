#!/usr/bin/env python3
"""Print the |dT| distribution of elastic corruption (legacy vs targeted).

Run:  python scripts/calibrate_elastic_delta.py
"""

import random

import torch

from omnivoice.elastic import corrupt_audio_region, corrupt_audio_region_targeted

C, MASK, N = 8, 1024, 1000


def sample_ratios(fn, **kw):
    rng = random.Random(0)
    ratios = []
    for _ in range(N):
        T = rng.randint(40, 500)
        prompt = rng.randint(0, T // 3)
        tokens = torch.randint(0, MASK, (C, T))
        inputs, labels = tokens.clone(), tokens.clone()
        m = torch.rand(C, T - prompt) < rng.random()
        inputs[:, prompt:][m] = MASK
        labels[:, prompt:][~m] = -100
        labels[:, :prompt] = -100
        ci, _, _ = fn(inputs, labels, prompt, MASK, rng=rng, **kw)
        ratios.append((ci.shape[1] - prompt) / (T - prompt))
    ratios.sort()
    return ratios


def report(name, r):
    q = lambda p: r[int(p * len(r))]
    print(f"{name:>10}: P1={q(0.01):.3f} P10={q(0.10):.3f} P25={q(0.25):.3f} "
          f"P50={q(0.50):.3f} P75={q(0.75):.3f} P90={q(0.90):.3f} P99={q(0.99):.3f}  "
          f"min={r[0]:.3f} max={r[-1]:.3f}")


if __name__ == "__main__":
    report("legacy", sample_ratios(
        corrupt_audio_region, merge_prob=0.08, insert_prob=0.04,
        end_append_max_ratio=0.25,
    ))
    report("targeted", sample_ratios(
        corrupt_audio_region_targeted, delta_max=0.3, mid_insert_frac=0.3,
    ))
    print("target   : U(0.7, 1.3) — targeted P1/P99 should hug 0.70/1.30; "
          "legacy should show its short-deficit ceiling (~0.92) for contrast")
