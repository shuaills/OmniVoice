#!/usr/bin/env python3
"""Post-hoc enrichment of beta_stats pack records (Slice 1.1, Codex SS5).

Reads <prefix>.packs.jsonl (raw per-pack counts) and emits the enriched
summary: mean/std/p5/p50/p95, bootstrap 95% CI, N_docs/N_eos per pack,
max |beta_a+beta_e+beta_v - sum_{c:D_c>0} wbar_c|, and D_c==0 incidence.
Works on in-flight or finished runs without re-consuming data.
"""
import argparse
import json

import numpy as np

WBAR = np.array([8, 8, 6, 6, 4, 4, 2, 2], dtype=np.float64)
WBAR = WBAR / WBAR.sum()


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs_jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.packs_jsonl)]
    ba = np.array([r["beta_a"] for r in recs])
    be = np.array([r["beta_e"] for r in recs])
    bv = np.array([r["beta_v"] for r in recs])
    ndocs = np.array([r["n_docs"] for r in recs])
    neos = np.array([r["N_eos"] for r in recs])

    dzero = np.zeros(8, dtype=int)
    sum_err = []
    for r in recs:
        active = 0.0
        for c in range(8):
            D_c = r["N_audio"][c] + (r["N_eos"] if c == 0 else 0) + r["N_void"][c]
            if D_c == 0:
                dzero[c] += 1
            else:
                active += WBAR[c]
        sum_err.append(abs(r["beta_a"] + r["beta_e"] + r["beta_v"] - active))
    sum_err = np.array(sum_err)

    def stats(x):
        return {
            "mean": float(x.mean()), "std": float(x.std()),
            "p5": float(np.percentile(x, 5)), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "boot_ci95": boot_ci(x),
        }

    Ea, Ee, Ev = ba.mean(), be.mean(), bv.mean()
    out = {
        "num_packs": len(recs),
        "beta_a": stats(ba), "beta_e": stats(be), "beta_v": stats(bv),
        "gamma": float(Ea),
        "lambda_eos": float(Ee / Ea), "lambda_void": float(Ev / Ea),
        "lambda_eos_boot_ci95": [float(l / Ea) for l in boot_ci(be)],
        "lambda_void_boot_ci95": [float(l / Ea) for l in boot_ci(bv)],
        "n_docs_per_pack": {"mean": float(ndocs.mean()), "p5": float(np.percentile(ndocs, 5)), "p95": float(np.percentile(ndocs, 95))},
        "n_eos_per_pack": {"mean": float(neos.mean()), "p5": float(np.percentile(neos, 5)), "p95": float(np.percentile(neos, 95))},
        "beta_sum_max_abs_err": float(sum_err.max()),
        "beta_sum_mean_abs_err": float(sum_err.mean()),
        "D_c_zero_counts": dzero.tolist(),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print("BETA_ENRICHED", json.dumps(out)[:600])


if __name__ == "__main__":
    main()
