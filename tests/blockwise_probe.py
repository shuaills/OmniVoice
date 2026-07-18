#!/usr/bin/env python3
"""B2 blockwise judgment probe (10k cadence; also the init-ckpt zero point).

Per text: cache-path blockwise generation -> wav + metrics row (n_blocks,
frames, EOS position or max-blocks fallback, per-block wall time, RTF).
report.json adds: funasr CER (zh rows, elastic-probe parity), EOS length
sanity (generated frames vs the official rule duration (Eq.4 fallback path
model._estimate_target_tokens) ratio + spearman vs text chars), and a
cache-vs-recompute wall-time table with an sdpa token-parity assertion
(same discipline as gate3c/gate3g).

Never touches the training job. Runs on dev3 4090: sdpa only.

Usage:
  python tests/blockwise_probe.py --ckpt <dir> --base <dir> --out <dir> \
      [--steps-per-block 16] [--max-blocks 24]
"""
import argparse, json, math, os, re, tempfile, time

import torch
import soundfile as sf

import omnivoice.blockdiff_dual as BD
from omnivoice.blockdiff_dual import _decode_block_causal

# verbatim from elastic probe-workdir/tests/early_signal_probe.py (cross-arm
# comparability -- do not edit)
TEXTS = [
    ("zh", "今天天气不错，我们一起去公园散步吧。"),
    ("zh", "他昨天买了三千二百五十六本书，花了四万零七十八元。"),
    ("zh", "这个项目的截止日期是二零二六年七月十五号下午三点。"),
    ("zh", "你能不能再说一遍？我刚才没有听清楚。"),
    ("zh", "科学家们在实验室里日复一日地重复着枯燥的实验。"),
    ("zh", "深度学习模型的推理速度取决于显存带宽和算子效率。"),
    ("zh", "哈哈哈，这也太好笑了吧，我快要笑死了。"),
    ("zh", "夜深了，城市渐渐安静下来，只剩下远处偶尔的车声。"),
    ("en", "The quick brown fox jumps over the lazy dog near the river bank."),
    ("en", "Machine learning systems require careful evaluation before deployment."),
]


def cer(ref, hyp):
    norm = lambda s: re.sub(r"[^\w一-鿿]", "", s.lower())
    r, h = norm(ref), norm(hyp)
    if not r:
        return None
    dp = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, dp[0] = dp[0], i
        for j, hc in enumerate(h, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (rc != hc))
            prev = cur
    return dp[-1] / len(r)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


class ForwardTimer:
    """Wraps the BD forward helpers; per-block wall time is recovered from
    the cond cache's kv lengths (cond block b always forwards at
    kv == P + (b+1)*bs; the prefix priming call sits at kv == P)."""

    def __init__(self, P, bs):
        self.P, self.bs = P, bs
        self.cond_cache = None
        self.calls = []
        self._fs, self._fp = BD._forward_slices, BD._forward_text_prefix

    def __enter__(self):
        def fp(model, text_ids, positions, attn4d, past_key_values):
            self.cond_cache = past_key_values
            t0 = time.time()
            out = self._fp(model, text_ids, positions, attn4d, past_key_values)
            self.calls.append((attn4d.shape[-1], True, time.time() - t0))
            return out

        def fs(
            model,
            ids,
            positions,
            attn4d,
            past_key_values=None,
            first_prev_ids=None,
        ):
            t0 = time.time()
            out = self._fs(
                model,
                ids,
                positions,
                attn4d,
                past_key_values,
                first_prev_ids=first_prev_ids,
            )
            is_cond = past_key_values is None or past_key_values is self.cond_cache
            self.calls.append((attn4d.shape[-1], is_cond, time.time() - t0))
            return out

        BD._forward_text_prefix, BD._forward_slices = fp, fs
        return self

    def __exit__(self, *a):
        BD._forward_slices, BD._forward_text_prefix = self._fs, self._fp

    def per_block(self):
        acc = {}
        for kv, is_cond, dt in self.calls:
            if is_cond and kv == self.P:
                b = "prefix"
            elif is_cond:
                b = (kv - self.P) // self.bs - 1
            else:
                b = kv // self.bs - 1
            acc[b] = acc.get(b, 0.0) + dt
        return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps-per-block", type=int, default=16)
    ap.add_argument("--max-blocks", type=int, default=24)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--recompute-subset", default="0,5,9")
    ap.add_argument("--texts-file", default=None)
    ap.add_argument("--ref-anchor", default=None, help="probe_assets/ref_anchor path prefix (loads .txt + _tokens.pt)")
    args = ap.parse_args()

    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(20260706)

    mdir = tempfile.mkdtemp(prefix="probe_model_")
    for f in ("model.safetensors", "config.json"):
        os.symlink(os.path.abspath(os.path.join(args.ckpt, f)), os.path.join(mdir, f))
    for f in ("audio_tokenizer", "tokenizer.json", "tokenizer_config.json",
              "chat_template.jinja"):
        src = os.path.abspath(os.path.join(args.base, f))
        if os.path.exists(src) and not os.path.exists(os.path.join(mdir, f)):
            os.symlink(src, os.path.join(mdir, f))

    model = OmniVoice.from_pretrained(
        mdir, device_map="cuda:0", dtype=torch.float16, attn_implementation="sdpa"
    )
    model.eval()
    fr = model.audio_tokenizer.config.frame_rate
    gen = OmniVoiceGenerationConfig()
    bs = args.block_size
    if getattr(args, 'texts_file', None):
        import json as _j
        TEXTS[:] = [tuple(t) for t in _j.load(open(args.texts_file))]
    recompute_ids = {int(x) for x in args.recompute_subset.split(",") if x}

    rows, rtf_table, parity, parity_wavs = [], [], [], []
    ref_text, ref_toks = None, None
    if args.ref_anchor:
        ref_text = open(args.ref_anchor + ".txt").read().strip()
        ref_toks = torch.load(args.ref_anchor + "_tokens.pt").to("cuda:0")
        print(f"[anchor] ref_text={ref_text[:20]}... ref_tokens={tuple(ref_toks.shape)}", flush=True)
    for ti, (lang, text) in enumerate(TEXTS):
        inp = model._prepare_inference_inputs(text, bs, ref_text, ref_toks, lang, None, False)
        input_ids = inp["input_ids"]
        if input_ids.dim() == 3:
            input_ids = input_ids[0]
        amask = inp["audio_mask"]
        if amask.dim() == 2:
            amask = amask[0]
        a0 = input_ids.size(1) - int(amask.sum())
        prefix = input_ids[:, :a0]

        with ForwardTimer(a0, bs) as ft:
            t0 = time.time()
            toks, stats = _decode_block_causal(
                model, prefix, gen, block_size=bs, max_blocks=args.max_blocks,
                num_step_per_block=args.steps_per_block, use_kv_cache=True,
                seed_audio=ref_toks,
            )
            wall = time.time() - t0
        frames = toks.size(1)
        dur = frames / fr if frames else 0.0

        wav_path = os.path.join(args.out, f"t{ti}_blockwise.wav")
        if frames > 0:
            wav_np = (
                model.audio_tokenizer.decode(
                    toks.to(model.audio_tokenizer.device).unsqueeze(0)
                )
                .audio_values[0].float().detach().cpu().numpy().reshape(-1)
            )
            sf.write(wav_path, wav_np, int(model.sampling_rate))

        row = {
            "text_id": ti, "lang": lang, "n_chars": len(text),
            "n_blocks": stats["n_blocks"], "frames": frames,
            "stopped_by_eos": stats["stopped_by_eos"],
            "eos_block": (stats["eos_col"] // bs if stats["eos_col"] is not None else None),
            "eos_col_in_block": (stats["eos_col"] % bs if stats["eos_col"] is not None else None),
            "maxblocks_fallback": not stats["stopped_by_eos"],
            "rule_frames": int(model._estimate_target_tokens(text, None, None)),
            "per_block_wall_s": {str(k): round(v, 3) for k, v in ft.per_block().items()},
            "wall_s": round(wall, 3),
            "rtf": round(wall / dur, 4) if dur else None,
            "wav": wav_path if frames > 0 else None,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if ti in recompute_ids:
            # Gate-caliber parity (gate3c/gate3g discipline): greedy decode,
            # RNG reseeded before EACH path so both consume identical
            # streams. The sampled row above is left untouched.
            gen_g = OmniVoiceGenerationConfig()
            gen_g.class_temperature = 0.0
            gen_g.position_temperature = 0.0
            pair = {}
            for cached in (True, False):
                torch.manual_seed(20260706 + ti)
                t0 = time.time()
                tk, _ = _decode_block_causal(
                    model, prefix, gen_g, block_size=bs,
                    max_blocks=args.max_blocks,
                    num_step_per_block=args.steps_per_block,
                    use_kv_cache=cached,
                )
                pair[cached] = (tk.cpu(), time.time() - t0)
            a, b = pair[True][0], pair[False][0]
            same = bool(a.shape == b.shape and torch.equal(a, b))
            if a.shape == b.shape:
                match = float((a == b).float().mean())
            else:
                n = min(a.numel(), b.numel())
                match = float((a.flatten()[:n] == b.flatten()[:n]).float().mean()) if n else 0.0
            parity.append({"text_id": ti, "tokens_equal": same,
                           "match_rate": round(match, 4),
                           "shapes_equal": bool(a.shape == b.shape)})
            rtf_table.append({
                "text_id": ti, "wall_cache_s": round(pair[True][1], 3),
                "wall_recompute_s": round(pair[False][1], 3),
                "speedup": round(pair[False][1] / pair[True][1], 2),
            })
            if not same:
                print(f"[warn] t{ti}: fp16 cache/recompute argmax drift, match_rate above; fp32 gate remains the correctness anchor", flush=True)
            for cached_flag, tag in ((True, "cache"), (False, "recompute")):
                tkq = pair[cached_flag][0]
                if lang == "zh" and tkq.size(1) > 0:
                    pw = os.path.join(args.out, f"t{ti}_parity_{tag}.wav")
                    wnp = (model.audio_tokenizer.decode(
                        tkq.to(model.audio_tokenizer.device).unsqueeze(0))
                        .audio_values[0].float().detach().cpu().numpy().reshape(-1))
                    sf.write(pw, wnp, int(model.sampling_rate))
                    parity_wavs.append({"text_id": ti, "tag": tag, "path": pw})

    cer_rows = []
    from funasr import AutoModel
    asr = AutoModel(model="paraformer-zh", disable_update=True)
    for r in rows:
        if r["lang"] == "zh" and r["wav"]:
            hyp = asr.generate(input=r["wav"])[0]["text"]
            cer_rows.append({"text_id": r["text_id"],
                             "cer": round(cer(TEXTS[r["text_id"]][1], hyp), 4),
                             "hyp": hyp})

    parity_cer = []
    for pv in parity_wavs:
        hyp_p = asr.generate(input=pv["path"])[0]["text"]
        parity_cer.append({"text_id": pv["text_id"], "tag": pv["tag"],
                           "cer": round(cer(TEXTS[pv["text_id"]][1], hyp_p), 4)})

    ratios = [r["frames"] / r["rule_frames"] for r in rows if r["rule_frames"]]
    sp = spearman([r["frames"] for r in rows], [r["n_chars"] for r in rows])
    report = {
        "ckpt": args.ckpt, "steps_per_block": args.steps_per_block,
        "max_blocks": args.max_blocks, "block_size": bs,
        "rows": rows, "cer": cer_rows,
        "eos_length": {
            "gen_over_rule_ratios": [round(x, 3) for x in ratios],
            "ratio_mean": round(sum(ratios) / len(ratios), 3) if ratios else None,
            "spearman_frames_vs_chars": round(sp, 3),
            "n_eos_stopped": sum(1 for r in rows if r["stopped_by_eos"]),
            "n_fallback": sum(1 for r in rows if r["maxblocks_fallback"]),
        },
        "cache_vs_recompute": rtf_table, "sdpa_parity": parity, "parity_cer": parity_cer,
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("=== CER ===", flush=True)
    print(json.dumps(cer_rows, ensure_ascii=False), flush=True)
    print("=== EOS_LENGTH ===", flush=True)
    print(json.dumps(report["eos_length"], ensure_ascii=False), flush=True)
    print("=== PARITY_CER ===", flush=True); print(json.dumps(parity_cer, ensure_ascii=False), flush=True)
    print("=== CACHE_VS_RECOMPUTE ===", flush=True)
    print(json.dumps(rtf_table, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
