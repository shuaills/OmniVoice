"""Shardable Seed-TTS testset generator for B2 blockwise decode.

Output layout matches yinfeng's eval.sh expectation: OUT/<utt_id>.wav
Resume-safe: existing wavs are skipped.
"""
import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time

import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('--tsv', required=True)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--base', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--shard', default='0/1', help='i/n')
ap.add_argument('--steps-per-block', type=int, default=16)
ap.add_argument('--max-blocks', type=int, default=24)
ap.add_argument('--block-size', type=int, default=32)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--guidance-scale', type=float, default=2.0)
ap.add_argument(
    '--silence-stop-seconds', type=float, default=0.0,
    help='force-stop after this much continuous digital silence; 0 disables',
)
ap.add_argument(
    '--silence-match-codebooks', type=int, default=2,
    help='leading codebooks that must match the steady-state silence token',
)
ap.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32', 'bf16'])
ap.add_argument('--lang', default=None)
args = ap.parse_args()

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402
from omnivoice.blockdiff_dual import _decode_block_causal  # noqa: E402
from omnivoice.utils.text import add_punctuation  # noqa: E402
import torchaudio.functional as AF  # noqa: E402

lang = args.lang or ('zh' if '/zh/' in args.tsv else 'en')
sd = os.path.dirname(os.path.abspath(args.tsv))
rows = list(csv.reader(open(args.tsv), delimiter='\t'))
i, n = (int(x) for x in args.shard.split('/'))
rows = [r for k, r in enumerate(rows) if k % n == i]
if args.limit:
    rows = rows[:args.limit]

os.makedirs(args.out, exist_ok=True)
mdir = tempfile.mkdtemp(prefix='b2eval_model_')
for f in ('model.safetensors', 'config.json'):
    os.symlink(os.path.abspath(os.path.join(args.ckpt, f)), os.path.join(mdir, f))
for f in ('audio_tokenizer', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
    src = os.path.abspath(os.path.join(args.base, f))
    if os.path.exists(src) and not os.path.exists(os.path.join(mdir, f)):
        os.symlink(src, os.path.join(mdir, f))
model = OmniVoice.from_pretrained(mdir, device_map='cuda:0', dtype={'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}[args.dtype],
                                  attn_implementation='sdpa')
model.eval()
tok = model.audio_tokenizer
tsr = int(model.sampling_rate)
frame_rate = float(tok.config.frame_rate)
if args.silence_stop_seconds < 0:
    raise ValueError('--silence-stop-seconds must be >= 0')
silence_run_frames = (
    int(math.ceil(args.silence_stop_seconds * frame_rate))
    if args.silence_stop_seconds > 0
    else 0
)
print(
    f'[force-stop] seconds={args.silence_stop_seconds:g} '
    f'frame_rate={frame_rate:g} frames={silence_run_frames} '
    f'match_codebooks={args.silence_match_codebooks}',
    flush=True,
)
gen = OmniVoiceGenerationConfig()
gen.guidance_scale = args.guidance_scale
bs = args.block_size

done = skipped = failed = 0
t_start = time.time()
meta_path = os.path.join(args.out, f'gen_meta_shard{i}.jsonl')
fail_path = os.path.join(args.out, f'failures_shard{i}.jsonl')
for k, row in enumerate(rows):
    utt_id, ptext, pwav, ttext = row[0], row[1], row[2], row[3]
    outwav = os.path.join(args.out, f'{utt_id}.wav')
    if os.path.exists(outwav):
        skipped += 1
        continue
    try:
        src = os.path.join(sd, 'prompt_wavs', os.path.basename(pwav))
        wav, sr = sf.read(src)
        w = torch.tensor(wav, dtype=torch.float32)
        if w.dim() > 1:
            w = w.mean(-1)
        if sr != tsr:
            w = AF.resample(w.unsqueeze(0), sr, tsr).squeeze(0)
        with torch.no_grad():
            enc = tok.encode(w.unsqueeze(0).unsqueeze(0).to(tok.device))
        at = enc.audio_codes if hasattr(enc, 'audio_codes') else enc
        if isinstance(at, (list, tuple)):
            at = at[0]
        ref_toks = at.squeeze()
        torch.manual_seed(20260707 + k * n + i)
        # Official released-pipeline parity: prompt text gets sentence-final
        # punctuation (preprocess_prompt does this in model.generate); without it
        # the ptext+ttext concat blurs the sentence boundary at the junction.
        inp = model._prepare_inference_inputs(ttext, bs, add_punctuation(ptext), ref_toks, lang, None, False)
        ii = inp['input_ids']
        am = inp['audio_mask']
        if ii.dim() == 3:
            ii = ii[0]
        if am.dim() == 2:
            am = am[0]
        a0 = ii.size(1) - int(am.sum())
        prefix = ii[:, :a0]
        toks, stats = _decode_block_causal(
            model, prefix, gen, block_size=bs, max_blocks=args.max_blocks,
            num_step_per_block=args.steps_per_block, use_kv_cache=True,
            seed_audio=ref_toks,
            min_gen_frames=max(8, int(0.3 * model._estimate_target_tokens(ttext, None, None))),
            silence_run_frames=silence_run_frames,
            silence_match_codebooks=args.silence_match_codebooks)
        _dd = os.environ.get('DUMP_TOKENS_DIR')
        if _dd:
            import numpy as _np
            os.makedirs(_dd, exist_ok=True)
            _np.save(os.path.join(_dd, utt_id + '.npy'), toks.cpu().numpy())
        if toks.size(1) == 0:
            raise RuntimeError('empty generation')
        wav_np = (tok.decode(toks.to(tok.device).unsqueeze(0))
                  .audio_values[0].float().detach().cpu().numpy().reshape(-1))
        sf.write(outwav, wav_np, tsr)
        done += 1
        with open(meta_path, 'a') as fh:
            fh.write(json.dumps({'utt_id': utt_id, 'frames': toks.size(1),
                                 'eos': stats['stopped_by_eos'],
                                 'silence_stop': stats['stopped_by_silence'],
                                 'stop_reason': stats['stop_reason'],
                                 'silence_col': stats['silence_col'],
                                 'silence_trigger_col': stats['silence_trigger_col'],
                                 'silence_run_frames': silence_run_frames,
                                 'n_blocks': stats['n_blocks']}, ensure_ascii=False) + '\n')
    except Exception as e:
        failed += 1
        with open(fail_path, 'a') as fh:
            fh.write(json.dumps({'utt_id': utt_id, 'err': str(e)[:200]}) + '\n')
    if (k + 1) % 20 == 0:
        el = time.time() - t_start
        print(f'[shard {args.shard}] {k+1}/{len(rows)} done={done} skip={skipped} '
              f'fail={failed} {el/max(done,1):.1f}s/item', flush=True)
print(f'[shard {args.shard}] FINISHED total={len(rows)} done={done} skip={skipped} fail={failed}', flush=True)
