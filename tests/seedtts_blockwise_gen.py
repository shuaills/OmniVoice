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
    '--generation-seed-index-map',
    default=None,
    help=(
        'optional JSON object mapping utterance id to its canonical source '
        'row index; when provided every generated row must be present'
    ),
)
ap.add_argument(
    '--eos-cfg-calibration',
    default='legacy',
    choices=['legacy', 'renorm', 'guided', 'mass_preserving'],
    help=(
        'EOS/CFG score calibration arm. legacy preserves historical mixed '
        'scores; the other choices are explicit experimental arms'
    ),
)
ap.add_argument(
    '--eos-cfg-trace',
    action='store_true',
    help=(
        'record a compact per-reveal-step EOS mass/margin/selection trace in '
        'generation metadata; disabled by default'
    ),
)
ap.add_argument(
    '--cfg-unconditional-seed-policy',
    default='shared',
    choices=['shared', 'drop_ref'],
    help=(
        'whether the CFG-unconditional branch shares reference-audio seed '
        'tokens (historical behavior) or drops them for a prompt-free CFG arm'
    ),
)
ap.add_argument(
    '--silence-stop-seconds', type=float, default=0.0,
    help='force-stop after this much continuous digital silence; 0 disables',
)
ap.add_argument(
    '--silence-match-codebooks', type=int, default=2,
    help='leading codebooks that must match the steady-state silence token',
)
ap.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32', 'bf16'])
ap.add_argument(
    '--prompt-contract',
    default='current',
    choices=['current', 'official-emilia'],
    help=(
        'current preserves the historical blockwise evaluator exactly; '
        'official-emilia mirrors the prompt processing in '
        'create_voice_clone_prompt('
        'preprocess_prompt=False): low-RMS normalization, tokenizer-hop '
        'truncation, and unmodified reference text. Pass --lang None as a '
        'separate switch for full Emilia inference-conditioning parity'
    ),
)
ap.add_argument(
    '--lang',
    default=None,
    help=(
        'language conditioning; omitted means path-inferred zh/en for the '
        'historical control. Pass literal None to disable language '
        'conditioning explicitly; this switch is orthogonal to prompt contract'
    ),
)
ap.add_argument(
    '--ref-text-punctuation',
    default='contract',
    choices=['contract', 'add', 'preserve'],
    help=(
        'reference-text boundary policy. contract resolves to add for current '
        'and preserve for official-emilia; official-emilia rejects add'
    ),
)
ap.add_argument(
    '--item-error-policy',
    default='contract',
    choices=['contract', 'record', 'fail-fast', 'fail-at-end'],
    help=(
        'record preserves the historical zero-exit behavior; contract resolves '
        'to record for current and fail-at-end for official-emilia'
    ),
)
args = ap.parse_args()

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402
from omnivoice.blockdiff_dual import _decode_block_causal  # noqa: E402
from omnivoice.eval.seedtts_blockwise_contract import (  # noqa: E402
    prepare_official_emilia_audio,
    prepare_reference_text,
    resolve_prompt_contract,
    restore_official_emilia_output_rms,
)
from omnivoice.utils.audio import load_audio  # noqa: E402
import torchaudio.functional as AF  # noqa: E402

prompt_contract = resolve_prompt_contract(
    args.prompt_contract,
    lang_arg=args.lang,
    tsv_path=args.tsv,
    ref_text_punctuation_arg=args.ref_text_punctuation,
)
lang = prompt_contract.language
cfg_unconditional_seed_policy = args.cfg_unconditional_seed_policy
item_error_policy = args.item_error_policy
if item_error_policy == 'contract':
    item_error_policy = (
        'record' if prompt_contract.name == 'current' else 'fail-at-end'
    )
if (
    args.prompt_contract != 'current'
    or args.lang is not None
    or args.ref_text_punctuation != 'contract'
    or args.item_error_policy != 'contract'
    or args.cfg_unconditional_seed_policy != 'shared'
):
    print(
        f'[prompt-contract] name={prompt_contract.name} lang={lang!r} '
        f'ref_text_punctuation={prompt_contract.ref_text_punctuation} '
        f'cfg_unconditional_seed_policy={cfg_unconditional_seed_policy} '
        f'item_error_policy={item_error_policy}',
        flush=True,
    )
sd = os.path.dirname(os.path.abspath(args.tsv))
rows = list(csv.reader(open(args.tsv), delimiter='\t'))
i, n = (int(x) for x in args.shard.split('/'))
rows = [r for k, r in enumerate(rows) if k % n == i]
if args.limit:
    rows = rows[:args.limit]

generation_seed_indices = None
if args.generation_seed_index_map is not None:
    with open(args.generation_seed_index_map, encoding='utf-8') as fh:
        generation_seed_indices = json.load(fh)
    if not isinstance(generation_seed_indices, dict):
        raise ValueError('--generation-seed-index-map must contain a JSON object')
    invalid_seed_indices = {
        key: value
        for key, value in generation_seed_indices.items()
        if not isinstance(key, str)
        or not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    }
    if invalid_seed_indices:
        raise ValueError(
            'invalid canonical seed indices: '
            f'{dict(list(invalid_seed_indices.items())[:5])}'
        )
    missing_seed_ids = [
        row[0] for row in rows if row[0] not in generation_seed_indices
    ]
    if missing_seed_ids:
        raise ValueError(
            'generation seed index map is missing shard utterances: '
            f'{missing_seed_ids[:5]}'
        )

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
print(
    f'[eos-cfg] calibration={args.eos_cfg_calibration} '
    f'trace={args.eos_cfg_trace}',
    flush=True,
)
gen = OmniVoiceGenerationConfig()
gen.guidance_scale = args.guidance_scale
gen.eos_cfg_calibration = args.eos_cfg_calibration
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
        prepared_ref = None
        if prompt_contract.name == 'current':
            # Keep this branch operation-for-operation equivalent to the
            # historical evaluator.  It is the control arm for prompt A/Bs.
            wav, sr = sf.read(src)
            w = torch.tensor(wav, dtype=torch.float32)
            if w.dim() > 1:
                w = w.mean(-1)
            if sr != tsr:
                w = AF.resample(w.unsqueeze(0), sr, tsr).squeeze(0)
        else:
            official_wav = load_audio(src, tsr)
            prepared_ref = prepare_official_emilia_audio(
                official_wav,
                hop_length=int(tok.config.hop_length),
            )
            w = torch.from_numpy(prepared_ref.waveform[0])
        with torch.no_grad():
            enc = tok.encode(w.unsqueeze(0).unsqueeze(0).to(tok.device))
        at = enc.audio_codes if hasattr(enc, 'audio_codes') else enc
        if isinstance(at, (list, tuple)):
            at = at[0]
        ref_toks = at.squeeze()
        generation_seed_index = (
            generation_seed_indices[utt_id]
            if generation_seed_indices is not None
            else k * n + i
        )
        generation_seed = 20260707 + generation_seed_index
        torch.manual_seed(generation_seed)
        resolved_ptext = prepare_reference_text(
            ptext,
            prompt_contract.ref_text_punctuation,
        )
        inp = model._prepare_inference_inputs(
            ttext, bs, resolved_ptext, ref_toks, lang, None, False
        )
        ii = inp['input_ids']
        am = inp['audio_mask']
        if ii.dim() == 3:
            ii = ii[0]
        if am.dim() == 2:
            am = am[0]
        a0 = ii.size(1) - int(am.sum())
        prefix = ii[:, :a0]
        min_gen_frames = max(
            8,
            int(0.3 * model._estimate_target_tokens(ttext, None, None)),
        )
        eos_cfg_trace = [] if args.eos_cfg_trace else None
        toks, stats = _decode_block_causal(
            model, prefix, gen, block_size=bs, max_blocks=args.max_blocks,
            num_step_per_block=args.steps_per_block, use_kv_cache=True,
            seed_audio=ref_toks,
            cfg_unconditional_seed_policy=cfg_unconditional_seed_policy,
            min_gen_frames=min_gen_frames,
            silence_run_frames=silence_run_frames,
            silence_match_codebooks=args.silence_match_codebooks,
            eos_cfg_trace=eos_cfg_trace)
        _dd = os.environ.get('DUMP_TOKENS_DIR')
        if _dd:
            import numpy as _np
            os.makedirs(_dd, exist_ok=True)
            _np.save(os.path.join(_dd, utt_id + '.npy'), toks.cpu().numpy())
        if toks.size(1) == 0:
            raise RuntimeError('empty generation')
        wav_np = (tok.decode(toks.to(tok.device).unsqueeze(0))
                  .audio_values[0].float().detach().cpu().numpy().reshape(-1))
        if prepared_ref is not None:
            wav_np = restore_official_emilia_output_rms(
                wav_np,
                original_ref_rms=prepared_ref.original_rms,
            )
        sf.write(outwav, wav_np, tsr)
        done += 1
        seed_frames = int(ref_toks.size(1))
        seed_frames_mod_block = seed_frames % bs
        first_target_block_frames = (
            bs - seed_frames_mod_block if seed_frames_mod_block else bs
        )
        meta = {'utt_id': utt_id, 'frames': toks.size(1),
                'eos': stats['stopped_by_eos'],
                'silence_stop': stats['stopped_by_silence'],
                'stop_reason': stats['stop_reason'],
                'silence_col': stats['silence_col'],
                'silence_trigger_col': stats['silence_trigger_col'],
                'silence_run_frames': silence_run_frames,
                'silence_match_codebooks': args.silence_match_codebooks,
                'min_gen_frames': min_gen_frames,
                'seed_frames': seed_frames,
                'seed_frames_mod_block': seed_frames_mod_block,
                'first_target_block_frames': first_target_block_frames,
                'eos_cfg_calibration': args.eos_cfg_calibration,
                'generation_seed_index': generation_seed_index,
                'generation_seed_value': generation_seed,
                'n_blocks': stats['n_blocks']}
        if eos_cfg_trace is not None:
            meta['eos_cfg_trace'] = eos_cfg_trace
        experimental_contract = (
            prepared_ref is not None
            or args.lang is not None
            or cfg_unconditional_seed_policy != 'shared'
            or args.guidance_scale != 2.0
            or args.eos_cfg_calibration != 'legacy'
        )
        if prepared_ref is not None:
            meta.update({
                'ref_rms': prepared_ref.original_rms,
                'ref_truncated_samples': prepared_ref.truncated_samples,
                'output_ref_rms_restored': prepared_ref.original_rms < 0.1,
            })
        if experimental_contract:
            meta.update({
                'prompt_contract': prompt_contract.name,
                'language': lang,
                'ref_text_punctuation': prompt_contract.ref_text_punctuation,
                'cfg_unconditional_seed_policy': cfg_unconditional_seed_policy,
                'guidance_scale': args.guidance_scale,
                'generation_seed': generation_seed,
            })
        with open(meta_path, 'a') as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + '\n')
    except Exception as e:
        failed += 1
        with open(fail_path, 'a') as fh:
            fh.write(json.dumps({'utt_id': utt_id, 'err': str(e)[:200]}) + '\n')
        if prompt_contract.name == 'official-emilia':
            print(
                f'[shard {args.shard}] FAILED utt_id={utt_id} '
                f'error={type(e).__name__}: {e}',
                file=sys.stderr,
                flush=True,
            )
        if item_error_policy == 'fail-fast':
            raise RuntimeError(
                f'item generation failed for utt_id={utt_id}'
            ) from e
    if (k + 1) % 20 == 0:
        el = time.time() - t_start
        print(f'[shard {args.shard}] {k+1}/{len(rows)} done={done} skip={skipped} '
              f'fail={failed} {el/max(done,1):.1f}s/item', flush=True)
print(f'[shard {args.shard}] FINISHED total={len(rows)} done={done} skip={skipped} fail={failed}', flush=True)
if failed and item_error_policy == 'fail-at-end':
    raise SystemExit(
        f'blockwise Seed-TTS generation completed with {failed} failed item(s); '
        f'see {fail_path}'
    )
