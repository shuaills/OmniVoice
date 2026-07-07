"""A/B test: does the leading 'emm' come from context-free vocoder decoding at
the seed/generation junction? A = decode(gen_tokens) (current pipeline);
B = decode(seed + gen tokens), then slice off the seed duration in waveform.
Same generated tokens both arms — isolates the decode boundary."""
import os, sys, json, torch, soundfile as sf
import torchaudio.functional as AF
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.blockdiff_dual import _decode_block_causal
import tempfile

CKPT = 'exp/block_b2/checkpoint-50000'
BASE = '/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block'
SD = '/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset/zh'
OUT = 'junction_ab'
UTTS = ["10002287-00000099","10002290-00000102","10002309-00000033","10002430-00000015",
        "10002481-00000105","10002750-00000099","10002751-00000071","10002753-00000006",
        "10002287-00000095","10002334-00000041","10003502-00000044","10004424-00000005"]

rows = {}
for g, ln in enumerate(open(os.path.join(SD, 'test.tsv'))):
    f = ln.rstrip('\n').split('\t')
    if len(f) >= 4:
        rows[f[0]] = (g, f[1], f[2], f[3])

mdir = tempfile.mkdtemp(prefix='b2ab_')
for f in ('model.safetensors', 'config.json'):
    os.symlink(os.path.abspath(os.path.join(CKPT, f)), os.path.join(mdir, f))
for f in ('audio_tokenizer', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
    src = os.path.abspath(os.path.join(BASE, f))
    if os.path.exists(src):
        os.symlink(src, os.path.join(mdir, f))
model = OmniVoice.from_pretrained(mdir, device_map='cuda:0', dtype=torch.float16,
                                  attn_implementation='sdpa')
model.eval()
tok = model.audio_tokenizer
tsr = int(model.sampling_rate)
gen = OmniVoiceGenerationConfig()
os.makedirs(OUT, exist_ok=True)

for utt in UTTS:
    g, ptext, pwav, ttext = rows[utt]
    w, sr = sf.read(os.path.join(SD, 'prompt_wavs', os.path.basename(pwav)))
    w = torch.tensor(w, dtype=torch.float32)
    if w.dim() > 1: w = w.mean(-1)
    if sr != tsr: w = AF.resample(w.unsqueeze(0), sr, tsr).squeeze(0)
    with torch.no_grad():
        enc = tok.encode(w.unsqueeze(0).unsqueeze(0).to(tok.device))
    at = enc.audio_codes if hasattr(enc, 'audio_codes') else enc
    if isinstance(at, (list, tuple)): at = at[0]
    ref = at.squeeze()
    torch.manual_seed(20260707 + g)
    inp = model._prepare_inference_inputs(ttext, 32, ptext, ref, 'zh', None, False)
    ii, am = inp['input_ids'], inp['audio_mask']
    if ii.dim() == 3: ii = ii[0]
    if am.dim() == 2: am = am[0]
    a0 = ii.size(1) - int(am.sum())
    toks, stats = _decode_block_causal(
        model, ii[:, :a0], gen, block_size=32, max_blocks=24,
        num_step_per_block=16, use_kv_cache=True, seed_audio=ref,
        min_gen_frames=max(8, int(0.3 * model._estimate_target_tokens(ttext, None, None))))
    with torch.no_grad():
        wavA = tok.decode(toks.to(tok.device).unsqueeze(0)).audio_values[0]
        both = torch.cat([ref.to(toks.device), toks], dim=1)
        wavB_full = tok.decode(both.to(tok.device).unsqueeze(0)).audio_values[0]
    wavA = wavA.float().cpu().numpy().reshape(-1)
    wavB_full = wavB_full.float().cpu().numpy().reshape(-1)
    seed_samples = int(ref.shape[1] / 25.0 * tsr)
    wavB = wavB_full[seed_samples:]
    sf.write(f'{OUT}/{utt}_A_gen_only.wav', wavA, tsr)
    sf.write(f'{OUT}/{utt}_B_ctx_sliced.wav', wavB, tsr)
    print(f'{utt} frames={toks.size(1)} lenA={len(wavA)/tsr:.2f}s lenB={len(wavB)/tsr:.2f}s', flush=True)
print('AB_GEN_DONE')
