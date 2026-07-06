import os, sys, json, tempfile, time, argparse
import torch, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockwise_probe import TEXTS, cer

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--base', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--tag', required=True)
args = ap.parse_args()

from omnivoice.models.omnivoice import OmniVoice
os.makedirs(args.out, exist_ok=True)
mdir = tempfile.mkdtemp(prefix='ovb_model_')
for f in ('model.safetensors', 'config.json'):
    os.symlink(os.path.abspath(os.path.join(args.ckpt, f)), os.path.join(mdir, f))
for f in ('audio_tokenizer', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
    src = os.path.abspath(os.path.join(args.base, f))
    if os.path.exists(src) and not os.path.exists(os.path.join(mdir, f)):
        os.symlink(src, os.path.join(mdir, f))
model = OmniVoice.from_pretrained(mdir, device_map='cuda:0', dtype=torch.float16, attn_implementation='sdpa')
model.eval()
fr = model.audio_tokenizer.config.frame_rate
rows = []
for ti, (lang, text) in enumerate(TEXTS):
    torch.manual_seed(20260707)
    t0 = time.time()
    audios = model.generate(text=text, language=lang, num_step=32)
    wall = time.time() - t0
    a0 = audios[0]
    wav_np = (a0.float().detach().cpu().numpy().reshape(-1)
              if torch.is_tensor(a0) else a0.reshape(-1).astype('float32'))
    dur = len(wav_np) / int(model.sampling_rate)
    wav = os.path.join(args.out, f't{ti}_{args.tag}.wav')
    sf.write(wav, wav_np, int(model.sampling_rate))
    rows.append({'text_id': ti, 'lang': lang, 'sec': round(dur, 2),
                 'rule_frames': int(model._estimate_target_tokens(text, None, None)),
                 'wall_s': round(wall, 2), 'rtf': round(wall / dur, 3), 'wav': wav})
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

from funasr import AutoModel
asr = AutoModel(model='paraformer-zh', disable_update=True)
cer_rows = []
for r in rows:
    if r['lang'] == 'zh':
        hyp = asr.generate(input=r['wav'])[0]['text']
        cer_rows.append({'text_id': r['text_id'], 'cer': round(cer(TEXTS[r['text_id']][1], hyp), 4), 'hyp': hyp})
json.dump({'tag': args.tag, 'ckpt': args.ckpt, 'rows': rows, 'cer': cer_rows},
          open(os.path.join(args.out, f'report_{args.tag}.json'), 'w'), ensure_ascii=False)
print('CER:', json.dumps(cer_rows, ensure_ascii=False), flush=True)
