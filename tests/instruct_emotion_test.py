import os, sys, json, tempfile
import torch, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockwise_probe import _decode_block_causal

CKPT = 'exp/block_b2/checkpoint-25000'
BASE = '/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block'
OUT = '../probe_b2_instruct_emotion'
TEXT = '我真的没想到，事情会变成这样。'
INSTRUCTS = [(None,'none'), ('用非常悲伤、低落的语气说','sad'),
             ('用非常开心、兴奋的语气说','happy'), ('用愤怒的语气说','angry')]
SEEDS = (20260707, 20260708, 20260709)

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
os.makedirs(OUT, exist_ok=True)
mdir = tempfile.mkdtemp(prefix='probe_model_')
for f in ('model.safetensors', 'config.json'):
    os.symlink(os.path.abspath(os.path.join(CKPT, f)), os.path.join(mdir, f))
for f in ('audio_tokenizer', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
    src = os.path.abspath(os.path.join(BASE, f))
    if os.path.exists(src) and not os.path.exists(os.path.join(mdir, f)):
        os.symlink(src, os.path.join(mdir, f))
model = OmniVoice.from_pretrained(mdir, device_map='cuda:0', dtype=torch.float16, attn_implementation='sdpa')
model.eval()
fr = model.audio_tokenizer.config.frame_rate
gen = OmniVoiceGenerationConfig()
rows = []
for ins, tag in INSTRUCTS:
    for seed in SEEDS:
        torch.manual_seed(seed)
        inp = model._prepare_inference_inputs(TEXT, 32, None, None, 'zh', ins, False)
        input_ids = inp['input_ids']
        if input_ids.dim() == 3:
            input_ids = input_ids[0]
        amask = inp['audio_mask']
        if amask.dim() == 2:
            amask = amask[0]
        a0 = input_ids.size(1) - int(amask.sum())
        prefix = input_ids[:, :a0]
        toks, stats = _decode_block_causal(model, prefix, gen, block_size=32,
            max_blocks=12, num_step_per_block=16, use_kv_cache=True)
        frames = toks.size(1)
        wav = os.path.join(OUT, f'instruct_{tag}_s{seed % 10}.wav')
        if frames:
            wnp = model.audio_tokenizer.decode(toks.to(model.audio_tokenizer.device)
                .unsqueeze(0)).audio_values[0].float().detach().cpu().numpy().reshape(-1)
            sf.write(wav, wnp, int(model.sampling_rate))
        rows.append({'instruct': tag, 'seed': seed % 10, 'frames': frames,
                     'sec': round(frames / fr, 2), 'eos': stats['stopped_by_eos']})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
json.dump(rows, open(os.path.join(OUT, 'result.json'), 'w'))
