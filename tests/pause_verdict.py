"""Two verdict tests for the onset-silence root cause.
T1: official ckpt, official full-canvas mode, postprocess OFF vs ON — leading
    low-energy duration pre-trim (does the official model also pause?).
T2: mid-sentence continuation on OUR blockwise model — prompt = first half of a
    sentence (audio+text), target = second half. Training-style boundary.
    If leading cb0=0 run ~0 here, the pause is linguistic, not pathological."""
import os, sys, torch, numpy as np, soundfile as sf
import torchaudio.functional as AF
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.blockdiff_dual import _decode_block_causal
import tempfile, copy
SD='/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset/zh'
OFFICIAL='/opt/gpfs/users/yinfeng/work/OmniVoice/pretrained_models/OmniVoice'
CKPT='exp/block_b2/checkpoint-50000'
BASE='/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block'
UTTS=["10002430-00000015","10002481-00000105","10002753-00000006","10003502-00000044"]
rows={}
for g,ln in enumerate(open(os.path.join(SD,'test.tsv'))):
    f=ln.rstrip('\n').split('\t')
    if len(f)>=4: rows[f[0]]=(g,f[1],f[2],f[3])

def lead_sil_sec(wav, sr, thr_ratio=0.05):
    w = np.abs(np.asarray(wav, dtype=np.float32))
    if w.size == 0: return 0.0
    thr = max(w.max() * thr_ratio, 1e-4)
    win = int(0.02*sr)
    n = 0
    while (n+1)*win < len(w) and w[n*win:(n+1)*win].max() < thr:
        n += 1
    return n*0.02

print("=== T1: official model, official mode ===", flush=True)
if os.path.isdir(OFFICIAL):
    om = OmniVoice.from_pretrained(OFFICIAL, device_map='cuda:0', dtype=torch.float16,
                                   attn_implementation='sdpa'); om.eval()
    for utt in UTTS[:3]:
        g,ptext,pwav,ttext=rows[utt]
        pw = os.path.join(SD,'prompt_wavs',os.path.basename(pwav))
        for post in (False, True):
            out = om.generate(text=ttext, ref_audio=pw, ref_text=ptext,
                              generation_config=OmniVoiceGenerationConfig(postprocess_output=post))
            wav = out[0] if isinstance(out, (list, tuple)) else out
            ls = lead_sil_sec(wav, int(om.sampling_rate))
            print("%s post=%s 头部低能量=%.2fs 总长=%.2fs" % (utt, post, ls, len(wav)/om.sampling_rate), flush=True)
    del om; torch.cuda.empty_cache()
else:
    print("OFFICIAL dir not found:", OFFICIAL, flush=True)

print("=== T2: mid-sentence continuation on B2 blockwise ===", flush=True)
mdir=tempfile.mkdtemp(prefix='b2mid_')
for f in ('model.safetensors','config.json'):
    os.symlink(os.path.abspath(os.path.join(CKPT,f)), os.path.join(mdir,f))
for f in ('audio_tokenizer','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
    s=os.path.abspath(os.path.join(BASE,f))
    if os.path.exists(s): os.symlink(s, os.path.join(mdir,f))
model=OmniVoice.from_pretrained(mdir,device_map='cuda:0',dtype=torch.float16,attn_implementation='sdpa'); model.eval()
tok=model.audio_tokenizer; tsr=int(model.sampling_rate); gen=OmniVoiceGenerationConfig()
os.makedirs('pause_verdict', exist_ok=True)
# mid-sentence: use each utt's own PROMPT sentence, split its text in half:
# seed = full prompt audio, but text = ptext + (nothing new)... instead:
# split ptext: feed text = full ptext, seed = first ~60% of prompt audio tokens.
for utt in UTTS:
    g,ptext,pwav,ttext=rows[utt]
    w,sr=sf.read(os.path.join(SD,'prompt_wavs',os.path.basename(pwav)))
    w=torch.tensor(w,dtype=torch.float32)
    if w.dim()>1: w=w.mean(-1)
    if sr!=tsr: w=AF.resample(w.unsqueeze(0),sr,tsr).squeeze(0)
    with torch.no_grad():
        enc=tok.encode(w.unsqueeze(0).unsqueeze(0).to(tok.device))
    at=enc.audio_codes if hasattr(enc,'audio_codes') else enc
    if isinstance(at,(list,tuple)): at=at[0]
    ref=at.squeeze()
    cut=int(ref.shape[1]*0.6)
    seed=ref[:, :cut]
    torch.manual_seed(20260707+g)
    # text = the FULL prompt sentence; audio seed = first 60% -> continuation is mid-sentence
    inp=model._prepare_inference_inputs(ptext, 32, None, None, 'zh', None, False)
    ii,am=inp['input_ids'],inp['audio_mask']
    if ii.dim()==3: ii=ii[0]
    if am.dim()==2: am=am[0]
    a0=ii.size(1)-int(am.sum())
    toks,_=_decode_block_causal(model,ii[:,:a0],gen,block_size=32,max_blocks=24,
        num_step_per_block=16,use_kv_cache=True,seed_audio=seed,
        min_gen_frames=8)
    cb0=toks[0]; lead=0
    while lead<toks.size(1) and cb0[lead].item()==0: lead+=1
    print("%s 句中续写: 生成%d帧 头部0串=%d帧" % (utt, toks.size(1), lead), flush=True)
    with torch.no_grad():
        wv=tok.decode(torch.cat([seed.to(toks.device),toks],dim=1).unsqueeze(0).to(tok.device)).audio_values[0]
    sf.write('pause_verdict/%s_midsent.wav'%utt, wv.float().cpu().numpy().reshape(-1), tsr)
print("VERDICT_DONE", flush=True)
