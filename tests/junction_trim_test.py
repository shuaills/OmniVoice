"""Bandaid test: trim leading (and trailing) cb0==0 frames from generated
tokens before vocoding. Token-level only — no model changes."""
import os, sys, torch, soundfile as sf
import torchaudio.functional as AF
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.blockdiff_dual import _decode_block_causal
import tempfile
CKPT='exp/block_b2/checkpoint-50000'; BASE='/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block'
SD='/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset/zh'
UTTS=["10002290-00000102","10002430-00000015","10002481-00000105","10002750-00000099",
      "10002753-00000006","10002287-00000095","10002287-00000099","10002309-00000033",
      "10002334-00000041","10003502-00000044","10004424-00000005","10002751-00000071"]
rows={}
for g,ln in enumerate(open(os.path.join(SD,'test.tsv'))):
    f=ln.rstrip('\n').split('\t')
    if len(f)>=4: rows[f[0]]=(g,f[1],f[2],f[3])
mdir=tempfile.mkdtemp(prefix='b2trim_')
for f in ('model.safetensors','config.json'):
    os.symlink(os.path.abspath(os.path.join(CKPT,f)), os.path.join(mdir,f))
for f in ('audio_tokenizer','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
    s=os.path.abspath(os.path.join(BASE,f))
    if os.path.exists(s): os.symlink(s, os.path.join(mdir,f))
model=OmniVoice.from_pretrained(mdir,device_map='cuda:0',dtype=torch.float16,attn_implementation='sdpa'); model.eval()
tok=model.audio_tokenizer; tsr=int(model.sampling_rate); gen=OmniVoiceGenerationConfig()
OUT='junction_trim'; os.makedirs(OUT, exist_ok=True)
import json
meta=[]
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
    torch.manual_seed(20260707+g)
    inp=model._prepare_inference_inputs(ttext,32,ptext,ref,'zh',None,False)
    ii,am=inp['input_ids'],inp['audio_mask']
    if ii.dim()==3: ii=ii[0]
    if am.dim()==2: am=am[0]
    a0=ii.size(1)-int(am.sum())
    toks,_=_decode_block_causal(model,ii[:,:a0],gen,block_size=32,max_blocks=24,
        num_step_per_block=16,use_kv_cache=True,seed_audio=ref,
        min_gen_frames=max(8,int(0.3*model._estimate_target_tokens(ttext,None,None))))
    cb0=toks[0]
    T=toks.size(1)
    lead=0
    while lead < T and cb0[lead].item() == 0: lead += 1
    tail=T
    while tail > lead and cb0[tail-1].item() == 0: tail -= 1
    trimmed = toks[:, lead:tail]
    if trimmed.size(1) == 0: trimmed = toks
    for tag, tt in (('RAW', toks), ('TRIM', trimmed)):
        with torch.no_grad():
            wv=tok.decode(tt.to(tok.device).unsqueeze(0)).audio_values[0]
        sf.write(f'{OUT}/{utt}_{tag}.wav', wv.float().cpu().numpy().reshape(-1), tsr)
    meta.append({'utt':utt,'T':T,'lead':lead,'tail_cut':T-tail})
    print(f'{utt} T={T} lead0={lead} tail0={T-tail}', flush=True)
json.dump(meta, open(f'{OUT}/meta.json','w'))
print('TRIM_GEN_DONE')
