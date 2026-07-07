"""Is the onset 'hum' = cb0 silence + speech-like residual channels?
Compare all-channel tokens of (a) encoded true silence, (b) our generation's
leading cb0=0 run, (c) mid-speech frames."""
import os, sys, torch, soundfile as sf
import torchaudio.functional as AF
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.blockdiff_dual import _decode_block_causal
import tempfile
SD='/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset/zh'
CKPT='exp/block_b2/checkpoint-50000'
BASE='/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block'
rows={}
for g,ln in enumerate(open(os.path.join(SD,'test.tsv'))):
    f=ln.rstrip('\n').split('\t')
    if len(f)>=4: rows[f[0]]=(g,f[1],f[2],f[3])
mdir=tempfile.mkdtemp(prefix='b2hum_')
for f in ('model.safetensors','config.json'):
    os.symlink(os.path.abspath(os.path.join(CKPT,f)), os.path.join(mdir,f))
for f in ('audio_tokenizer','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
    s=os.path.abspath(os.path.join(BASE,f))
    if os.path.exists(s): os.symlink(s, os.path.join(mdir,f))
model=OmniVoice.from_pretrained(mdir,device_map='cuda:0',dtype=torch.float16,attn_implementation='sdpa'); model.eval()
tok=model.audio_tokenizer; tsr=int(model.sampling_rate); gen=OmniVoiceGenerationConfig()
# (a) true silence tokens
sil = torch.zeros(1,1,tsr*2)
with torch.no_grad():
    enc=tok.encode(sil.to(tok.device))
at=enc.audio_codes if hasattr(enc,'audio_codes') else enc
if isinstance(at,(list,tuple)): at=at[0]
sil_toks=at.squeeze()
print("true-silence tokens per channel (mode of 50 frames):")
for c in range(8):
    vals=sil_toks[c,:50].tolist()
    mode=max(set(vals), key=vals.count)
    print("  cb%d: mode=%d uniq=%d" % (c, mode, len(set(vals))))
# (b) one continuation generation
utt="10002430-00000015"
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
cb0=toks[0]; lead=0
while lead<toks.size(1) and cb0[lead].item()==0: lead+=1
print("\ngeneration lead0=%d frames; tokens of first %d frames:" % (lead, min(lead,12)))
sil_modes=[max(set(sil_toks[c,:50].tolist()), key=sil_toks[c,:50].tolist().count) for c in range(8)]
for c in range(8):
    seg=toks[c,:min(lead,12)].tolist()
    match=sum(1 for v in seg if v==sil_modes[c])
    print("  cb%d: %s  (=真静音众数的比例 %d/%d)" % (c, seg[:8], match, len(seg)))
print("HUM_CHECK_DONE")
