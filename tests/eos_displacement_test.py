"""Does the onset silence track the EOS-ban window? min_gen_frames in {0,8,30},
sentence-continuation setting. Also dump p(eos) at first-block masked positions
on the first denoise iteration (pre-ban posterior)."""
import os, sys, torch, soundfile as sf
import torchaudio.functional as AF
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.blockdiff_dual import _decode_block_causal, block_eos_id
import tempfile
SD='/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset/zh'
CKPT='exp/block_b2/checkpoint-50000'
BASE='/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block'
UTTS=["10002430-00000015","10002481-00000105","10002753-00000006","10003502-00000044"]
rows={}
for g,ln in enumerate(open(os.path.join(SD,'test.tsv'))):
    f=ln.rstrip('\n').split('\t')
    if len(f)>=4: rows[f[0]]=(g,f[1],f[2],f[3])
mdir=tempfile.mkdtemp(prefix='b2eos_')
for f in ('model.safetensors','config.json'):
    os.symlink(os.path.abspath(os.path.join(CKPT,f)), os.path.join(mdir,f))
for f in ('audio_tokenizer','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
    s=os.path.abspath(os.path.join(BASE,f))
    if os.path.exists(s): os.symlink(s, os.path.join(mdir,f))
model=OmniVoice.from_pretrained(mdir,device_map='cuda:0',dtype=torch.float16,attn_implementation='sdpa'); model.eval()
tok=model.audio_tokenizer; tsr=int(model.sampling_rate); gen=OmniVoiceGenerationConfig()

def prep(utt):
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
    inp=model._prepare_inference_inputs(ttext,32,ptext,ref,'zh',None,False)
    ii,am=inp['input_ids'],inp['audio_mask']
    if ii.dim()==3: ii=ii[0]
    if am.dim()==2: am=am[0]
    a0=ii.size(1)-int(am.sum())
    return g, ii[:,:a0], ref

print("=== lead0 vs ban window ===", flush=True)
for utt in UTTS:
    g, prefix, ref = prep(utt)
    line=[utt]
    for mg in (0, 8, 30):
        torch.manual_seed(20260707+g)
        toks,stats=_decode_block_causal(model,prefix,gen,block_size=32,max_blocks=24,
            num_step_per_block=16,use_kv_cache=True,seed_audio=ref,min_gen_frames=mg)
        cb0=toks[0]; lead=0
        while lead<toks.size(1) and cb0[lead].item()==0: lead+=1
        line.append("ban=%d: T=%d lead0=%d eos@%s" % (mg, toks.size(1), lead,
                    stats.get('eos_block','?')))
    print("  ".join(str(x) for x in line), flush=True)
print("EOS_DISP_DONE", flush=True)
