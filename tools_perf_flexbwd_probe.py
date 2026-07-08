"""Settle world (a) vs (b) for the 16x flex-bwd anomaly.
Instruments REAL single-GPU training steps with:
1. CUDA-event windows around ONLY the flex backward autograd node
   (StartMark on flex output fires just before it; EndMark on q/k/v fires
   just after) -> device wall incl. stream gaps, per layer x step.
2. torch profiler (CUPTI) on 2 steps -> raw kernel exec durations.
Compare: window ~51ms & kernel ~51ms => (a) slow kernel in context.
         window ~51ms & kernel ~3ms  => (b) ~48ms stream stall inside node.
         window ~3ms                 => profiler op-attribution lied entirely.
"""
import sys, time, torch
from omnivoice.training.config import TrainingConfig
from omnivoice.training.builder import build_model_and_tokenizer, build_dataloaders

NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8

records = []   # (step, call, ev_start, ev_end)
state = {"step": -1}

class _StartMark(torch.autograd.Function):
    @staticmethod
    def forward(ctx, idx, t):
        ctx.idx = idx
        return t
    @staticmethod
    def backward(ctx, g):
        records[ctx.idx][2].record()
        return None, g

class _EndMark(torch.autograd.Function):
    @staticmethod
    def forward(ctx, idx, q, k, v):
        ctx.idx = idx
        return q, k, v
    @staticmethod
    def backward(ctx, gq, gk, gv):
        records[ctx.idx][3].record()
        return None, gq, gk, gv

import transformers.integrations.flex_attention as fa_mod
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
_orig = fa_mod.flex_attention_forward

def probed(module, query, key, value, attention_mask, *a, **kw):
    idx = len(records)
    records.append([state["step"], idx, torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True)])
    query, key, value = _EndMark.apply(idx, query, key, value)
    out, lse = _orig(module, query, key, value, attention_mask, *a, **kw)
    out = _StartMark.apply(idx, out)
    return out, lse

fa_mod.flex_attention_forward = probed
try:
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = probed
except Exception:
    ALL_ATTENTION_FUNCTIONS.register("flex_attention", probed)

config = TrainingConfig.from_json("examples/config/train_config_perf.json")
config.output_dir = "/tmp/perf_probe_out"
config.data_config = "examples/config/data_config_internal_b2g.json"
config.num_workers = 4
model, tok = build_model_and_tokenizer(config)
model = model.cuda(); model.train()
loader, _ = build_dataloaders(config, tok)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

def get_loss(o):
    if isinstance(o, dict): return o["loss"]
    if hasattr(o, "loss") and o.loss is not None: return o.loss
    return o[0]

from torch.profiler import profile, schedule, ProfilerActivity
prof = profile(activities=[ProfilerActivity.CUDA],
               schedule=schedule(wait=NSTEPS - 3, warmup=1, active=2))
prof.start()
it = iter(loader)
bwd_walls = []
for step in range(NSTEPS):
    state["step"] = step
    batch = next(it)
    batch = {k2: (v2.cuda() if torch.is_tensor(v2) else v2) for k2, v2 in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**batch)
    loss = get_loss(out)
    torch.cuda.synchronize(); t0 = time.time()
    loss.backward()
    torch.cuda.synchronize(); bwd_walls.append(time.time() - t0)
    opt.step(); opt.zero_grad(set_to_none=True)
    prof.step()
prof.stop()
torch.cuda.synchronize()

per_step = {}
for s, i, e0, e1 in records:
    try:
        ms = e0.elapsed_time(e1)
    except Exception:
        continue
    per_step.setdefault(s, []).append(ms)
print("\n=== flex-bwd node windows (device ms) per step ===")
print(f"{'step':>4} {'n':>3} {'sum':>8} {'mean':>7} {'min':>7} {'max':>7} {'bwd_wall_ms':>11}")
for s in sorted(per_step):
    w = per_step[s]
    print(f"{s:>4} {len(w):>3} {sum(w):8.1f} {sum(w)/len(w):7.2f} {min(w):7.2f} {max(w):7.2f} {bwd_walls[s]*1000:11.0f}")
lay = per_step.get(NSTEPS - 1, [])
if lay:
    print("last step per-layer (reverse exec order):",
          " ".join(f"{x:.1f}" for x in lay))
print("\n=== CUPTI kernel durations (profiled steps), tem/flex kernels ===")
for ka in prof.key_averages():
    if "tem" in ka.key or "flex" in ka.key.lower():
        print(f"{ka.key[:60]:60} n={ka.count} mean={ka.device_time/1000:.2f}ms total={ka.device_time_total/1e6:.3f}s")
