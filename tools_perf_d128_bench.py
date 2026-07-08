"""Confirm the 16x root cause: head_dim=128 (real Qwen3-0.6B shape) vs the
D=64 my earlier benches wrongly used. Then attack: kernel_options + compile
mode on the standalone flex call (the singleton pattern, NOT whole-graph)."""
import time, torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from omnivoice.blockdiff_dual import get_block_causal_mask_mod

dev = "cuda"
L, HQ, HKV = 20480, 16, 8
torch.manual_seed(0)
doc_ids, tags, blks = [], [], []
for d_, dl in ((0, 8622), (1, 10498)):
    p = dl // 10; half = (dl - p) // 2
    doc_ids += [d_] * (p + 2 * half)
    tags += [0] * p + [1] * half + [2] * half
    blks += [0] * p + [i // 32 for i in range(half)] + [i // 32 for i in range(half)]
pad = L - len(doc_ids)
doc_ids += [-1] * pad; tags += [-1] * pad; blks += [0] * pad
t = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
bm = create_block_mask(get_block_causal_mask_mod(t(doc_ids), t(tags), t(blks)),
                       B=None, H=None, Q_LEN=L, KV_LEN=L, _compile=True, device=dev)

def bench(label, D, fn=None, iters=8, **kw):
    q = torch.randn(1, HQ, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, HKV, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, HKV, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    f = fn or torch.compile(flex_attention, dynamic=False)
    try:
        for _ in range(3):
            o = f(q, k, v, block_mask=bm, enable_gqa=True, **kw)
            o.sum().backward(); q.grad = k.grad = v.grad = None
        torch.cuda.synchronize(); tf = tb = 0.0
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.time()
            o = f(q, k, v, block_mask=bm, enable_gqa=True, **kw)
            torch.cuda.synchronize(); tf += time.time() - t0
            s = o.sum()
            torch.cuda.synchronize(); t0 = time.time()
            s.backward()
            torch.cuda.synchronize(); tb += time.time() - t0
            q.grad = k.grad = v.grad = None
        print(f"{label}: fwd {tf/iters*1000:7.2f}  bwd {tb/iters*1000:7.2f} ms")
    except Exception as e:
        print(f"{label}: FAILED {type(e).__name__}: {str(e)[:100]}")

bench("D=64  control (old wrong shape)  ", 64)
bench("D=128 REAL shape, default compile", 128)
bench("D=128 max-autotune-no-cudagraphs ", 128,
      fn=torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"))
for opts in ({"BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32},
             {"BLOCK_M1": 16, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 16},
             {"BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32},
             {"num_stages": 1}):
    bench(f"D=128 opts {str(opts)[:38]:38}", 128, kernel_options=opts)
