"""Last two suspects for the 16x training-context flex bwd slowdown:
B) template traced under active torch.autocast (training does this; benches did not)
C) autotune poisoning: first call on a degenerate (mostly-pad) mask freezes bad configs
A is the control (fresh compile, real-ish mask, no autocast)."""
import time, torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from omnivoice.blockdiff_dual import get_block_causal_mask_mod, TAG_PREFIX, TAG_CLEAN, TAG_NOISY

dev = "cuda"
L, HQ, HKV, D = 20480, 16, 8, 64
torch.manual_seed(0)

def mk_mask(doclens):
    doc_ids, tags, blks = [], [], []
    for d_, dl in enumerate(doclens):
        p = dl // 10; half = (dl - p) // 2
        doc_ids += [d_] * (p + 2 * half)
        tags += [TAG_PREFIX] * p + [TAG_CLEAN] * half + [TAG_NOISY] * half
        blks += [0] * p + [i // 32 for i in range(half)] + [i // 32 for i in range(half)]
    pad = L - len(doc_ids)
    doc_ids += [-1] * pad; tags += [-1] * pad; blks += [0] * pad
    t = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
    return create_block_mask(get_block_causal_mask_mod(t(doc_ids), t(tags), t(blks)),
                             B=None, H=None, Q_LEN=L, KV_LEN=L, _compile=True, device=dev)

bm_real = mk_mask([8622, 10498])
bm_degen = mk_mask([500])   # 500 tokens + 19980 pad — plausible tiny first pack
print(f"real sparsity={bm_real.sparsity():.1f}%  degen sparsity={bm_degen.sparsity():.1f}%")

q = torch.randn(1, HQ, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, HKV, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, HKV, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)

def measure(cf, bm, autocast=False, iters=10, warm=3):
    import contextlib
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    for _ in range(warm):
        with ctx:
            o = cf(q, k, v, block_mask=bm, enable_gqa=True)
        o.sum().backward(); q.grad = k.grad = v.grad = None
    torch.cuda.synchronize(); tf = tb = 0.0
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.time()
        with ctx:
            o = cf(q, k, v, block_mask=bm, enable_gqa=True)
        torch.cuda.synchronize(); tf += time.time() - t0
        s = o.sum()
        torch.cuda.synchronize(); t0 = time.time()
        s.backward()
        torch.cuda.synchronize(); tb += time.time() - t0
        q.grad = k.grad = v.grad = None
    return tf / iters * 1000, tb / iters * 1000

cfA = torch.compile(flex_attention, dynamic=False)
print("A control fresh+real          : fwd %7.2f  bwd %7.2f" % measure(cfA, bm_real))

cfB = torch.compile(flex_attention, dynamic=False)
print("B traced under autocast       : fwd %7.2f  bwd %7.2f" % measure(cfB, bm_real, autocast=True))

cfC = torch.compile(flex_attention, dynamic=False)
_ = measure(cfC, bm_degen, iters=2, warm=2)  # poison: first compile/autotune on degenerate mask
print("C degen-first, then real mask : fwd %7.2f  bwd %7.2f" % measure(cfC, bm_real))
