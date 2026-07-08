"""Micro-bench flex_attention fwd/bwd on realistic B2 dual-copy geometry.
Isolates the 71% cost center: is the backward template mistuned, and does
max-autotune fix it? Single GPU, ~3 min."""
import torch, time
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from omnivoice.blockdiff_dual import get_block_causal_mask_mod, TAG_PREFIX, TAG_CLEAN, TAG_NOISY

torch.manual_seed(0)
L, H, D = 20480, 16, 64  # Qwen3-0.6B: 16 heads x 64
dev = "cuda"

# realistic pack: 9 docs ~2270 tok each: 10% prefix + 45% clean + 45% noisy, block 32
doc_ids, tags, blks = [], [], []
ndoc, dlen = 9, 2270
for d in range(ndoc):
    p = dlen // 10
    half = (dlen - p) // 2
    doc_ids += [d] * (p + 2 * half)
    tags += [TAG_PREFIX] * p + [TAG_CLEAN] * half + [TAG_NOISY] * half
    blks += [0] * p + [i // 32 for i in range(half)] + [i // 32 for i in range(half)]
pad = L - len(doc_ids)
doc_ids += [-1] * pad; tags += [-1] * pad; blks += [0] * pad
doc_ids = torch.tensor(doc_ids, dtype=torch.int32, device=dev)
tags = torch.tensor(tags, dtype=torch.int32, device=dev)
blks = torch.tensor(blks, dtype=torch.int32, device=dev)

mask_mod = get_block_causal_mask_mod(doc_ids, tags, blks)
t0 = time.time()
bm = create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, _compile=True, device=dev)
torch.cuda.synchronize()
print(f"create_block_mask: {time.time()-t0:.2f}s (includes compile); sparsity={bm.sparsity():.1f}%")
# steady-state mask build cost
t0 = time.time()
for _ in range(5):
    create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, _compile=True, device=dev)
torch.cuda.synchronize()
print(f"create_block_mask steady: {(time.time()-t0)/5*1000:.1f} ms/call")

q = torch.randn(1, H, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, H, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, H, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)

def bench(fn, label, iters=15):
    for _ in range(3):
        out = fn(q, k, v, block_mask=bm)
        out.sum().backward()
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    tf = tb = 0.0
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.time()
        out = fn(q, k, v, block_mask=bm)
        torch.cuda.synchronize(); tf += time.time() - t0
        g = out.sum()
        torch.cuda.synchronize(); t0 = time.time()
        g.backward()
        torch.cuda.synchronize(); tb += time.time() - t0
        q.grad = k.grad = v.grad = None
    print(f"{label}: fwd {tf/iters*1000:7.2f} ms  bwd {tb/iters*1000:7.2f} ms  ratio {tb/max(tf,1e-9):.1f}")

bench(torch.compile(flex_attention, dynamic=False), "compile-default        ")
bench(torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"),
      "max-autotune-no-cg     ")
