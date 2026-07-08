"""Flex fwd/bwd on the captured real mask under the CURRENT allocator config,
after fragmenting memory like a training process would. Compare across
PYTORCH_CUDA_ALLOC_CONF settings (set by wrapper)."""
import os, time, torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from omnivoice.blockdiff_dual import get_block_causal_mask_mod, TAG_PREFIX, TAG_CLEAN, TAG_NOISY

print("ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<default>"))
dev = "cuda"
L, HQ, HKV, D = 20480, 16, 8, 64
torch.manual_seed(0)

# fragment ~50GB like a resident training process (weights+activations+optimizer)
junk = []
for _ in range(400):
    junk.append(torch.randn(32 * 1024 * 1024 // 2, device=dev, dtype=torch.bfloat16))  # 32MB
for i in range(0, 400, 2):
    junk[i] = None  # free every other block -> fragmentation
frag = torch.randn(8 * 1024 * 1024 * 1024 // 2, device=dev, dtype=torch.bfloat16)  # 8GB slab
print(f"reserved={torch.cuda.memory_reserved()/2**30:.1f}GB allocated={torch.cuda.memory_allocated()/2**30:.1f}GB")

doc_ids, tags, blks = [], [], []
for d_, dl in ((0, 8622), (1, 10498)):
    p = dl // 10; half = (dl - p) // 2
    doc_ids += [d_] * (p + 2 * half)
    tags += [TAG_PREFIX] * p + [TAG_CLEAN] * half + [TAG_NOISY] * half
    blks += [0] * p + [i // 32 for i in range(half)] + [i // 32 for i in range(half)]
pad = L - len(doc_ids)
doc_ids += [-1] * pad; tags += [-1] * pad; blks += [0] * pad
doc_ids = torch.tensor(doc_ids, dtype=torch.int32, device=dev)
tags = torch.tensor(tags, dtype=torch.int32, device=dev)
blks = torch.tensor(blks, dtype=torch.int32, device=dev)
bm = create_block_mask(get_block_causal_mask_mod(doc_ids, tags, blks),
                       B=None, H=None, Q_LEN=L, KV_LEN=L, _compile=True, device=dev)
q = torch.randn(1, HQ, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, HKV, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, HKV, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
cf = torch.compile(flex_attention, dynamic=False)
for _ in range(3):
    o = cf(q, k, v, block_mask=bm, enable_gqa=True); o.sum().backward()
    q.grad = k.grad = v.grad = None
torch.cuda.synchronize(); tf = tb = 0.0
for _ in range(10):
    torch.cuda.synchronize(); t0 = time.time()
    o = cf(q, k, v, block_mask=bm, enable_gqa=True)
    torch.cuda.synchronize(); tf += time.time() - t0
    s = o.sum()
    torch.cuda.synchronize(); t0 = time.time()
    s.backward()
    torch.cuda.synchronize(); tb += time.time() - t0
    q.grad = k.grad = v.grad = None
print(f"RESULT fwd {tf:.1f}ms/10 bwd {tb*100:.1f}ms/10 -> fwd {tf*100:.2f} ms  bwd {tb*100:.2f} ms")
