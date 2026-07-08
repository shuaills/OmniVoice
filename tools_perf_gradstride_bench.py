"""Does a transposed-strided grad_out (what HF's transpose backward feeds flex)
explain the 16x backward slowdown?"""
import time, torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from omnivoice.blockdiff_dual import get_block_causal_mask_mod

dev = "cuda"
L, HQ, HKV, D = 20480, 16, 8, 64
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
q = torch.randn(1, HQ, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, HKV, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, HKV, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
cf = torch.compile(flex_attention, dynamic=False)

g_contig = torch.randn(1, HQ, L, D, device=dev, dtype=torch.bfloat16)
g_strided = torch.randn(1, L, HQ, D, device=dev, dtype=torch.bfloat16).transpose(1, 2)

def bench(label, grad):
    for _ in range(3):
        o = cf(q, k, v, block_mask=bm, enable_gqa=True)
        o.backward(gradient=grad); q.grad = k.grad = v.grad = None
    torch.cuda.synchronize(); tb = 0.0
    for _ in range(10):
        o = cf(q, k, v, block_mask=bm, enable_gqa=True)
        torch.cuda.synchronize(); t0 = time.time()
        o.backward(gradient=grad)
        torch.cuda.synchronize(); tb += time.time() - t0
        q.grad = k.grad = v.grad = None
    print(f"{label}: bwd {tb*100:7.2f} ms")

bench("G1 contiguous grad_out         ", g_contig)
bench("G2 transposed-strided grad_out ", g_strided)
bench("G3 strided->contiguous grad_out", g_strided.contiguous())
