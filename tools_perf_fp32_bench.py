"""Final confirmation: fp32 q/k/v at D=128 (what training actually feeds flex)
vs bf16. Plus DeepSpec lead-3 variants: repeat_interleave KV, forced contiguous."""
import time, torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from omnivoice.blockdiff_dual import get_block_causal_mask_mod

dev = "cuda"
L, HQ, HKV, D = 20480, 16, 8, 128
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
cf = torch.compile(flex_attention, dynamic=False)

def bench(label, dtype, rep_kv=False, contig=True):
    q = torch.randn(1, L, HQ, D, device=dev, dtype=dtype).transpose(1, 2)
    k = torch.randn(1, L, HKV, D, device=dev, dtype=dtype).transpose(1, 2)
    v = torch.randn(1, L, HKV, D, device=dev, dtype=dtype).transpose(1, 2)
    if contig:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if rep_kv:
        k = k.repeat_interleave(2, 1); v = v.repeat_interleave(2, 1)
    q.requires_grad_(); k.requires_grad_(); v.requires_grad_()
    kw = {} if rep_kv else {"enable_gqa": True}
    for _ in range(3):
        o = cf(q, k, v, block_mask=bm, **kw)
        o.sum().backward(); q.grad = k.grad = v.grad = None
    torch.cuda.synchronize(); tf = tb = 0.0
    for _ in range(8):
        torch.cuda.synchronize(); t0 = time.time()
        o = cf(q, k, v, block_mask=bm, **kw)
        torch.cuda.synchronize(); tf += time.time() - t0
        s = o.sum()
        torch.cuda.synchronize(); t0 = time.time()
        s.backward()
        torch.cuda.synchronize(); tb += time.time() - t0
        q.grad = k.grad = v.grad = None
    print(f"{label}: fwd {tf/8*1000:7.2f}  bwd {tb/8*1000:7.2f} ms")

bench("bf16 D128 contig (ref)          ", torch.bfloat16)
bench("FP32 D128 contig  <- TRAINING   ", torch.float32)
bench("FP32 D128 strided (exact train) ", torch.float32, contig=False)
bench("FP32 D128 repeat_interleave KV  ", torch.float32, rep_kv=True)
bench("bf16 D128 strided               ", torch.bfloat16, contig=False)
