"""Bisect the 31x flex backward anomaly. Exact Qwen3-0.6B GQA shapes
(16 q-heads / 8 kv-heads, D=64, L=20480) on the real B2 mask. Toggles:
lse return, identity score_mod, enable_gqa, and the exact HF wrapper."""
import torch, time
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from omnivoice.blockdiff_dual import get_block_causal_mask_mod, TAG_PREFIX, TAG_CLEAN, TAG_NOISY

torch.manual_seed(0)
L, HQ, HKV, D = 20480, 16, 8, 64
dev = "cuda"

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
bm = create_block_mask(get_block_causal_mask_mod(doc_ids, tags, blks),
                       B=None, H=None, Q_LEN=L, KV_LEN=L, _compile=True, device=dev)
print(f"sparsity={bm.sparsity():.1f}%")

q = torch.randn(1, HQ, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, HKV, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, HKV, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)

def bench(label, fn, iters=10):
    for _ in range(3):
        r = fn()
        (r[0] if isinstance(r, tuple) else r).sum().backward()
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    tf = tb = 0.0
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.time()
        r = fn()
        torch.cuda.synchronize(); tf += time.time() - t0
        s = (r[0] if isinstance(r, tuple) else r).sum()
        torch.cuda.synchronize(); t0 = time.time()
        s.backward()
        torch.cuda.synchronize(); tb += time.time() - t0
        q.grad = k.grad = v.grad = None
    print(f"{label}: fwd {tf/iters*1000:7.2f} ms  bwd {tb/iters*1000:7.2f} ms")

cf = torch.compile(flex_attention, dynamic=False)

def identity_score_mod(score, b, h, qi, ki):
    return score

bench("A gqa, no lse, no smod ", lambda: cf(q, k, v, block_mask=bm, enable_gqa=True))
bench("B gqa, lse            ", lambda: cf(q, k, v, block_mask=bm, enable_gqa=True, return_lse=True))
bench("C gqa, lse, id smod   ", lambda: cf(q, k, v, block_mask=bm, enable_gqa=True, return_lse=True, score_mod=identity_score_mod))
bench("D lse only, no gqa(rep)", lambda: cf(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), block_mask=bm, return_lse=True))

# E: the exact HF wrapper (its own singleton compile, training=True)
from transformers.integrations.flex_attention import flex_attention_forward
class Dummy(torch.nn.Module):
    pass
mod = Dummy(); mod.training = True
def hf_call():
    out, lse = flex_attention_forward(mod, q, k, v, attention_mask=bm, scaling=None)
    return out
bench("E exact HF wrapper     ", hf_call)
