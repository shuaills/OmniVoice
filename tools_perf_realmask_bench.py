"""Bench flex fwd/bwd on a REAL batch's mask (captured from the b2g dataloader)
vs the synthetic one. Adds backward kernel_options variants. Answers: is the
51.5ms/layer backward reproduced by mask geometry alone, and can options fix it?"""
import torch, time
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from omnivoice.blockdiff_dual import get_block_causal_mask_mod
from omnivoice.training.config import TrainingConfig
from omnivoice.training.builder import build_model_and_tokenizer, build_dataloaders
import omnivoice.training.builder as B

config = TrainingConfig.from_json("examples/config/train_config_perf.json")
config.output_dir = "/tmp/perf_realmask_out"
config.data_config = "examples/config/data_config_internal_b2g.json"
config.num_workers = 0

# tokenizer only; skip weight load: build tokenizer via the builder's path
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config.llm_name_or_path)
train_loader, _ = build_dataloaders(config, tokenizer)
batch = next(iter(train_loader))
dev = "cuda"
doc_ids = batch["document_ids"][0].to(dev)
tags = batch["copy_tags"][0].to(dev)
blks = batch["block_ids"][0].to(dev)
L = doc_ids.numel()
ndocs = int(doc_ids.max().item()) + 1
lens = [(doc_ids == i).sum().item() for i in range(ndocs)]
print(f"REAL batch: L={L} docs={ndocs} lens={lens}")

bm = create_block_mask(get_block_causal_mask_mod(doc_ids, tags, blks),
                       B=None, H=None, Q_LEN=L, KV_LEN=L, _compile=True, device=dev)
print(f"REAL sparsity={bm.sparsity():.1f}%")

HQ, HKV, D = 16, 8, 64
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
bench("A real, gqa            ", lambda: cf(q, k, v, block_mask=bm, enable_gqa=True))
# training-layout variant: projections produce [1, L, H, D] then transpose ->
# strided views, unlike the contiguous alloc above
qs = torch.randn(1, L, HQ, D, device=dev, dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
ks = torch.randn(1, L, HKV, D, device=dev, dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
vs = torch.randn(1, L, HKV, D, device=dev, dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
def bench_s(label, fn, iters=10):
    for _ in range(3):
        r = fn(); (r[0] if isinstance(r, tuple) else r).sum().backward()
        qs.grad = ks.grad = vs.grad = None
    torch.cuda.synchronize(); tf = tb = 0.0
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.time()
        r = fn()
        torch.cuda.synchronize(); tf += time.time() - t0
        ss = (r[0] if isinstance(r, tuple) else r).sum()
        torch.cuda.synchronize(); t0 = time.time()
        ss.backward()
        torch.cuda.synchronize(); tb += time.time() - t0
        qs.grad = ks.grad = vs.grad = None
    print(f"{label}: fwd {tf/iters*1000:7.2f} ms  bwd {tb/iters*1000:7.2f} ms")
bench_s("S real, strided layout ", lambda: cf(qs, ks, vs, block_mask=bm, enable_gqa=True))
bench("B real, kv repeated     ", lambda: cf(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), block_mask=bm))
from transformers.integrations.flex_attention import flex_attention_forward
class Dummy(torch.nn.Module):
    pass
mod = Dummy(); mod.training = True
bench("E real, exact HF wrapper", lambda: flex_attention_forward(mod, q, k, v, attention_mask=bm, scaling=None)[0])
for opts in ({"BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32},
             {"num_warps": 8},
             {"num_stages": 2}):
    try:
        bench(f"K real opts {str(opts)[:34]:34}", lambda: cf(q, k, v, block_mask=bm, enable_gqa=True, kernel_options=opts))
    except Exception as e:
        print(f"K {opts}: FAILED {type(e).__name__}: {str(e)[:80]}")
