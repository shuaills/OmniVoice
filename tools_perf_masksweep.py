"""Sweep flex bwd cost across 20 REAL batches (one process, compile once).
Per mask: doc stats, sparsity, fwd/bwd ms. If slow masks exist, retry the
slowest with BlockMask block_size=64 to test the partial-block hypothesis."""
import time, torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from omnivoice.blockdiff_dual import get_block_causal_mask_mod
from omnivoice.training.config import TrainingConfig
from omnivoice.training.builder import build_dataloaders
from transformers import AutoTokenizer

config = TrainingConfig.from_json("examples/config/train_config_perf.json")
config.output_dir = "/tmp/perf_sweep_out"
config.data_config = "examples/config/data_config_internal_b2g.json"
config.num_workers = 2
tokenizer = AutoTokenizer.from_pretrained(config.llm_name_or_path)
train_loader, _ = build_dataloaders(config, tokenizer)

dev = "cuda"
HQ, HKV, D = 16, 8, 64
cf = torch.compile(flex_attention, dynamic=False)
q = torch.randn(1, HQ, 20480, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, HKV, 20480, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, HKV, 20480, D, device=dev, dtype=torch.bfloat16, requires_grad=True)

def timed(bm, iters=5):
    for _ in range(2):
        o = cf(q, k, v, block_mask=bm, enable_gqa=True); o.sum().backward()
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize(); tf = tb = 0.0
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.time()
        o = cf(q, k, v, block_mask=bm, enable_gqa=True)
        torch.cuda.synchronize(); tf += time.time() - t0
        s = o.sum()
        torch.cuda.synchronize(); t0 = time.time()
        s.backward()
        torch.cuda.synchronize(); tb += time.time() - t0
        q.grad = k.grad = v.grad = None
    return tf / iters * 1000, tb / iters * 1000

it = iter(train_loader)
worst = (0.0, None, None)
print(f"{'i':>2} {'docs':>4} {'minlen':>6} {'medlen':>6} {'spars':>6} {'fwd':>7} {'bwd':>8}")
for i in range(20):
    b = next(it)
    d, t_, bl = b["document_ids"][0].to(dev), b["copy_tags"][0].to(dev), b["block_ids"][0].to(dev)
    nd = int(d.max().item()) + 1
    lens = sorted((d == j).sum().item() for j in range(nd))
    bm = create_block_mask(get_block_causal_mask_mod(d, t_, bl), B=None, H=None,
                           Q_LEN=20480, KV_LEN=20480, _compile=True, device=dev)
    tf, tb = timed(bm)
    print(f"{i:>2} {nd:>4} {lens[0]:>6} {lens[len(lens)//2]:>6} {bm.sparsity():5.1f}% {tf:7.2f} {tb:8.2f}", flush=True)
    if tb > worst[0]:
        worst = (tb, (d, t_, bl), i)

if worst[0] > 10 and worst[1] is not None:
    d, t_, bl = worst[1]
    print(f"retrying worst mask (batch {worst[2]}, bwd {worst[0]:.1f}ms) with BLOCK_SIZE=64:")
    bm64 = create_block_mask(get_block_causal_mask_mod(d, t_, bl), B=None, H=None,
                             Q_LEN=20480, KV_LEN=20480, BLOCK_SIZE=64, _compile=True, device=dev)
    tf, tb = timed(bm64)
    print(f"BLOCK_SIZE=64: fwd {tf:7.2f}  bwd {tb:8.2f}")
