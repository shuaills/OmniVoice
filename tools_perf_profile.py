"""50-step single-GPU profile of the b2g train step: 35 warmup, 8 profiled.
Exports chrome trace + prints key_averages by CUDA time. Mirrors trainer:
outputs = model(**batch); AdamW; bf16 autocast."""
import sys, time, torch
from torch.profiler import profile, schedule, ProfilerActivity
from omnivoice.training.config import TrainingConfig
from omnivoice.training.builder import build_model_and_tokenizer, build_dataloaders

train_config, out_trace = sys.argv[1], sys.argv[2]
config = TrainingConfig.from_json(train_config)
config.output_dir = "/tmp/perf_profile_out"
config.data_config = "examples/config/data_config_internal_b2g.json"
config.num_workers = 4

model, tokenizer = build_model_and_tokenizer(config)
model = model.cuda()
model.train()
train_loader, _ = build_dataloaders(config, tokenizer)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

def get_loss(out):
    if isinstance(out, dict):
        return out["loss"]
    if hasattr(out, "loss") and out.loss is not None:
        return out.loss
    return out[0]

it = iter(train_loader)
prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=30, warmup=5, active=8),
    on_trace_ready=lambda p: p.export_chrome_trace(out_trace),
)
prof.start()
t0 = time.time()
for step in range(45):
    batch = next(it)
    batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
             for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**batch)
    loss = get_loss(out)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    prof.step()
    if step % 10 == 0:
        torch.cuda.synchronize()
        print(f"profile step {step} wall={time.time()-t0:.1f}s loss={loss.item():.4f}", flush=True)
prof.stop()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
print("TRACE_WRITTEN", out_trace)
