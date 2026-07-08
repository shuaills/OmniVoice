"""Ground-truth step decomposition, no profiler: explicit cuda-synced timing
of data / h2d / forward / backward / optimizer for 12 full-model steps."""
import time, torch
from omnivoice.training.config import TrainingConfig
from omnivoice.training.builder import build_model_and_tokenizer, build_dataloaders

config = TrainingConfig.from_json("examples/config/train_config_perf.json")
config.output_dir = "/tmp/perf_phase_out"
config.data_config = "examples/config/data_config_internal_b2g.json"
config.num_workers = 4

model, tokenizer = build_model_and_tokenizer(config)
model = model.cuda(); model.train()
train_loader, _ = build_dataloaders(config, tokenizer)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

def get_loss(out):
    if isinstance(out, dict): return out["loss"]
    if hasattr(out, "loss") and out.loss is not None: return out.loss
    return out[0]

it = iter(train_loader)
print(f"{'step':>4} {'data':>7} {'h2d':>6} {'fwd':>7} {'bwd':>7} {'opt':>6} {'total':>7}  (ms)")
for step in range(12):
    t0 = time.time(); batch = next(it); t_data = time.time() - t0
    t0 = time.time()
    batch = {k: (v.cuda(non_blocking=False) if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.cuda.synchronize(); t_h2d = time.time() - t0
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**batch)
    loss = get_loss(out)
    torch.cuda.synchronize(); t_fwd = time.time() - t0
    t0 = time.time(); loss.backward()
    torch.cuda.synchronize(); t_bwd = time.time() - t0
    t0 = time.time()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t_opt = time.time() - t0
    tot = t_data + t_h2d + t_fwd + t_bwd + t_opt
    print(f"{step:>4} {t_data*1000:7.0f} {t_h2d*1000:6.0f} {t_fwd*1000:7.0f} {t_bwd*1000:7.0f} {t_opt*1000:6.0f} {tot*1000:7.0f}")
