"""Print the ACTUAL dtypes/shapes/strides/mask entering flex in real training."""
import torch
from omnivoice.training.config import TrainingConfig
from omnivoice.training.builder import build_model_and_tokenizer, build_dataloaders

import transformers.integrations.flex_attention as fa_mod
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
_orig = fa_mod.flex_attention_forward
printed = {"n": 0}

def probed(module, query, key, value, attention_mask, *a, **kw):
    if printed["n"] < 3:
        printed["n"] += 1
        bm = attention_mask
        print(f"CALL q={tuple(query.shape)} {query.dtype} contig={query.is_contiguous()} "
              f"k={key.dtype} v={value.dtype} "
              f"mask={type(bm).__name__} sparsity={bm.sparsity():.1f}%", flush=True)
        print(f"  autocast_enabled={torch.is_autocast_enabled('cuda')} "
              f"vproj_w={module.v_proj.weight.dtype} "
              f"qnorm_w={module.q_norm.weight.dtype}", flush=True)
        print(f"  scaling_arg={a[:1] if a else None} kwargs_keys={sorted(kw.keys())}", flush=True)
    return _orig(module, query, key, value, attention_mask, *a, **kw)

fa_mod.flex_attention_forward = probed
try:
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = probed
except Exception:
    ALL_ATTENTION_FUNCTIONS.register("flex_attention", probed)

config = TrainingConfig.from_json("examples/config/train_config_perf.json")
config.output_dir = "/tmp/perf_dtype_out"
config.data_config = "examples/config/data_config_internal_b2g.json"
config.num_workers = 2
model, tok = build_model_and_tokenizer(config)
model = model.cuda(); model.train()
loader, _ = build_dataloaders(config, tok)
it = iter(loader)
for step in range(2):
    batch = next(it)
    batch = {k2: (v2.cuda() if torch.is_tensor(v2) else v2) for k2, v2 in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**batch)
    loss = out["loss"] if isinstance(out, dict) else (out.loss if hasattr(out, "loss") else out[0])
    loss.backward()
    model.zero_grad(set_to_none=True)
    print(f"step {step} done", flush=True)
