"""Compare step-matched tqdm loss values between two training logs.
Usage: tools_perf_parity.py BASE_LOG ARM_LOG  (steps 10..100 by 10)
Same seed + same GPU count => same data order; rel-diff beyond bf16 kernel
noise or any NaN => parity FAIL."""
import re, sys

def losses(path):
    text = open(path, errors="ignore").read()
    out = {}
    for step in range(10, 110, 10):
        last = None
        for m in re.finditer(r"\b%d/\d+ \[[^\r\n]*?loss=([0-9.naif]+)" % step, text):
            last = m.group(1)
        if last is not None:
            out[step] = last
    return out

a, b = losses(sys.argv[1]), losses(sys.argv[2])
common = sorted(set(a) & set(b))
if not common:
    print("PARITY_NO_DATA"); sys.exit(0)
worst = 0.0
bad = False
for s in common:
    try:
        va, vb = float(a[s]), float(b[s])
    except ValueError:
        print(f"PARITY_NAN step {s}: base={a[s]} arm={b[s]}"); bad = True; continue
    if va != va or vb != vb:
        print(f"PARITY_NAN step {s}"); bad = True; continue
    rel = abs(va - vb) / max(abs(va), 1e-9)
    worst = max(worst, rel)
    print(f"step {s}: base={va:.4f} arm={vb:.4f} rel={rel:.4%}")
verdict = "FAIL" if (bad or worst > 0.02) else "PASS"
print(f"PARITY_{verdict} worst_rel={worst:.4%}")
