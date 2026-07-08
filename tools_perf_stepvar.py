"""Per-step wall times from tqdm cumulative elapsed -> mean/std/p95, steps LO-HI."""
import re, sys
lo, hi = int(sys.argv[2]), int(sys.argv[3])
text = open(sys.argv[1], errors="ignore").read()
el = {}
for m in re.finditer(r"\b(\d+)/\d+ \[(\d+):(\d+)(?::(\d+))?<", text):
    st = int(m.group(1))
    g = m.groups()
    t = int(g[1])*3600 + int(g[2])*60 + int(g[3]) if g[3] else int(g[1])*60 + int(g[2])
    el[st] = t
xs = [el[i+1] - el[i] for i in range(lo, hi) if i in el and i+1 in el]
if not xs:
    print("NO_DATA"); raise SystemExit
import statistics as st_
xs.sort()
print(f"steps {lo}-{hi}: n={len(xs)} mean={st_.mean(xs):.2f}s std={st_.pstdev(xs):.2f} p95={xs[int(len(xs)*0.95)]}s max={xs[-1]}s")
