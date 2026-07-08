import re, sys
# parse tqdm cumulative elapsed at two step marks -> s/it between them
lo, hi = int(sys.argv[2]), int(sys.argv[3])
def elapsed_at(step, text):
    pat = re.compile(r"\b%d/320 \[(\d+):(\d+)(?::(\d+))?<" % step)
    m = None
    for m0 in pat.finditer(text):
        m = m0
    if not m: return None
    g = m.groups()
    if g[2] is not None:
        return int(g[0])*3600 + int(g[1])*60 + int(g[2])
    return int(g[0])*60 + int(g[1])
text = open(sys.argv[1], errors='ignore').read()
a, b = elapsed_at(lo, text), elapsed_at(hi, text)
if a is None or b is None:
    print("PARSE_FAIL", a, b)
else:
    print("SPEED steps %d-%d: %.3f s/it" % (lo, hi, (b - a) / (hi - lo)))
