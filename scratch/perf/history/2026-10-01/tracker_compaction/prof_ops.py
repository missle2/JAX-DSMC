"""Per-op dump of a serial CPU profile: python3 prof_ops.py PROFILE_DIR HLO_DUMP [N]"""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xspace
prof, dump = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
path = glob.glob(os.path.join(dump, "*jit__run*after_optimizations.txt"))[0]
src, text = {}, {}
for line in open(path):
    m = re.match(r"\s*(?:ROOT )?%?([\w.\-]+) = (.*)", line)
    if m:
        s = re.search(r'source_file="([^"]+)" source_line=(\d+)', line)
        src.setdefault(m.group(1), f"{os.path.basename(s.group(1))}:{s.group(2)}" if s else "?")
        text.setdefault(m.group(1), m.group(2)[:150])
p = glob.glob(os.path.join(prof, "plugins", "profile", "*", "*.xplane.pb"))[-1]
buf, planes = xspace.planes(p)
host = next(s for n, s in planes if n == "/host:CPU")
ops, cnt = {}, {}
for _l, _n, _s, dur, st in xspace.plane_events(buf, host, event_stats=True):
    op = st.get("hlo_op")
    if op is None or st.get("hlo_module", "").find("_run") < 0: continue
    if op.startswith(("while", "conditional", "call", "fusion_region")): continue
    ops[op] = ops.get(op, 0) + dur / 1e6 / 100; cnt[op] = cnt.get(op, 0) + 1
flt = sys.argv[4] if len(sys.argv) > 4 else None
tot = 0
for op, ms in sorted(ops.items(), key=lambda t: -t[1]):
    if flt and src.get(op, "?") != flt: continue
    tot += ms
    if N > 0:
        print(f"{ms:7.3f} {cnt[op]/100:6.2f} {op:40s} {src.get(op,'?'):24s} {text.get(op,'')[:110]}")
        N -= 1
print("total", tot)
