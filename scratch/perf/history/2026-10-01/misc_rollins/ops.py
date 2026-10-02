"""Top leaf ops of a profile_forward_cpu.py trace, with op_name for unattributed ops.
usage: python3 ops.py BODY TAG [N]"""
import glob, os, re, sys
sys.path.insert(0, "/home/aakash/jax_dsmc/wt_misc/scratch/perf")
import xspace
body, tag = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
dump = f"/tmp/hlo_cpu_{body}{tag}"
path = glob.glob(os.path.join(dump, "*jit__run*after_optimizations.txt"))[0]
meta = {}
for line in open(path):
    m = re.match(r"\s*(?:ROOT )?%?([\w.\-]+) = (\S+) ", line)
    if m:
        a = re.search(r'op_name="([^"]+)"', line)
        b = re.search(r'source_file="([^"]+)" source_line=(\d+)', line)
        src = os.path.basename(b.group(1)) + ":" + b.group(2) if b else "?"
        meta.setdefault(m.group(1), (m.group(2), a.group(1) if a else "", src))
out = f"/home/aakash/jax_dsmc/wt_misc/scratch/perf/profile/forward_cpu_serial_{body}{tag}"
p = glob.glob(os.path.join(out, "plugins", "profile", "*", "*.xplane.pb"))[-1]
buf, planes = xspace.planes(p)
host = next(s for nm, s in planes if nm == "/host:CPU")
ops = {}
for _l, _n, _s, dur, stats in xspace.plane_events(buf, host, event_stats=True):
    op = stats.get("hlo_op")
    if op is None or "_run" not in stats.get("hlo_module", ""):
        continue
    if op.startswith(("while", "conditional", "call", "fusion_region")):
        continue
    ops[op] = ops.get(op, 0.0) + dur / 1e6 / 100
tot = sum(ops.values())
print(f"total {tot:.3f} ms/step")
for op, ms in sorted(ops.items(), key=lambda t: -t[1])[:n]:
    shp, name, src = meta.get(op, ("", "", "?"))
    print(f"{ms:6.3f} {op:40s} {shp[:28]:28s} {src:22s} {name[-90:]}")
# grouping by op_name keywords
groups = {}
for op, ms in ops.items():
    shp, name, src = meta.get(op, ("", "", "?"))
    g = "threefry/random" if any(k in name for k in ("threefry", "_uniform", "random_bits", "_normal", "_randint", "random_split", "random_fold_in", "_split", "fold_in")) else "other"
    groups[(g, src.split(":")[0])] = groups.get((g, src.split(":")[0]), 0) + ms
print("--- random-related by file")
for k, v in sorted(groups.items(), key=lambda t: -t[1]):
    if k[0] != "other" and v > 0.01:
        print(f"{v:6.3f} {k}")
print("random total", sum(v for k, v in groups.items() if k[0] != "other"))
