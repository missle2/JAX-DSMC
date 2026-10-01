"""Per-op table for one stage: name, calls/step, us/call, output shape, fused root kind."""
import glob, json, re, sys
tag, stage = sys.argv[1], sys.argv[2]
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 60
d = json.load(open(f"/home/aakash/jax_dsmc/JAX-DSMC-sparta/examples/sparta_jax/results/profile_{tag}.json"))
f = sorted(glob.glob(f"/tmp/sj_hlo_{tag}/*jit_run*after_optimizations.txt"))[-1]
shape, root = {}, {}
comp = {}
cur = None
for line in open(f):
    m = re.match(r"^%?([\w.\-]+) .*\{$", line)
    if line.startswith("%") or line.startswith("ENTRY"):
        cur = line.split()[0].lstrip("%")
    m = re.match(r"\s*(ROOT )?%([\w.\-]+) = (\S+) (\w[\w\-]*)\(", line)
    if m:
        shape[m.group(2)] = m.group(3)
        if m.group(1):
            root[cur] = m.group(4)
        c = re.search(r"calls=%([\w.\-]+)", line)
        if c:
            comp[m.group(2)] = c.group(1)
rows = [r for r in d["all_ops"] if r[0] == stage]
tot = sum(r[3] for r in rows)
print(f"{stage}: {len(rows)} distinct ops, {sum(r[2] for r in rows):.1f} calls/step, {tot*1e3:.1f} us/step")
for s, o, n, t in rows[:nmax]:
    rk = root.get(comp.get(o, ""), "")
    print(f"  {o:40s} {n:6.2f} {t*1e3/n if n else 0:8.2f} us/call  {shape.get(o,'?'):28s} {rk}")
