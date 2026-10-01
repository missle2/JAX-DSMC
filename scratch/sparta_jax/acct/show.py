import json, sys, re, collections
d = json.load(open(f"/home/aakash/jax_dsmc/JAX-DSMC-sparta/examples/sparta_jax/results/profile_{sys.argv[1]}.json"))
print("wall", round(d["wall_ms_per_step"], 3), "ops", round(d["op_ms_per_step"], 3), d["ops_per_step"])
for k, v in d["stages"].items():
    print(f"  {k:12s} {v['ms_per_step']:.3f} {v['ops_per_step']:.1f}")
want = sys.argv[2:] 
for st in want:
    agg = collections.defaultdict(lambda: [0, 0.0])
    for s, o, n, t in d["all_ops"]:
        if s == st:
            k = re.sub(r"\.\d+$", "", o); agg[k][0] += n; agg[k][1] += t
    print("==", st)
    for k, (n, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:14]:
        print(f"    {k:36s} {n:7.2f} {t*1000:8.1f}us")
