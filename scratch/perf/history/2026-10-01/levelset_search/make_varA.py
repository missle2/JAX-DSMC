new = open('/home/aakash/jax_dsmc/wt_levelset/jax_dsmc/mesh/levelset.py').read()
old = open('/home/aakash/jax_dsmc/wt_base/jax_dsmc/mesh/levelset.py').read()
a = new.index('# --- cull tables')
b = new.index('# ---------------------------------------------------------------------------\n# Ray-surface hit.')
chunk = new[a:b]
chunk = chunk[:chunk.index('def _near_slots(')] + chunk[chunk.index('def _inside_rows('):]
oi = old.index('def inside(ls: LevelSet')
oe = old.index('# ---------------------------------------------------------------------------\n# Ray-surface hit.')
old = old[:oi] + old[oe:]
i = old.index('def regularised_quotient(')
open('varA/jax_dsmc/mesh/levelset.py', 'w').write(old[:i] + chunk + old[i:])
