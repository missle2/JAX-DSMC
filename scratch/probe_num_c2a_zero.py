"""CLAIM 2(a): majorant violation -> clip saturates -> gradient identically zero, silently."""
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
EPS=1e-30
def safe_log(p,eps=EPS): return jnp.log(jnp.maximum(p,eps))
def acceptance_prob(q,Sigma): return jnp.clip(q/jax.lax.stop_gradient(Sigma),0.0,1.0)
def log_h(q,Sigma,acc):
    p=acceptance_prob(q,Sigma); return jnp.where(acc, safe_log(p), safe_log(1.0-p))

Sigma=1.0
print("d/dq of acceptance_prob(q,Sigma=1.0) and of log_h:")
print(f"{'q':>8} {'p':>10} {'dp/dq':>10} {'logh(acc)':>12} {'dlogh/dq acc':>14} {'logh(rej)':>12} {'dlogh/dq rej':>14}")
for q in [0.5,0.9,0.99,0.999,1.0,1.0000001,1.001,1.5,10.0,1e6]:
    q=float(q)
    p=float(acceptance_prob(q,Sigma))
    dp=float(jax.grad(acceptance_prob)(q,Sigma))
    la=float(log_h(q,Sigma,True));  dla=float(jax.grad(log_h)(q,Sigma,True))
    lr=float(log_h(q,Sigma,False)); dlr=float(jax.grad(log_h)(q,Sigma,False))
    print(f"{q:8.6g} {p:10.6f} {dp:10.3e} {la:12.4e} {dla:14.4e} {lr:12.4e} {dlr:14.4e}")

print("\n Chain to a velocity: q = |vi-vj|^{2(1-omega)}, omega=0.75, Sigma=1.0")
def q_of(v):
    vr2=jnp.sum((v[0]-v[1])**2); return jnp.exp(0.25*jnp.log(vr2))
def L(v, acc): return log_h(q_of(v), 1.0, acc)
for s in [0.5, 0.9, 1.0, 1.2, 3.0]:
    v=jnp.array([[s,0.,0.],[0.,0.,0.]])
    print(f"   |vr|={s:4.2f} q={float(q_of(v)):8.5f} "
          f"|dlogh/dv|_acc={float(jnp.linalg.norm(jax.grad(L)(v,True))):11.4e} "
          f"|dlogh/dv|_rej={float(jnp.linalg.norm(jax.grad(L)(v,False))):11.4e}")
print("\n note: when p==1 exactly, u<p is ALWAYS true (u in [0,1)) so the reject")
print("       branch is unreachable; every violated attempt contributes log h = 0,")
print("       d log h = 0 -- the score term for that attempt vanishes entirely.")

print()
print("="*78); print("CLAIM 2(b): violation counts, live ratchet vs stale max")
print("="*78)
rng=np.random.default_rng(0)
def run(nsteps, npair, heat_rate, rule):
    """q ~ heavy-ish tail, cell slowly heating. returns (violations, mean acceptance)"""
    Sigma = 0.30   # coarse seed, deliberately low
    viol=0; acc=0; tot=0
    for s in range(nsteps):
        scale = 1.0*(1.0+heat_rate)**s
        q = scale*np.abs(rng.normal(size=npair))**1.2   # per-attempt rate
        if rule=='stale':
            S=Sigma
            viol += int((q>S).sum())
            p=np.clip(q/S,0,1); acc+=int((rng.random(npair)<p).sum()); tot+=npair
            Sigma=max(Sigma, q.max())
        else: # live ratchet
            S=Sigma; run_ = 0
            for k in range(npair):
                S=max(S,q[k])
                if q[k]>S: viol+=1
                p=min(q[k]/S,1.0); acc+= int(rng.random()<p); tot+=1
            Sigma=S
    return viol, acc/tot
for hr,label in [(0.0,'isothermal'),(0.02,'heating 2%/step'),(0.05,'heating 5%/step')]:
    for rule in ['stale','live']:
        v,a = run(60, 200, hr, rule)
        print(f"  {label:16s} {rule:6s}: violations={v:6d}/12000 attempts  acceptance={a:.4f}")
