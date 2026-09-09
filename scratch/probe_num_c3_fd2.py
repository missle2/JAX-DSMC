"""CLAIM 3: decision flips under common random numbers, 1-D spatial DSMC. jit'd."""
import jax, jax.numpy as jnp, numpy as np, functools
jax.config.update("jax_enable_x64", True)
EPS=1e-30
C, NP, M, AMAX = 8, 256, 6, 16
OMEGA=0.75; LO,HI=0.0,1.0; DT=2.0e-4; SIG=300.0; SIGMA=1.0
BASE=jax.random.PRNGKey(2024)
Z =jax.random.normal(jax.random.fold_in(BASE,1),(NP,3))
X0=jax.random.uniform(jax.random.fold_in(BASE,2),(NP,))

def safe_pow(x,p):
    ok=x>EPS; return jnp.where(ok,jnp.exp(p*jnp.log(jnp.where(ok,x,1.0))),0.0)
def safe_sqrt(x):
    ok=x>EPS; return jnp.where(ok,jnp.sqrt(jnp.where(ok,x,1.0)),0.0)

def core(temp, lo, hi, dt):
    v = Z*jnp.sqrt(temp)*SIG/np.sqrt(300.0)
    x = lo+(hi-lo)*X0
    remain = jnp.full((C,),0.5)
    R = {'cell':[], 'counts':[], 'natt':[], 'pi':[], 'pj':[], 'acc':[]}
    for s in range(M):
        x = x+v[:,0]*dt; x = lo+jnp.mod(x-lo,hi-lo)
        dx=(hi-lo)/C
        c = jnp.clip(jnp.floor((x-lo)/dx),0,C-1).astype(jnp.int32)
        order = jnp.argsort(c, stable=True)
        counts = jnp.sum(jax.nn.one_hot(c,C,dtype=jnp.float64),axis=0)
        starts = jnp.concatenate([jnp.zeros(1),jnp.cumsum(counts)[:-1]]).astype(jnp.int32)
        f = 0.5*counts*(counts-1.0)*SIGMA*dt*4.0e4/dx + remain
        n = jnp.floor(f).astype(jnp.int32); remain = f-jnp.floor(f)
        ci = counts.astype(jnp.int32)
        R['cell'].append(c); R['counts'].append(ci); R['natt'].append(n)
        def slot(carry, a):
            v, = carry
            ka=jax.random.fold_in(jax.random.fold_in(BASE,100+s),a)
            k1,k2,k3,k4=jax.random.split(ka,4)
            npc=jnp.maximum(ci,1)
            i=jax.random.randint(k1,(C,),0,npc)
            r=jax.random.randint(k2,(C,),0,jnp.maximum(npc-1,1))
            j=(i+1+r)%npc
            pi=order[jnp.clip(starts+i,0,NP-1)]; pj=order[jnp.clip(starts+j,0,NP-1)]
            active=(a<n)&(ci>=2)
            vi,vj=v[pi],v[pj]
            q=safe_pow(jnp.sum((vi-vj)**2,axis=-1),1.0-OMEGA)*1.0e-2
            p=jnp.clip(q/jax.lax.stop_gradient(SIGMA),0.0,1.0)
            u=jax.random.uniform(k3,(C,)); acc=active&(u<p)
            kk1,kk2=jax.random.split(k4)
            cX=2*jax.random.uniform(kk1,(C,))-1; sX=safe_sqrt(1-cX*cX)
            ep=2*jnp.pi*jax.random.uniform(kk2,(C,))
            vr=vi-vj; vrm=safe_sqrt(jnp.sum(vr*vr,axis=-1))
            vrn=vrm[:,None]*jnp.stack([cX,sX*jnp.cos(ep),sX*jnp.sin(ep)],axis=1)
            vcm=0.5*(vi+vj)
            v=v.at[pi].set(jnp.where(acc[:,None],vcm+0.5*vrn,vi))
            v=v.at[pj].set(jnp.where(acc[:,None],vcm-0.5*vrn,vj))
            return (v,), (jnp.where(active,pi,-1), jnp.where(active,pj,-1), acc)
        (v,), (pis,pjs,accs) = jax.lax.scan(slot,(v,),jnp.arange(AMAX))
        R['pi'].append(pis); R['pj'].append(pjs); R['acc'].append(accs)
    J = jnp.mean(v[:,0]**2)/SIG**2
    return J, {k:jnp.stack(vv) for k,vv in R.items()}

fwd  = jax.jit(core)
Jfun = jax.jit(lambda T: core(T,LO,HI,DT)[0])
gAD  = float(jax.jit(jax.grad(lambda T: core(T,LO,HI,DT)[0]))(300.0))

def flips(a,b): return {k:int(np.sum(np.asarray(a[k])!=np.asarray(b[k]))) for k in a}
T0=300.0
print("="*104)
print(f"CLAIM 3(a-b): 1-D spatial, C={C} cells, N={NP}, M={M} steps, A_max={AMAX}. Perturb `temp` relatively by h.")
print("   Both evaluations use the SAME base_key (common random numbers).")
print("="*104)
J0,_=fwd(T0,LO,HI,DT)
print(f"J(T0)={float(J0):.10f}   AD dJ/dT = {gAD:.8e}")
print(f"{'h(rel)':>9} {'cell':>6} {'counts':>7} {'n_att':>6} {'pair_i':>7} {'pair_j':>7} {'accept':>7} {'anyflip':>8} "
      f"{'FD dJ/dT':>14} {'rel err vs AD':>14}")
for e in np.arange(1,15,0.5):
    h=10.0**(-e)
    Jp,dp=fwd(T0*(1+h),LO,HI,DT); Jn,dn=fwd(T0*(1-h),LO,HI,DT)
    fl=flips(dp,dn); tot=sum(fl.values())
    fd=(float(Jp)-float(Jn))/(2*h*T0)
    print(f"{h:9.2e} {fl['cell']:6d} {fl['counts']:7d} {fl['natt']:6d} {fl['pi']:7d} {fl['pj']:7d} {fl['acc']:7d} "
          f"{'YES' if tot else 'none':>8} {fd:14.6e} {abs(fd-gAD)/abs(gAD):14.3e}")

print()
print("="*104)
print("Same sweep for other parameters (flip totals only, plus FD rel err vs AD)")
print("="*104)
for pname, base, fn in [("lo (grid origin)", LO, lambda th: (300.0, th, HI, DT)),
                        ("hi (grid extent)", HI, lambda th: (300.0, LO, th, DT)),
                        ("dt",               DT, lambda th: (300.0, LO, HI, th))]:
    gA = float(jax.jit(jax.grad(lambda th: core(*fn(th))[0]))(base))
    print(f"\n  parameter {pname}, base={base}, AD grad={gA:.6e}")
    print(f"  {'h(abs)':>10} {'cell':>6} {'counts':>7} {'n_att':>6} {'pair_i':>7} {'pair_j':>7} {'accept':>7} {'FD':>13} {'rel err':>11}")
    for e in np.arange(2,15,1.0):
        h=abs(base)*10.0**(-e) if base!=0 else 10.0**(-e)
        if h==0: continue
        Jp,dp=fwd(*fn(base+h)); Jn,dn=fwd(*fn(base-h))
        fl=flips(dp,dn); fd=(float(Jp)-float(Jn))/(2*h)
        print(f"  {h:10.2e} {fl['cell']:6d} {fl['counts']:7d} {fl['natt']:6d} {fl['pi']:7d} {fl['pj']:7d} {fl['acc']:7d} "
              f"{fd:13.5e} {abs(fd-gA)/max(abs(gA),1e-300):11.3e}")
