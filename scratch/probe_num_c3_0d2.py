"""CLAIM 3(c): 0-D, fixed N, pair_perm -- are the Oracle tests' FD checks safe? (jit'd)"""
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
EPS=1e-30; N,M = 64,8; SIG=300.0
def safe_pow(x,p):
    ok=x>EPS; return jnp.where(ok,jnp.exp(p*jnp.log(jnp.where(ok,x,1.0))),0.0)
def safe_sqrt(x):
    ok=x>EPS; return jnp.where(ok,jnp.sqrt(jnp.where(ok,x,1.0)),0.0)
def make(seed):
    B=jax.random.PRNGKey(seed); return B, jax.random.normal(jax.random.fold_in(B,1),(N,3))

def run0d(temp, B, Z, omega, Sigma):
    v = Z*jnp.sqrt(temp)*SIG/np.sqrt(300.0)
    def step(carry, s):
        v, = carry
        k=jax.random.fold_in(B,500+s); kp,ka,ks=jax.random.split(k,3)
        perm=jax.random.permutation(kp,N); i,j=perm[0::2],perm[1::2]
        vi,vj=v[i],v[j]
        q=safe_pow(jnp.sum((vi-vj)**2,axis=-1),1.0-omega)
        p=jnp.clip(q/jax.lax.stop_gradient(Sigma),0.0,1.0)
        u=jax.random.uniform(ka,(N//2,)); acc=u<p
        k1,k2=jax.random.split(ks)
        cX=2*jax.random.uniform(k1,(N//2,))-1; sX=safe_sqrt(1-cX*cX)
        ep=2*jnp.pi*jax.random.uniform(k2,(N//2,))
        vr=vi-vj; vrm=safe_sqrt(jnp.sum(vr*vr,axis=-1))
        vrn=vrm[:,None]*jnp.stack([cX,sX*jnp.cos(ep),sX*jnp.sin(ep)],axis=1)
        vcm=0.5*(vi+vj)
        v=v.at[i].set(jnp.where(acc[:,None],vcm+0.5*vrn,vi))
        v=v.at[j].set(jnp.where(acc[:,None],vcm-0.5*vrn,vj))
        return (v,), (perm, acc)
    (v,),(perms,accs)=jax.lax.scan(step,(v,),jnp.arange(M))
    return jnp.mean(v[:,0]**2)/SIG**2, perms, accs

fwd = jax.jit(run0d)
grd = jax.jit(jax.grad(lambda T,B,Z,o,S: run0d(T,B,Z,o,S)[0]))
T0=300.0; B,Z = make(77)
def qmax(Zz,omega):
    v0=np.asarray(Zz)*SIG; i0,j0=np.triu_indices(N,1)
    return float((np.sum((v0[i0]-v0[j0])**2,axis=-1)**(1.0-omega)).max())

for name,omega,Sm in [("Oracle 1: omega=1 (q const), Sigma=q so p==1 always",1.0,None),
                      ("Oracle 2: omega=0.75, Sigma = 2.0 x max q",0.75,2.0),
                      ("Oracle 2': omega=0.75, Sigma = 1.05 x max q",0.75,1.05)]:
    Sigma = 1.0 if omega==1.0 else qmax(Z,omega)*Sm
    _,_,a0 = fwd(T0,B,Z,omega,Sigma)
    gA = float(grd(T0,B,Z,omega,Sigma))
    print("="*100); print(name); print("="*100)
    print("  mean acceptance = %.4f    AD dJ/dT = %.8e"%(float(np.mean(np.asarray(a0))),gA))
    print("  %9s %11s %13s %15s %11s"%('h(rel)','perm flips','accept flips','FD dJ/dT','rel err'))
    for e in np.arange(1,15,1.0):
        h=10.0**(-e)
        Jp,pp,ap=fwd(T0*(1+h),B,Z,omega,Sigma); Jn,pn,an=fwd(T0*(1-h),B,Z,omega,Sigma)
        pf=int(np.sum(np.asarray(pp)!=np.asarray(pn))); af=int(np.sum(np.asarray(ap)!=np.asarray(an)))
        fd=(float(Jp)-float(Jn))/(2*h*T0)
        print("  %9.1e %11d %13d %15.7e %11.3e"%(h,pf,af,fd,abs(fd-gA)/max(abs(gA),1e-300)))
    print()

print("="*100)
print("Accept-bit flip frequency in 0-D with pair_perm, 400 independent seeds,")
print("omega=0.75, Sigma = 2 x max q,  %d steps x %d pairs = %d accept bits per seed"%(M,N//2,M*N//2))
print("="*100)
for e in [2,3,4,5,6,8,10,12]:
    h=10.0**(-e); nf=0; tot=0; sw=0
    for sd in range(400):
        B2,Z2 = make(1000+sd); S=qmax(Z2,0.75)*2.0
        _,_,ap=fwd(300.0*(1+h),B2,Z2,0.75,S); _,_,an=fwd(300.0*(1-h),B2,Z2,0.75,S)
        d=int(np.sum(np.asarray(ap)!=np.asarray(an))); nf+=d; tot+=np.asarray(ap).size; sw+=(d>0)
    print("  h=%8.1e: %5d flips / %d bits = %9.3e per bit;  %3d/400 seeds had >=1 flip"%(h,nf,tot,nf/tot,sw))
