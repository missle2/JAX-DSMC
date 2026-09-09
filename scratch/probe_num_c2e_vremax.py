"""CLAIM 2(e): how much does a monotone (never-reset) vremax inflate, and what does it cost?"""
import numpy as np
rng=np.random.default_rng(3)
def qmax_sample(K, T, omega):
    vr = rng.normal(size=(K,3))*np.sqrt(2*T)
    s = np.linalg.norm(vr,axis=1)
    return (s**2)**(1-omega)*s**0 * s**(0) * ( (s**2)**(0) ) * ( (s)**(1) if omega==0.5 else (s**2)**(1-omega) )
def q_samples(K,T,omega):
    vr = rng.normal(size=(K,3))*np.sqrt(2*T)
    s = np.linalg.norm(vr,axis=1)
    return (s**2)**(1.0-omega)          # sigma_T*vr up to a constant
print("Steady-state over-inflation of a never-reset vremax (isothermal, T=1):")
print(f"{'attempts seen':>15} {'max q':>12} {'/ max q in 1e4':>16}")
base=None
for K in [1e4,1e5,1e6,1e7,1e8]:
    K=int(K); mx=0.0
    n=0
    while n<K:
        b=min(10**6,K-n); mx=max(mx,q_samples(b,1.0,0.5).max()); n+=b
    if base is None: base=mx
    print(f"{K:15d} {mx:12.5f} {mx/base:16.3f}")
print()
print("Transient inflation: cell heats T:1->4 then cools back to 1. omega=0.5")
Sig=0.0; hist=[]
for step in range(300):
    T = 1.0 + 3.0*np.exp(-((step-100)/25.0)**2)      # a heating pulse
    q = q_samples(20000,T,0.5)
    Sig = max(Sig, q.max())
    hist.append((step,T,q.max(),Sig,(q/Sig).mean()))
for s in [0,50,95,100,105,150,200,299]:
    st,T,qm,S,acc = hist[s]
    print(f"  step {st:3d}  T={T:5.2f}  max q this step={qm:7.4f}  Sigma(monotone)={S:7.4f}"
          f"  Sigma/maxq={S/qm:6.3f}  mean acceptance={acc:.4f}")
print()
qm_final = hist[-1][2]; S_final = hist[-1][3]
print(f"  permanent inflation after the transient: Sigma/max_q = {S_final/qm_final:.3f}")
print(f"  => attempts per step n ~ Sigma grow by {S_final/qm_final:.3f}x  (pure compute cost)")
print(f"  => acceptance rate falls by the same factor; collision statistics unchanged")
print()
print("A_max hazard (spec 8.2): n = 0.5 npc(npc-1) Sigma dt fnum/V.  If A_max was sized")
print("from the cold-start Sigma, a monotone Sigma that inflates by X silently drops")
print("attempts once n > A_max.  X measured above = %.2f."%(S_final/qm_final))
