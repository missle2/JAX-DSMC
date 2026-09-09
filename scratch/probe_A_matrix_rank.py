"""Probe: is [P1] eq. AB_def's "B = A^T = A^-1" literally true for the 6x6 matrix?

Answer: no. A is rank 4 (singular values [1,1,1,1,0,0]), so A^-1 does not exist.
B = A^T is correct and is all the adjoint needs. The involution [P1] describes is a
property of the MAP -- re-applying the collision with sigma and alpha exchanged
recovers the input -- not of the matrix.

Why rank 4: at fixed sigma the output depends on (v,v1) only through v+v1 (3 dof)
and |v-v1| (1 dof). Rotations of the relative-velocity direction are annihilated.

Drove PSEUDOCODE section 17.1. Run: source ~/activate_jax_env.sh
"""

import numpy as np

rng = np.random.default_rng(1)
ok = lambda n, c: print(("PASS " if c else "FAIL ") + n)


def coll(z, s):
    v, v1 = z[:3], z[3:]
    n = np.linalg.norm(v - v1)
    return np.concatenate([0.5 * ((v + v1) + n * s), 0.5 * ((v + v1) - n * s)])


v = rng.normal(size=3)
v1 = rng.normal(size=3)
sig = rng.normal(size=3)
sig /= np.linalg.norm(sig)
al = (v - v1) / np.linalg.norm(v - v1)
I = np.eye(3)
A = 0.5 * np.block(
    [[I + np.outer(sig, al), I - np.outer(sig, al)], [I - np.outer(sig, al), I + np.outer(sig, al)]]
)
B = 0.5 * np.block(
    [[I + np.outer(al, sig), I - np.outer(al, sig)], [I - np.outer(al, sig), I + np.outer(al, sig)]]
)

ok("B == A^T  (paper eq. AB_def)", np.allclose(B, A.T))
print("   ||B@A - I||  =", np.abs(B @ A - np.eye(6)).max())
print("   ||A^-1 - A^T||=", np.abs(np.linalg.inv(A) - A.T).max())
print("   singular values of A:", np.round(np.linalg.svd(A, compute_uv=False), 4))

# Is the MAP an involution? apply collision with sigma, then again with alpha
z = np.concatenate([v, v1])
zp = coll(z, sig)
al2 = zp[:3] - zp[3:]
al2 /= np.linalg.norm(al2)
ok("post-collision direction == sigma", np.allclose(al2, sig))
zpp = coll(zp, al)  # inverse map: same form, sigma -> alpha
ok("MAP is an involution (coll(coll(z,sig),al) == z)", np.allclose(zpp, z))


# True Jacobian of the inverse map (alpha recomputed from its own input)
def inv_map(zp_, al_):
    return coll(zp_, al_)


h = 1e-6
Jinv = np.zeros((6, 6))
for k in range(6):
    e = np.zeros(6)
    e[k] = h
    Jinv[:, k] = (inv_map(zp + e, al) - inv_map(zp - e, al)) / (2 * h)
print("   ||Jinv@A - I|| =", np.abs(Jinv @ A - np.eye(6)).max())
ok("B == Jacobian of inverse map at fixed alpha", np.allclose(B, Jinv, atol=1e-6))
