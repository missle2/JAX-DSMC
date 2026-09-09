import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
D, dt = 1, 0.017
key = jax.random.PRNGKey(3)
M = jax.random.normal(key, (3,3))          # deliberately NON-symmetric linear collision map
P    = lambda v: v[:, :D]                  # R^3 -> R^D, first D components
Pst  = lambda a: jnp.pad(a, ((0,0),(0,3-D)))   # adjoint: zero-pad R^D -> R^3
C    = lambda v: v @ M.T                   # collision map; B = (dC/dv)^T = M^T applied to cotangent
Badj = lambda b: b @ M                     # b -> M^T b  (rows)  == vjp of C

def step_move_then_collide(x, v):          # spec section 12
    x1 = x + dt * P(v)
    v1 = C(v)
    return x1, v1
def step_collide_then_move(x, v):          # [P3]
    v1 = C(v)
    x1 = x + dt * P(v1)
    return x1, v1

N = 5
x = jax.random.normal(jax.random.PRNGKey(1), (N, D))
v = jax.random.normal(jax.random.PRNGKey(2), (N, 3))
alpha1 = jax.random.normal(jax.random.PRNGKey(4), (N, D))   # cotangent on x_{k+1}
beta1  = jax.random.normal(jax.random.PRNGKey(5), (N, 3))   # cotangent on v_{k+1}

for name, f in [("move->collide (spec 12)", step_move_then_collide),
                ("collide->move ([P3])",    step_collide_then_move)]:
    _, vjp = jax.vjp(f, x, v)
    alpha0, beta0 = vjp((alpha1, beta1))
    outside = Badj(beta1) + dt * Pst(alpha1)     # B b + dt P* a
    inside  = Badj(beta1 + dt * Pst(alpha1))     # B (b + dt P* a)
    print(f"{name:26s} alpha_k == alpha_{{k+1}}: {bool(jnp.allclose(alpha0, alpha1))}")
    print(f"{'':26s} beta_k == B b + dt P* a  (OUTSIDE): {bool(jnp.allclose(beta0, outside))}")
    print(f"{'':26s} beta_k == B (b + dt P* a) (INSIDE) : {bool(jnp.allclose(beta0, inside))}")
    print(f"{'':26s} max|beta_k - outside| = {float(jnp.max(jnp.abs(beta0-outside))):.3e}"
          f"   max|beta_k - inside| = {float(jnp.max(jnp.abs(beta0-inside))):.3e}")

# same with a genuinely nonlinear (VSS-like) collision map, B_k = Jacobian^T
def Cn(v): return v @ M.T + 0.3*jnp.tanh(v)*jnp.sum(v*v,axis=1,keepdims=True)
def mtc(x,v): return x + dt*P(v), Cn(v)
def ctm(x,v):
    v1 = Cn(v); return x + dt*P(v1), v1
_, vjpn = jax.vjp(lambda vv: Cn(vv), v); Bn = lambda b: vjpn(b)[0]
print("\nnonlinear C:")
for name, f, ref in [("move->collide", mtc, lambda: Bn(beta1) + dt*Pst(alpha1)),
                     ("collide->move", ctm, lambda: Bn(beta1 + dt*Pst(alpha1)))]:
    _, vj = jax.vjp(f, x, v); a0, b0 = vj((alpha1, beta1))
    o = Bn(beta1) + dt*Pst(alpha1); i = Bn(beta1 + dt*Pst(alpha1))
    print(f"  {name}: OUTSIDE match={bool(jnp.allclose(b0,o))}  INSIDE match={bool(jnp.allclose(b0,i))}"
          f"  (|diff| {float(jnp.max(jnp.abs(b0-o))):.2e} / {float(jnp.max(jnp.abs(b0-i))):.2e})")

# how big is the discrepancy in practice? relative size of the two candidate betas
o = Badj(beta1) + dt*Pst(alpha1); i = Badj(beta1 + dt*Pst(alpha1))
print(f"\nrelative difference between the two forms: {float(jnp.linalg.norm(o-i)/jnp.linalg.norm(o)):.4f}"
      f"   (= O(dt * ||B - I||); dt={dt})")
