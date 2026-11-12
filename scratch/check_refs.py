"""Round-1 references and circle: sanity of the fine data, coarse vs fine circle."""
import json, math, sys
import numpy as np
import jax.numpy as jnp
from examples.couette import GAS, NRHO, TEMP, mean_free_path, viscosity_ce
from examples.learned_collision import evaluate as ev, flows, moments
from examples.learned_collision.shear_wave import RESULTS
from jax_dsmc.constants import K_BOLTZ

lam, vth = mean_free_path(), math.sqrt(K_BOLTZ * TEMP / GAS.mass)
nu = viscosity_ce(TEMP) / (NRHO * GAS.mass)
for name in ("shear_a0.1_L32", "shear_a0.5_L32", "shear_a0.1_L128", "shear_a0.5_L128"):
    ref, meta = ev.load_ref(name, "v1")
    k = 2 * math.pi / (meta["length"] * lam)
    g_ce = nu * k**2 * lam / vth
    g = ev.reference_rate(ref, meta, 2.0)
    mean = ref["mean_dx2"]; var = ref["var_dx2"]
    print(f"{name:16s} seeds {meta['seeds']} blocks {meta['blocks']} gamma/gamma_CE {g/g_ce:.4f}  pools {ref['pool_x'].shape}")
for name in ("shock_M3", "shock_M10", "shock_M6"):
    try:
        ref, meta = ev.load_ref(name, "v1")
    except Exception as e:
        print(name, "not loadable:", type(e).__name__); continue
    r = ref["prof_raw"]
    m = np.asarray(moments.central(jnp.asarray(r[:, 0]), jnp.asarray(r[:, 1:4]), jnp.asarray(r[:, 4:13].reshape(-1, 3, 3)), jnp.asarray(r[:, 13:16]), unbiased=False))
    n = r[:, 0] / meta["ppc_up"]; T = (m[:, 4] + m[:, 5] + m[:, 6]) / 3 / meta["vth_ms"]**2
    st = flows.shock_states(meta["mach"]); e = 20
    xs = ref["shock_xs"]
    # maximum-slope thickness in lambda_1
    dndx = np.gradient(n, meta["dx_fine"]); thick = (n[-e:].mean() - n[:e].mean()) / dndx.max()
    print(f"{name:10s} n {n[:e].mean():.3f}->{n[-e:].mean():.3f} (RH {st['ratio']:.3f})  T {T[:e].mean():.3f}->{T[-e:].mean():.2f} (RH {st['temp2']/TEMP:.2f})  "
          f"x_s mean {xs.mean():.2f} std {xs.std():.2f} drift/seed {np.round(xs[-1]-xs[0],2)}  thickness {thick:.2f} lambda1  samples/bin min {ref['prof_samples'].min():.0f}")
fine = json.loads((RESULTS / "circle_ntc_s8_v1.json").read_text())
print("fine circle: slots", fine["slots"], "alive", fine["alive_end"], "warm/sample blocks", fine["warm_blocks"], fine["sample_blocks"], "dt", fine["dt"], "s", round(fine["seconds"]))
Fn, Ft, Fu = (np.array(fine[k]) for k in ("nrho", "temp", "speed"))
fluid = Fn > 0.05
for op in ("ntc", "esfp_recon", "esbgk_recon"):
    c = json.loads((RESULTS / f"circle_{op}_s1_v1.json").read_text())
    n, t, u = (np.array(c[k]) for k in ("nrho", "temp", "speed"))
    shock = fluid & (Fn > 1.5)
    rel = lambda a, b, m: float(np.sqrt(np.mean((a[m] - b[m]) ** 2)) / np.sqrt(np.mean(b[m] ** 2)))
    sl_n = np.array(c["stagnation_n"]); sl_f = np.array(fine["stagnation_n"])
    print(f"{op:12s} drag {c['drag']:.3f}+-{c['drag_se']:.3f} vs {fine['drag']:.3f}+-{fine['drag_se']:.3f} ({100*(c['drag']/fine['drag']-1):+.1f}%)  "
          f"shock-layer L2 n {rel(n,Fn,shock):.3f} T {rel(t,Ft,shock):.3f}  peak stag n {sl_n.max():.2f} vs {sl_f.max():.2f}  alive {c['alive_end']}")
print("coarse dx/lambda hist (log2 bins from -2):", json.loads((RESULTS / "circle_ntc_s1_v1.json").read_text())["log2_dx_lambda_hist"])
