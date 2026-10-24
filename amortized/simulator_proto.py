"""Prototype: jaxsedfit as a batched simulator of LSST photometry (host + AGN).

Run with the jaxcpu conda env (jaxsedfit installed):
    python amortized/simulator_proto.py

Wraps jaxsedfit's NumPyro model (evaluate_sed_model, likelihood off, component fluxes on)
with numpyro.handlers.seed + substitute + trace, so chosen parameters are fixed and the
rest are drawn from jaxsedfit's own priors; jit + vmap over a batch. Reports timing, the
sampled sites, and which latent parameters produce unphysical fluxes.
"""
import time, sys
import numpy as np
import jax, jax.numpy as jnp
from numpyro import handlers
from functools import partial
from jaxsedfit.config import FitConfig, Observation, PhotometryData
from jaxsedfit.benchmark import _build_chimera_agn_config
from jaxsedfit.preload import build_model_context
from jaxsedfit.model import evaluate_sed_model

jax.config.update("jax_enable_x64", True)
FILTERS = [f"rubin.lsst.{b}" for b in "ugrizy"]
zgrid = np.linspace(0.02, 3.0, 300)
cfg = FitConfig(
    observation=Observation(redshift=0.5, redshift_mode="fit", redshift_err=1.0),
    photometry=PhotometryData(filter_names=FILTERS, fluxes=[1e-3] * 6, errors=[1e-4] * 6, is_upper_limit=[False] * 6),
    agn=_build_chimera_agn_config(),
)
cfg.galaxy.dsps_ssp_fn = "/Users/colinburke/research/jaxqsofit/tempdata.h5"
cfg.prior_config.redshift.z_grid = list(zgrid)
cfg.prior_config.redshift.pdf = list(np.ones_like(zgrid))
cfg.likelihood.redshift_projection_n_grid = 512
t = time.time(); ctx = build_model_context(cfg); print(f"context built in {time.time()-t:.1f} s")

model = partial(evaluate_sed_model, ctx, include_components=True, add_likelihood=False, return_state=False)

def sim_one(key, p):
    tr = handlers.trace(handlers.substitute(handlers.seed(model, key), data=p)).get_trace()
    return {k: tr[k]["value"] for k in ("pred_fluxes", "agn_fluxes", "host_fluxes") if k in tr}, \
           {k: v["value"] for k, v in tr.items() if v["type"] == "sample"}

# what sites does the model sample?
_, latents = sim_one(jax.random.PRNGKey(0), {})
print("sampled sites:", sorted(latents))

rng = np.random.default_rng(0)
def draw(n):
    return {
        "redshift": jnp.asarray(rng.uniform(0.05, 2.5, n)),
        "log_stellar_mass": jnp.asarray(rng.uniform(7.0, 11.5, n)),
        "log_agn_amp": jnp.asarray(np.log(10 ** rng.uniform(33.0, 38.5, n))),  # lambda L_lambda(5100 A) in W
    }
for n in (16, 256, 1024):
    p = draw(n); keys = jax.random.split(jax.random.PRNGKey(1), n)
    f = jax.jit(jax.vmap(sim_one))
    t = time.time(); out, lat = f(keys, p); jax.block_until_ready(out["pred_fluxes"]); t1 = time.time() - t
    t = time.time(); out, lat = f(keys, p); jax.block_until_ready(out["pred_fluxes"]); t2 = time.time() - t
    print(f"n={n}: first call (compile+run) {t1:.1f} s, second {t2:.2f} s -> {t2/n*1e3:.2f} ms per simulation")
fl = np.asarray(out["pred_fluxes"])  # mJy
mag = -2.5 * np.log10(fl / 3.631e6)   # AB from mJy (3631 Jy = 3.631e6 mJy)
print("r-band AB mag percentiles 5/50/95:", np.round(np.nanpercentile(mag[:, 2], [5, 50, 95]), 2))
print("AGN fraction in r, percentiles:", np.round(np.nanpercentile(np.asarray(out["agn_fluxes"])[:, 2] / fl[:, 2], [5, 50, 95]), 3))
print("finite fraction:", np.isfinite(fl).all(1).mean())

# which latent parameters drive the near-zero fluxes?
import pandas as pd
L = pd.DataFrame({k: np.asarray(v) for k, v in lat.items() if np.ndim(v) == 1})
L["rmag"] = mag[:, 2]; L["faint"] = L.rmag > 30
cols = [c for c in ("redshift", "log_stellar_mass", "log_ebv_gal", "log_ebv_agn", "log_sfh_age_gyr", "log_sfh_tau_gyr", "log_agn_amp") if c in L]
print("\nmedian latent values, faint (r>30) vs normal:")
print(L.groupby("faint")[cols].median().round(2).to_string())
print("ranges of default priors: ", {c: (round(float(L[c].min()), 2), round(float(L[c].max()), 2)) for c in ("log_ebv_gal", "log_ebv_agn", "log_sfh_age_gyr", "log_sfh_tau_gyr")})
