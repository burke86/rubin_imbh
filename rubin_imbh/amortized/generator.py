"""Training-set generator: population priors -> jaxsedfit photometry -> DP2-like observables.

Two stages:

* :func:`simulate_population` (needs jaxsedfit >= the ``simulate`` API): draws population
  priors, simulates LSST ugrizy total and variable-AGN fluxes, applies the magnitude limit.
* :func:`observe` (NumPy only): gives each simulated object the coadd noise, cadence and
  nightly errors of a real library object of similar brightness, simulates AGN variability
  (one damped random walk shared by all bands), and reduces the light curves with the
  pipeline's :func:`rubin_imbh.varstats.lightcurve_stats`, so simulated variability summaries
  are computed exactly as for real objects.

Units: fluxes in nJy (jaxsedfit's mJy x 1e6); ``log_agn_amp`` is ln(lambda L_lambda(5100 A) / W)
as in jaxsedfit; ``log_L5100_erg`` is log10 of the same in erg/s.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..varstats import BANDS, Columns, Cuts, lightcurve_stats
from .library import match_library
from .variability import drw_unit_batch, fvar_by_band

LSST_FILTERS = [f"rubin.lsst.{b}" for b in BANDS]
LSST_EFF_WAVE = np.array([3671.0, 4827.0, 6223.0, 7546.0, 8691.0, 9712.0])  # Angstrom, approximate
MJY_TO_NJY = 1.0e6
W_TO_ERG_S = 1.0e7


@dataclass
class Priors:
    """Population priors for the training set (all ranges inclusive)."""

    z_range: tuple[float, float] = (0.02, 4.0)  # ~5% of r < 25.5 galaxies lie at z > 3 (COSMOS2020)
    log_mass_range: tuple[float, float] = (6.5, 12.0)
    ebv_gal_sigma: float = 0.3          # half-normal
    ebv_agn_sigma: float = 0.5          # half-normal
    ebv_min: float = 1.0e-3
    agn_fraction: float = 0.5           # the rest have no AGN
    log_L5100_erg_range: tuple[float, float] = (40.0, 46.0)  # log-uniform, erg/s
    # the no-AGN half gets a negligible, continuous luminosity (normalizing flows handle a
    # point mass badly); posteriors below ~10^40 erg/s mean "no detectable AGN"
    no_agn_log_L5100_erg_range: tuple[float, float] = (36.0, 40.0)
    # asymptotic fractional rms (SF_inf / sqrt(2)) at rest-frame 5100 A; the rms seen over a
    # DP2-length baseline is lower for tau_obs comparable to or longer than the baseline
    fvar_median: float = 0.15
    fvar_sigma_dex: float = 0.4
    fvar_alpha_range: tuple[float, float] = (0.3, 0.7)  # f_var ~ (lambda_rest/5100)^-alpha
    tau_rest_median_days: float = 200.0
    tau_rest_sigma_dex: float = 0.3
    r_mag_limit: float = 25.5           # applied to the true total r-band flux

    def draw(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        has_agn = rng.random(n) < self.agn_fraction
        log_L = np.where(has_agn, rng.uniform(*self.log_L5100_erg_range, n), rng.uniform(*self.no_agn_log_L5100_erg_range, n))
        ebv_gal = np.maximum(np.abs(rng.normal(0, self.ebv_gal_sigma, n)), self.ebv_min)
        ebv_agn = np.maximum(np.abs(rng.normal(0, self.ebv_agn_sigma, n)), self.ebv_min)
        return {
            "redshift": rng.uniform(*self.z_range, n),
            "log_stellar_mass": rng.uniform(*self.log_mass_range, n),
            "log_ebv_gal": np.log(ebv_gal),
            "log_ebv_agn": np.log(ebv_agn),
            "log_agn_amp": np.log(10.0**log_L / W_TO_ERG_S),
            "_has_agn": has_agn,
            "_log_L5100_erg": log_L,
        }

    def draw_variability(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        return {
            "fvar0": 10 ** (np.log10(self.fvar_median) + rng.normal(0, self.fvar_sigma_dex, n)),
            "fvar_alpha": rng.uniform(*self.fvar_alpha_range, n),
            "tau_rest_days": 10 ** (np.log10(self.tau_rest_median_days) + rng.normal(0, self.tau_rest_sigma_dex, n)),
        }


def lsst_simulation_config(dsps_ssp_fn: str, z_range=(0.02, 4.0), n_redshift_grid: int = 640):
    """jaxsedfit simulation config for LSST ugrizy with a sampled redshift over ``z_range``."""
    import jaxsedfit
    from jaxsedfit.benchmark import _build_chimera_agn_config  # AGN line / Fe II templates

    return jaxsedfit.simulation_config(LSST_FILTERS, redshift_range=z_range, n_redshift_grid=n_redshift_grid,
                                       dsps_ssp_fn=dsps_ssp_fn, agn=_build_chimera_agn_config())


def simulate_population(n: int, priors: Priors, rng: np.random.Generator, config, *, seed: int = 0,
                        batch_size: int = 4096, max_rounds: int = 50) -> pd.DataFrame:
    """Draw priors and simulate until ``n`` objects pass the r-band magnitude limit.

    Returns a DataFrame with the drawn parameters (fixed and jaxsedfit-drawn), true total
    fluxes ``true_<b>`` and variable-AGN fluxes ``varagn_<b>`` in nJy.
    """
    import jaxsedfit
    from jaxsedfit.preload import build_model_context

    ctx = build_model_context(config)
    flux_lim = 10 ** ((31.4 - priors.r_mag_limit) / 2.5)
    out, have = [], 0
    for k in range(max_rounds):
        m = max(2 * (n - have), 1024)
        p = priors.draw(m, rng)
        fixed = {key: v for key, v in p.items() if not key.startswith("_")}
        res = jaxsedfit.simulate(config, fixed, seed=seed + k, batch_size=batch_size, context=ctx)
        df = pd.DataFrame({key: v for key, v in res.params.items() if np.ndim(v) == 1})
        df["has_agn"] = p["_has_agn"]
        df["log_L5100_erg"] = p["_log_L5100_erg"]
        for j, b in enumerate(BANDS):
            df[f"true_{b}"] = res.fluxes[:, j] * MJY_TO_NJY
            df[f"varagn_{b}"] = res.components["variable_agn_fluxes"][:, j] * MJY_TO_NJY
        keep = np.isfinite(df[[f"true_{b}" for b in BANDS]]).all(axis=1) & (df["true_r"] > flux_lim)
        out.append(df[keep])
        have += int(keep.sum())
        if have >= n:
            break
    return pd.concat(out, ignore_index=True).iloc[:n].reset_index(drop=True)


@dataclass
class ObserveConfig:
    layer: str | None = "wfd"
    sys_frac: dict = field(default_factory=lambda: {b: 0.0075 for b in "riz"})
    err_scale: float = 1.0
    var_bands: tuple[str, ...] = tuple("griz")  # bands whose variability summaries are kept
    match_k: int = 25
    # per-object, per-band multiplicative zero-point / calibration offsets (rms), so the network
    # does not over-trust colours; applied to the coadd fluxes
    zp_sigma: float = 0.01


def observe(sim: pd.DataFrame, library: pd.DataFrame, priors: Priors, rng: np.random.Generator,
            cfg: ObserveConfig = ObserveConfig()) -> pd.DataFrame:
    """Add DP2-like observables to simulated objects (columns ``true_<b>``, ``varagn_<b>``, ``redshift``).

    Adds: ``coadd_<b>``, ``coadd_err_<b>`` (nJy); per variability band the pipeline statistics
    ``<b>_nEpoch``, ``<b>_varSigClip``, ``<b>_rmsErr``, ``<b>_xsAbs`` (excess variance in nJy^2)
    and the combined ``all_varSigClip``, ``all_dof``; the variability truth ``fvar0``,
    ``fvar_alpha``, ``tau_rest_days``; and ``lib_index`` (the matched library row).
    """
    n = len(sim)
    out = sim.copy()
    idx = match_library(library, sim["true_r"].to_numpy(), rng, cfg.layer, cfg.match_k)
    out["lib_index"] = idx
    lib = library.iloc[idx].reset_index(drop=True)

    # coadd photometry, with calibration offsets
    for b in BANDS:
        err = lib[f"{b}_coadd_err"].to_numpy(float)
        zp = 1.0 + cfg.zp_sigma * rng.standard_normal(n)
        out[f"coadd_{b}"] = out[f"true_{b}"].to_numpy() * zp + rng.normal(0.0, err)
        out[f"coadd_err_{b}"] = err

    # variability: shared unit DRW at the union of each object's nights, scaled per band
    var = priors.draw_variability(n, rng)
    for k, v in var.items():
        out[k] = v
    z = out["redshift"].to_numpy()
    rest = LSST_EFF_WAVE[None, :] / (1.0 + z[:, None])
    rms = fvar_by_band(var["fvar0"], var["fvar_alpha"], rest) * np.abs(out[[f"varagn_{b}" for b in BANDS]].to_numpy())
    vb = [BANDS.index(b) for b in cfg.var_bands]
    per_band_t = [[np.asarray(t, float) for t in lib[f"{BANDS[j]}_night_mjd"]] for j in vb]
    per_band_e = [[np.asarray(e, float) for e in lib[f"{BANDS[j]}_night_err"]] for j in vb]
    lens = np.array([[len(per_band_t[q][i]) for q in range(len(vb))] for i in range(n)])
    L = int(lens.sum(axis=1).max()) if n else 0
    T = np.full((n, max(L, 1)), np.nan)
    B = np.full((n, max(L, 1)), -1, dtype=int)
    for i in range(n):
        ts = np.concatenate([per_band_t[q][i] for q in range(len(vb))]) if L else np.zeros(0)
        bs = np.concatenate([np.full(lens[i, q], q) for q in range(len(vb))]) if L else np.zeros(0, int)
        o = np.argsort(ts)
        T[i, : len(ts)] = ts[o]
        B[i, : len(ts)] = bs[o]
    X = drw_unit_batch(T, var["tau_rest_days"] * (1.0 + z), rng)

    rows_obj, rows_band, rows_f, rows_e, rows_s = [], [], [], [], []
    for i in range(n):
        m = B[i] >= 0
        for q, j in enumerate(vb):
            sel = m & (B[i] == q)
            if not sel.any():
                continue
            # library nights are time-ordered within a band, and the union sort keeps that
            # order, so the band's errors line up with its selected epochs
            e = per_band_e[q][i]
            f = rms[i, j] * X[i, sel] + rng.normal(0.0, e[: sel.sum()])
            rows_obj.append(np.full(sel.sum(), i))
            rows_band.append(np.full(sel.sum(), BANDS[j]))
            rows_f.append(f)
            rows_e.append(e[: sel.sum()])
            rows_s.append(np.full(sel.sum(), out[f"true_{BANDS[j]}"].iat[i]))
    flat = pd.DataFrame({
        "band": np.concatenate(rows_band) if rows_band else np.zeros(0, str),
        "psfDiffFlux": np.concatenate(rows_f) if rows_f else np.zeros(0),
        "psfDiffFluxErr": np.concatenate(rows_e) if rows_e else np.zeros(0),
        "psfFlux": np.concatenate(rows_s) if rows_s else np.zeros(0),
    })
    obj = np.concatenate(rows_obj) if rows_obj else np.zeros(0, int)
    cuts = Cuts(err_scale=cfg.err_scale, sys_frac=cfg.sys_frac, nightly=False,
                select_bands=tuple(b for b in "riz" if b in cfg.var_bands),
                combo_bands=tuple(b for b in "riz" if b in cfg.var_bands))
    st = lightcurve_stats(obj, flat, n, Columns(), cuts, bands=tuple(cfg.var_bands))
    for b in cfg.var_bands:
        out[f"{b}_nEpoch"] = st[f"{b}_nEpoch"].to_numpy()
        out[f"{b}_varSigClip"] = st[f"{b}_varSigClip"].to_numpy()
        out[f"{b}_rmsErr"] = st[f"{b}_rmsErr"].to_numpy()
        out[f"{b}_xsAbs"] = (st[f"{b}_excessVar"] * st[f"{b}_medFlux"] ** 2).to_numpy()
    out["all_varSigClip"] = st["all_varSigClip"].to_numpy()
    out["all_dof"] = st["all_dof"].to_numpy()
    out["true_rms_r"] = rms[:, BANDS.index("r")]
    return out
