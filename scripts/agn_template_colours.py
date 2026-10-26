"""Is the jaxsedfit AGN SED itself right? Colours of the AGN component alone vs a quasar composite.

The colour tracks in diagnose_agn_colors.py mix the AGN template with the host, dust and the
prior. Here the AGN is isolated: jaxsedfit's variable-AGN band fluxes (no host, both dust
screens at the 1e-3 floor) for a grid of redshifts, with the AGN shape parameters (line, Fe II
and Balmer strengths, optical slope) drawn from jaxpz ``Priors``, against synthetic LSST colours of the Selsing et al. (2016) composite of
luminous, host-free quasars and against point-like Milliquas AGN in DP2.

The composite starts at 1000 A and already contains the Lya forest of its z ~ 1-2 quasars, so
a composite colour is only shown when both bands lie redward of 1250 A in the rest frame;
bluer bands test the IGM treatment, not the template.

usage: python scripts/agn_template_colours.py --ssp tempdata.h5 \
           --composite data/ref/Selsing2015.dat --real data/sims/val_milliquas_v7w.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jaxsedfit.mplstyle import use_style  # noqa: E402

BANDS = "ugrizy"
ZP_NJY = 31.4
C_AA = 2.99792458e18
Z_GRID = np.round(np.arange(0.1, 4.01, 0.1), 2)
Z_EDGES = np.arange(0.05, 4.06, 0.1)
REST_MIN_COMPOSITE = 1250.0


def band_edges(curves, frac=0.01):
    """Wavelength range where each filter transmits more than ``frac`` of its peak."""
    out = []
    for c in curves:
        w, t = np.asarray(c.wave, float), np.asarray(c.transmission, float)
        on = t > frac * t.max()
        out.append((w[on].min(), w[on].max()))
    return out


def composite_colours(path, curves, z_grid):
    """Synthetic AB colours of the composite redshifted to each z (NaN where it does not apply)."""
    wl, fl = np.loadtxt(path, usecols=(0, 1), unpack=True)
    edges = band_edges(curves)
    mags = np.full((len(z_grid), len(curves)), np.nan)
    for i, z in enumerate(z_grid):
        w_obs = wl * (1.0 + z)
        for j, (c, (lo, hi)) in enumerate(zip(curves, edges)):
            if lo < REST_MIN_COMPOSITE * (1.0 + z) or hi > w_obs.max():
                continue
            w, t = np.asarray(c.wave, float), np.asarray(c.transmission, float)
            f = np.interp(w, w_obs, fl)
            # photon-counting AB: <f_nu> = int f_lambda lambda T / int (c / lambda) T
            fnu = np.trapezoid(f * w * t, w) / np.trapezoid(C_AA / w * t, w)
            mags[i, j] = -2.5 * np.log10(fnu)
    return mags[:, :-1] - mags[:, 1:]


def jaxsedfit_agn_colours(ssp, z_grid, n_per_z, log_L5100_erg, calibrated, seed=0):
    """Median and 16/84% colours of the AGN component alone, per redshift."""
    import jaxsedfit
    from jaxpz.generator import MJY_TO_NJY, W_TO_ERG_S, Priors, lsst_simulation_config

    cfg = lsst_simulation_config(ssp, z_range=(0.02, 4.5), calibrated_agn=calibrated)
    z = np.repeat(z_grid, n_per_z)
    n = z.size
    priors = Priors() if calibrated else Priors(broad_lines_median=0.5, pl_slope_mean=-2.1)  # pre-calibration prior
    shape = priors.draw(n, np.random.default_rng(seed))
    keys = ["broad_lines_strength", "pl_slope"] + (["narrow_lines_strength", "feii_norm", "balmer_norm"] if calibrated else [])
    params = {
        "redshift": z,
        "log_stellar_mass": np.full(n, 6.5),
        "log_ebv_gal": np.full(n, np.log(1e-3)),
        "log_ebv_agn": np.full(n, np.log(1e-3)),
        "log_agn_amp": np.full(n, np.log(10.0**log_L5100_erg / W_TO_ERG_S)),
        **{k: shape[k] for k in keys},
    }
    res = jaxsedfit.simulate(cfg, params, seed=seed)
    flux = res.components["variable_agn_fluxes"] * MJY_TO_NJY
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = ZP_NJY - 2.5 * np.log10(flux)
    col = (mag[:, :-1] - mag[:, 1:]).reshape(len(z_grid), n_per_z, -1)
    return np.nanpercentile(col, [16, 50, 84], axis=1)


def milliquas_colours(path, min_snr):
    """Median and 16/84% colours of point-like Milliquas AGN in redshift bins."""
    d = pd.read_parquet(path)
    d = d[d["pointlike"].fillna(False).astype(bool)]
    z = d["ref_z_ref"].to_numpy(float)
    out = np.full((3, len(Z_EDGES) - 1, len(BANDS) - 1), np.nan)
    for j, (b1, b2) in enumerate(zip(BANDS[:-1], BANDS[1:])):
        f1, f2 = d[f"coadd_{b1}"].to_numpy(float), d[f"coadd_{b2}"].to_numpy(float)
        ok = (f1 / d[f"coadd_err_{b1}"] > min_snr).to_numpy() & (f2 / d[f"coadd_err_{b2}"] > min_snr).to_numpy()
        c = -2.5 * np.log10(np.where(ok, f1 / f2, np.nan))
        for i, (a, b) in enumerate(zip(Z_EDGES[:-1], Z_EDGES[1:])):
            s = ok & (z >= a) & (z < b)
            if s.sum() >= 10:
                out[:, i, j] = np.percentile(c[s], [16, 50, 84])
    return out, len(d)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ssp", required=True)
    p.add_argument("--composite", default="data/ref/Selsing2015.dat")
    p.add_argument("--real", default="data/sims/val_milliquas_v7w.parquet")
    p.add_argument("--log-L5100", type=float, default=45.5, help="log lambda L_5100 [erg/s]")
    p.add_argument("--agn-template", choices=("calibrated", "grahsp"), default="calibrated",
                   help="jaxpz's composite-calibrated AGN template or the GRAHSP template with the old prior")
    p.add_argument("--n-per-z", type=int, default=32)
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument("--out", default="data/sims/plots")
    args = p.parse_args(argv)
    from jaxpz.generator import LSST_FILTERS
    from jaxsedfit.filters import load_filter_curves

    curves = load_filter_curves(LSST_FILTERS)
    comp = composite_colours(args.composite, curves, Z_GRID)
    sim = jaxsedfit_agn_colours(args.ssp, Z_GRID, args.n_per_z, args.log_L5100, args.agn_template == "calibrated")
    real, n_real = milliquas_colours(args.real, args.min_snr)
    z_real = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with use_style():
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.4), sharex=True)
        for j, ax in enumerate(axes):
            ax.fill_between(Z_GRID, sim[0, :, j], sim[2, :, j], color="C0", alpha=0.25, lw=0)
            ax.plot(Z_GRID, sim[1, :, j], color="C0", label=f"jaxsedfit AGN only ({args.agn_template})")
            ax.fill_between(z_real, real[0, :, j], real[2, :, j], color="C3", alpha=0.2, lw=0)
            ax.plot(z_real, real[1, :, j], color="C3", label="point-like Milliquas in DP2")
            ax.plot(Z_GRID, comp[:, j], color="k", ls="--", label="Selsing+16 composite")
            ax.set_xlabel(r"$z$")
            ax.set_ylabel(f"{BANDS[j]} - {BANDS[j + 1]}")
            ax.set_xlim(0, 4)
        axes[0].legend(fontsize=9, loc="upper left")
        fig.tight_layout()
        fn = out / f"agn_template_colours_{args.agn_template}.pdf"
        fig.savefig(fn)
        plt.close(fig)
    print(f"{fn}: {n_real} point-like Milliquas AGN; L5100 = 1e{args.log_L5100} erg/s, {args.agn_template} template")
    cols = [f"{a}-{b}" for a, b in zip(BANDS[:-1], BANDS[1:])]
    print("jaxsedfit AGN - composite (median colour, mag):")
    print(pd.DataFrame(sim[1] - comp, index=Z_GRID, columns=cols).round(2).iloc[::2].to_string())
    real_on_grid = np.array([[np.interp(z, z_real, real[1, :, j], left=np.nan, right=np.nan) for j in range(5)] for z in Z_GRID])
    print("jaxsedfit AGN - Milliquas (median colour, mag):")
    print(pd.DataFrame(sim[1] - real_on_grid, index=Z_GRID, columns=cols).round(2).iloc[::2].to_string())
    print("composite - Milliquas (median colour, mag):")
    print(pd.DataFrame(comp - real_on_grid, index=Z_GRID, columns=cols).round(2).iloc[::2].to_string())


if __name__ == "__main__":
    main()
