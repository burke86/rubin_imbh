"""Why do AGN photo-z fail at z > 0.4? Two diagnostics, no retraining.

1. Colour tracks: LSST colours vs spectroscopic redshift for real Milliquas AGN measured in
   DP2 (validate_real.py output), against simulated AGN-dominated objects from a training set
   (AGN > ``--agn-frac`` of the r flux, r < ``--r-max``). Medians and 16-84% bands per redshift
   bin, one panel per colour. A systematic offset at some redshift points to the AGN SED
   (emission lines, continuum) rather than to the prior.
2. Prior coverage: the fraction of the weighted training set that is AGN-dominated, per
   redshift bin, under the p(z|m)-only (v6c) and hybrid (v6w) priors.

usage: python scripts/diagnose_agn_colors.py --sims data/sims/train_noll_ddf.parquet \
           --real data/sims/val_milliquas_v6c.parquet --out data/sims/plots
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jaxsedfit.mplstyle import use_style  # noqa: E402

BANDS = "ugrizy"
ZP_NJY = 31.4
Z_EDGES = np.arange(0.0, 4.01, 0.2)


def mags(flux):
    """AB magnitudes from nJy fluxes (NaN for non-positive flux)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, ZP_NJY - 2.5 * np.log10(flux), np.nan)


def binned(z, y, edges=Z_EDGES, min_n=15):
    """Median and 16/84 percentiles of y in redshift bins (NaN where fewer than min_n)."""
    centres, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (z >= a) & (z < b) & np.isfinite(y)
        centres.append(0.5 * (a + b))
        if s.sum() >= min_n:
            q = np.percentile(y[s], [16, 50, 84])
            lo.append(q[0]); med.append(q[1]); hi.append(q[2])
        else:
            lo.append(np.nan); med.append(np.nan); hi.append(np.nan)
    return np.array(centres), np.array(med), np.array(lo), np.array(hi)


def colour_tracks(sims: pd.DataFrame, real: pd.DataFrame, agn_frac: float, r_max: float, min_snr: float, out: Path):
    """Plot real vs simulated AGN colour-redshift tracks and return the per-bin median offsets."""
    frac = np.clip(sims["varagn_r"] / sims["true_r"], 0, 1)
    sim_mag = {b: mags(sims[f"true_{b}"].to_numpy(float)) for b in BANDS}
    sel = (frac > agn_frac).to_numpy() & (sim_mag["r"] < r_max)
    real_mag = {b: mags(real[f"coadd_{b}"].to_numpy(float)) for b in BANDS}
    real_ok = {b: (real[f"coadd_{b}"] / real[f"coadd_err_{b}"] > min_snr).to_numpy() for b in BANDS}
    zs, zr = sims["redshift"].to_numpy(float), real["ref_z_ref"].to_numpy(float)
    offsets = {}
    with use_style():
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.4), sharex=True)
        for ax, (b1, b2) in zip(axes, zip(BANDS[:-1], BANDS[1:])):
            c_sim = (sim_mag[b1] - sim_mag[b2])[sel]
            ok = real_ok[b1] & real_ok[b2]
            c_real = (real_mag[b1] - real_mag[b2])[ok]
            x, m_s, l_s, h_s = binned(zs[sel], c_sim)
            _, m_r, l_r, h_r = binned(zr[ok], c_real)
            ax.fill_between(x, l_s, h_s, color="C0", alpha=0.25, lw=0)
            ax.plot(x, m_s, color="C0", label=f"simulated AGN (f$_{{AGN,r}}$ > {agn_frac})")
            ax.fill_between(x, l_r, h_r, color="C3", alpha=0.25, lw=0)
            ax.plot(x, m_r, color="C3", label="Milliquas in DP2")
            ax.set_xlabel(r"$z$")
            ax.set_ylabel(f"{b1} - {b2}")
            ax.set_xlim(0, 4)
            offsets[f"{b1}-{b2}"] = m_s - m_r
        axes[0].legend(fontsize=9, loc="upper left")
        fig.tight_layout()
        fn = out / "agn_colour_tracks.pdf"
        fig.savefig(fn)
        plt.close(fig)
    return fn, x, offsets, int(sel.sum()), int(np.isfinite(zr).sum())


def prior_coverage(sims: pd.DataFrame, nzm_path: Path, prior_json: Path, scale: float, out: Path):
    """Fraction of the weighted training set that is AGN-dominated, per redshift bin, for both priors."""
    from jaxpz import Priors
    from jaxpz.agn_prior import EmpiricalSARDF
    from jaxpz.nz_prior import EmpiricalNzm, hybrid_weights, nzm_weights

    over = json.loads(prior_json.read_text())
    priors = dataclasses.replace(Priors(), **{k: tuple(v) if isinstance(v, list) else v for k, v in over.items()})
    nzm = EmpiricalNzm.from_cosmos2020(str(nzm_path))
    weights = {"p(z|m) only (v6c)": nzm_weights(sims, nzm),
               "hybrid SARDF (v6w)": hybrid_weights(sims, nzm, erdf=EmpiricalSARDF(scale=scale), priors=priors)}
    frac = np.clip(sims["varagn_r"] / sims["true_r"], 0, 1).to_numpy()
    z = sims["redshift"].to_numpy(float)
    rows = []
    with use_style():
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for (name, w), color in zip(weights.items(), ("C0", "C1")):
            share = []
            for a, b in zip(Z_EDGES[:-1], Z_EDGES[1:]):
                s = (z >= a) & (z < b)
                tot = w[s].sum()
                share.append(w[s & (frac > 0.5)].sum() / tot if tot > 0 else np.nan)
                rows.append((name, f"{a:.1f}-{b:.1f}", share[-1], w[s & (frac > 0.5)].sum() / w.sum()))
            ax.plot(0.5 * (Z_EDGES[:-1] + Z_EDGES[1:]), share, color=color, label=name)
        ax.set_yscale("log")
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"weighted fraction with f$_{\rm AGN,r}$ > 0.5")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fn = out / "agn_prior_coverage.pdf"
        fig.savefig(fn)
        plt.close(fig)
    return fn, pd.DataFrame(rows, columns=["prior", "z", "agn_dominated_fraction_in_bin", "share_of_all_training"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sims", default="data/sims/train_noll_ddf.parquet")
    p.add_argument("--real", default="data/sims/val_milliquas_v6c.parquet")
    p.add_argument("--nzm", default="data/cosmos2020_classic_zr.parquet")
    p.add_argument("--prior-json", default="data/sims/broad_prior.json")
    p.add_argument("--agn-fraction-scale", type=float, default=2.2096)
    p.add_argument("--agn-frac", type=float, default=0.8)
    p.add_argument("--r-max", type=float, default=23.0)
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument("--out", default="data/sims/plots")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sims = pd.read_parquet(args.sims)
    real = pd.read_parquet(args.real)
    fn, x, offsets, n_sim, n_real = colour_tracks(sims, real, args.agn_frac, args.r_max, args.min_snr, out)
    print(f"{fn}: {n_sim} simulated AGN-dominated objects, {n_real} Milliquas AGN")
    print("median colour offset, simulated - real, per redshift bin:")
    print(pd.DataFrame(offsets, index=[f"{c:.1f}" for c in x]).round(2).to_string())
    fn, cov = prior_coverage(sims, Path(args.nzm), Path(args.prior_json), args.agn_fraction_scale, out)
    print(f"{fn}")
    print(cov.pivot(index="z", columns="prior", values="agn_dominated_fraction_in_bin").to_string(float_format=lambda v: f"{v:.2e}"))


if __name__ == "__main__":
    main()
