"""Stellar-mass validation of galaxies against COSMOS2020 (publication figure).

Compares jaxpz host masses with COSMOS2020 LePhare masses fitted at the spectroscopic redshift,
for DP2 galaxies with a COSMOS spec-z: objects that are not variable (all_varSigClip < 5), not
broad-line AGN and not Gaia quasars, as in plot_photoz.py. Both sides use the spec-z: our masses
are conditioned on it (condition_specz.py ``cond_*`` columns). One panel shows the residual
log M_jaxpz - log M_COSMOS against log M_COSMOS, colored by the width of our 68% mass interval,
with the running median and 16-84% band, and N, the median offset, NMAD scatter and the fraction
off by more than 0.5 dex.

usage: python scripts/plot_mass_validation.py data/sims/val_cosmos-specz_v12_condall.parquet \
           --masses data/cosmos_specz_lephare_masses.parquet --out data/sims/plots/mass_validation_v12.pdf
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
from plot_photoz import match_masses, split  # noqa: E402

MASS_RANGE = (7.5, 12.0)
RESID_RANGE = (-1.5, 1.5)
OUTLIER_DEX = 0.5


def running(x, y, edges, min_n=20):
    """Median and 16/84% of y in bins of x (bins with fewer than min_n objects are NaN)."""
    c = 0.5 * (edges[1:] + edges[:-1])
    q = np.full((3, len(c)), np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        s = y[(x >= lo) & (x < hi)]
        if len(s) >= min_n:
            q[:, i] = np.percentile(s, [16, 50, 84])
    return c, q


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("table", help="condition_specz.py output for the COSMOS spec-z sample (all objects)")
    p.add_argument("--masses", default="data/cosmos_specz_lephare_masses.parquet")
    p.add_argument("--out", default="data/sims/plots/mass_validation.pdf")
    args = p.parse_args(argv)
    d = split("cosmos-specz", pd.read_parquet(args.table))["galaxies"]
    d = match_masses(d, pd.read_parquet(args.masses))
    d = d[~d["is_bl_agn"].astype(bool)]
    ref = d["logM_ref"].to_numpy(float)
    ours = d["cond_log_stellar_mass_med"].to_numpy(float)
    width = (d["cond_log_stellar_mass_hi"] - d["cond_log_stellar_mass_lo"]).to_numpy(float)
    ok = np.isfinite(ref) & np.isfinite(ours)
    ref, resid, width = ref[ok], (ours - ref)[ok], width[ok]
    off = np.median(resid)
    nmad = 1.4826 * np.median(np.abs(resid - off))
    out_frac = np.mean(np.abs(resid) > OUTLIER_DEX)
    with use_style():
        fig, ax = plt.subplots(figsize=(3.5, 2.9))
        order = np.argsort(-width)  # best constrained on top
        art = ax.scatter(ref[order], resid[order], c=width[order], s=1.2, lw=0, cmap="viridis_r", vmin=0.0, vmax=1.5,
                         rasterized=True)
        c, q = running(ref, resid, np.arange(MASS_RANGE[0], MASS_RANGE[1] + 0.01, 0.25))
        ax.fill_between(c, q[0], q[2], color="k", alpha=0.15, lw=0)
        ax.plot(c, q[1], color="k", lw=1.2)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.text(0.04, 0.96, f"N = {len(ref):,}\noffset {off:+.2f} dex, NMAD {nmad:.2f} dex\n"
                f"|Δ| > {OUTLIER_DEX:g} dex: {100 * out_frac:.1f}%", transform=ax.transAxes, va="top", ha="left",
                fontsize=7, linespacing=1.2, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
        ax.set_xlim(*MASS_RANGE)
        ax.set_ylim(*RESID_RANGE)
        ax.set_xticks([8, 9, 10, 11, 12])
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=8)
        ax.set_xlabel(r"$\log M_\star^{\rm COSMOS2020}$", fontsize=9)
        ax.set_ylabel(r"$\log M_\star^{\rm jaxpz} - \log M_\star^{\rm COSMOS2020}$", fontsize=9)
        cax = ax.inset_axes((0.06, 0.1, 0.42, 0.045))
        cb = fig.colorbar(art, cax=cax, orientation="horizontal")
        cb.set_label(r"68% $\log M_\star$ interval (dex)", fontsize=6)
        cb.ax.xaxis.set_label_position("top")
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_linewidth(0.6)
        fig.tight_layout()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(f"{out}: {len(ref)} galaxies, offset {off:+.3f}, NMAD {nmad:.3f}, |Δ| > {OUTLIER_DEX:g}: {100 * out_frac:.1f}%, "
          f"median 68% width {np.median(width):.2f} dex")


if __name__ == "__main__":
    main()
