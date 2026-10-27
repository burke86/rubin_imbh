"""Stellar-mass validation of galaxies against COSMOS2020 (publication figure).

Compares jaxpz host masses with COSMOS2020 LePhare masses fitted at the spectroscopic redshift,
for DP2 galaxies with a COSMOS spec-z: objects that are not variable (all_varSigClip < 5), not
broad-line AGN and not Gaia quasars, as in plot_photoz.py. Both sides use the spec-z: our masses
are conditioned on it (condition_specz.py ``cond_*`` columns). One panel shows the density of the
residual log M_jaxpz - log M_COSMOS against log M_COSMOS (log-scaled 2D histogram), with N, the
median offset, NMAD scatter and the fraction off by more than 0.5 dex.

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
from matplotlib.colors import LogNorm  # noqa: E402
from jaxsedfit.mplstyle import use_style  # noqa: E402
from plot_photoz import match_masses, split  # noqa: E402

MASS_RANGE = (7.5, 12.0)
RESID_RANGE = (-1.5, 1.5)
OUTLIER_DEX = 0.5


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
    ok = np.isfinite(ref) & np.isfinite(ours)
    ref, resid = ref[ok], (ours - ref)[ok]
    off = np.median(resid)
    nmad = 1.4826 * np.median(np.abs(resid - off))
    out_frac = np.mean(np.abs(resid) > OUTLIER_DEX)
    with use_style():
        fig, ax = plt.subplots(figsize=(3.5, 2.9))
        *_, art = ax.hist2d(ref, resid, bins=(np.linspace(*MASS_RANGE, 91), np.linspace(*RESID_RANGE, 61)),
                            norm=LogNorm(vmin=1), cmap="magma_r", rasterized=True)
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
        cb.set_label("objects per bin", fontsize=6)
        cb.ax.xaxis.set_label_position("top")
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_linewidth(0.6)
        cb.ax.minorticks_off()
        cb.set_ticks([1, 3, 10, 30])
        cb.set_ticklabels(["1", "3", "10", "30"])
        fig.tight_layout()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(f"{out}: {len(ref)} galaxies, offset {off:+.3f}, NMAD {nmad:.3f}, |Δ| > {OUTLIER_DEX:g}: {100 * out_frac:.1f}%")


if __name__ == "__main__":
    main()
