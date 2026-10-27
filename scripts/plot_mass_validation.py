"""Stellar-mass validation against COSMOS2020 in bins of AGN luminosity (publication figure).

Compares jaxpz host masses with COSMOS2020 LePhare masses fitted at the spectroscopic redshift,
for DP2 objects with a COSMOS spec-z. Both sides use the spec-z: our masses and AGN luminosities
are conditioned on it (condition_specz.py ``cond_*`` columns). One panel per bin of our
posterior-median AGN luminosity log lambda L_5100 shows the residual log M_jaxpz - log M_COSMOS
against log M_COSMOS, colored by the width of our 68% mass interval, with the running median and
16-84% band, and N, the median offset, NMAD scatter and the fraction off by more than 0.5 dex, for
all objects and for those whose 68% interval is narrower than --constrained dex.

LePhare fits galaxy templates only, so its masses for AGN-dominated objects are not a clean truth;
where our interval is wide the mass is set mostly by the prior.

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
from plot_photoz import match_masses  # noqa: E402

L_EDGES = [-np.inf, 42.0, 43.0, 44.0, np.inf]
MASS_RANGE = (7.5, 12.0)
RESID_RANGE = (-1.6, 1.9)
OUTLIER_DEX = 0.5


def bin_label(lo: float, hi: float) -> str:
    q = r"\log \lambda L_{5100}"
    if not np.isfinite(lo):
        return rf"${q} < {hi:g}$"
    if not np.isfinite(hi):
        return rf"${q} > {lo:g}$"
    return rf"${lo:g} < {q} < {hi:g}$"


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
    p.add_argument("--constrained", type=float, default=0.5, help="68%% interval (dex) for 'constrained'")
    args = p.parse_args(argv)
    d = match_masses(pd.read_parquet(args.table), pd.read_parquet(args.masses))
    ref = d["logM_ref"].to_numpy(float)
    ours = d["cond_log_stellar_mass_med"].to_numpy(float)
    lum = d["cond_log_L5100_erg_med"].to_numpy(float)
    width = (d["cond_log_stellar_mass_hi"] - d["cond_log_stellar_mass_lo"]).to_numpy(float)
    ok = np.isfinite(ref) & np.isfinite(ours) & np.isfinite(lum)
    resid = ours - ref
    edges = np.arange(MASS_RANGE[0], MASS_RANGE[1] + 0.01, 0.25)
    with use_style():
        fig, axes = plt.subplots(1, len(L_EDGES) - 1, figsize=(7.2, 2.6), sharex=True, sharey=True)
        for ax, lo, hi in zip(axes, L_EDGES[:-1], L_EDGES[1:]):
            s = ok & (lum >= lo) & (lum < hi)
            x, r, wd = ref[s], resid[s], width[s]
            order = np.argsort(-wd)  # best constrained on top
            art = ax.scatter(x[order], r[order], c=wd[order], s=1.5 if s.sum() > 2000 else 4, lw=0, cmap="viridis_r",
                             vmin=0.0, vmax=2.0, rasterized=True)
            c, q = running(x, r, edges)
            ax.fill_between(c, q[0], q[2], color="k", alpha=0.15, lw=0)
            ax.plot(c, q[1], color="k", lw=1.2)
            ax.axhline(0, color="k", lw=0.6, ls=":")
            off = np.median(r)
            nmad = 1.4826 * np.median(np.abs(r - off))
            good = wd < args.constrained
            rg = r[good]
            line = (f"constrained ({100 * good.mean():.0f}%): {np.median(rg):+.2f}, "
                    f"{1.4826 * np.median(np.abs(rg - np.median(rg))):.2f}" if good.sum() >= 10 else "constrained: none")
            ax.text(0.04, 0.96, f"{bin_label(lo, hi)}\nN = {s.sum():,}: {off:+.2f}, {nmad:.2f}\n{line}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=6.5, linespacing=1.2,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
            ax.set_xlim(*MASS_RANGE)
            ax.set_yticks([-1, 0, 1])
            ax.set_ylim(*RESID_RANGE)
            ax.set_xticks([8, 9, 10, 11, 12])
            ax.tick_params(labelsize=8)
            ax.set_xlabel(r"$\log M_\star^{\rm COSMOS2020}$", fontsize=9)
            print(f"{bin_label(lo, hi)}: N={s.sum()} offset {off:+.3f} NMAD {nmad:.3f} "
                  f"outliers {100 * np.mean(np.abs(r) > OUTLIER_DEX):.1f}%; {line}")
        axes[0].set_ylabel(r"$\log M_\star^{\rm jaxpz} - \log M_\star^{\rm COSMOS2020}$", fontsize=9)
        fig.subplots_adjust(left=0.085, right=0.99, bottom=0.19, top=0.97, wspace=0.06)
        cax = axes[1].inset_axes((0.08, 0.1, 0.42, 0.045))
        cb = fig.colorbar(art, cax=cax, orientation="horizontal")
        cb.set_label(r"68% $\log M_\star$ interval (dex)", fontsize=6)
        cb.ax.xaxis.set_label_position("top")
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_linewidth(0.6)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
