"""Host stellar mass vs redshift for variable DP2 AGN, from jaxpz posterior summaries.

Takes validate_real.py outputs (objects with a reference redshift) or jaxpz predict outputs (no
reference redshift) and keeps variable objects (all_varSigClip >= --min-sig), de-duplicated by
objectId. Left panel: host mass against the reference (spectroscopic) redshift where there is one,
conditioned on it if the table has ``cond_log_stellar_mass_*`` columns (the posterior given the
data and z = z_spec), else the redshift-marginal posterior; right panel: the marginal posterior
against the photo-z posterior median, as for a catalogue without spectra. Points are coloured by the posterior-median AGN luminosity log lambda L_5100; the medians
and 16-84% spread of the host mass are shown in redshift bins, for all objects and for those whose
mass is constrained (68% interval narrower than --mass-width dex).

Marginal masses of objects with a catastrophic photo-z follow the wrong redshift.

usage: python scripts/plot_mass_redshift.py data/sims/val_milliquas_v10.parquet \
           data/sims/val_cosmos-specz_v10.parquet data/sims/val_cdfs-xray_v10.parquet \
           --out data/sims/plots/mass_redshift_variable_agn_v10.pdf
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

Z_EDGES = np.arange(0.0, 4.51, 0.25)
MASS_RANGE = (7.0, 12.5)


def load(paths, min_sig: float) -> pd.DataFrame:
    """Variable objects from the posterior tables, one row per objectId (first file wins)."""
    parts = []
    for p in paths:
        d = pd.read_parquet(p)
        d = d[d["all_varSigClip"] >= min_sig].copy()
        d["source"] = Path(p).stem
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    return d.drop_duplicates("objectId") if "objectId" in d else d


def binned(ax, z, m, color, label, ls="-"):
    """Median and 16-84% of m in redshift bins with at least 10 objects."""
    c, med, lo, hi = [], [], [], []
    for a, b in zip(Z_EDGES[:-1], Z_EDGES[1:]):
        s = (z >= a) & (z < b) & np.isfinite(m)
        if s.sum() >= 10:
            c.append(0.5 * (a + b))
            q = np.percentile(m[s], [16, 50, 84])
            lo.append(q[0]); med.append(q[1]); hi.append(q[2])
    if c:
        ax.plot(c, med, color=color, ls=ls, lw=2, label=label)
        ax.fill_between(c, lo, hi, color=color, alpha=0.15, lw=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tables", nargs="+", help="validate_real.py or jaxpz predict outputs")
    p.add_argument("--min-sig", type=float, default=5.0, help="variability selection: all_varSigClip >= this")
    p.add_argument("--mass-width", type=float, default=0.5, help="68% mass interval (dex) for 'constrained'")
    p.add_argument("--out", default="data/sims/plots/mass_redshift_variable_agn.pdf")
    args = p.parse_args(argv)
    d = load(args.tables, args.min_sig)
    x_axes = [("post_redshift_med", r"photo-$z$ (posterior median)", "post")]
    if "ref_z_ref" in d and np.isfinite(d["ref_z_ref"]).any():
        x_axes.insert(0, ("ref_z_ref", r"$z_{\rm spec}$", "cond" if "cond_log_stellar_mass_med" in d else "post"))
    with use_style():
        fig, axes = plt.subplots(1, len(x_axes), figsize=(6.4 * len(x_axes) + 1.2, 5.6), sharey=True, squeeze=False)
        for ax, (col, label, pre) in zip(axes[0], x_axes):
            m = d[f"{pre}_log_stellar_mass_med"].to_numpy(float)
            width = (d[f"{pre}_log_stellar_mass_hi"] - d[f"{pre}_log_stellar_mass_lo"]).to_numpy(float)
            lum = d[f"{pre}_log_L5100_erg_med"].to_numpy(float)
            constrained = width < args.mass_width
            z = d[col].to_numpy(float)
            order = np.argsort(lum)
            art = ax.scatter(z[order], m[order], c=lum[order], s=4, lw=0, cmap="magma", vmin=42, vmax=46.5,
                             rasterized=True)
            binned(ax, z, m, "C0", "all")
            binned(ax, z[constrained], m[constrained], "C2", rf"68% mass interval < {args.mass_width:g} dex", ls="--")
            ax.set_xlim(0, 4.5)
            ax.set_xticks(np.arange(0.0, 4.01, 1.0))
            ax.set_ylim(*MASS_RANGE)
            ax.set_xlabel(label)
            ax.set_title(r"conditioned on $z_{\rm spec}$" if pre == "cond" else "marginal over redshift", fontsize=12)
            ax.text(0.03, 0.97, f"N = {np.isfinite(m).sum()} variable AGN\n{100 * np.mean(constrained[np.isfinite(m)]):.0f}% with constrained mass",
                    transform=ax.transAxes, va="top", fontsize=10, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
        axes[0, 0].set_ylabel(r"host $\log M_\star / M_\odot$ (posterior median)")
        axes[0, 0].legend(fontsize=10, loc="lower right")
        fig.tight_layout(rect=(0, 0, 0.92, 1))
        cax = fig.add_axes((0.93, 0.2, 0.015, 0.6))
        fig.colorbar(art, cax=cax, label=r"$\log \lambda L_{5100}$ [erg s$^{-1}$] (posterior median)")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
    print(f"{out}: {len(d)} variable objects, sources {d['source'].value_counts().to_dict()}; "
          f"last panel median 68% mass interval {np.nanmedian(width):.2f} dex, {100 * constrained.mean():.0f}% < {args.mass_width:g} dex")


if __name__ == "__main__":
    main()
