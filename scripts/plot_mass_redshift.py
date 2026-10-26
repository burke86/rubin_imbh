"""Host stellar mass vs redshift for variable DP2 AGN, from jaxpz posterior summaries.

Takes validate_real.py outputs (objects with a reference redshift) or jaxpz predict outputs (no
reference redshift) and keeps variable objects (all_varSigClip >= --min-sig), de-duplicated by
objectId. Left panel: host mass against the reference (spectroscopic) redshift where there is one,
always conditioned on it (``cond_log_stellar_mass_*`` columns from condition_specz.py: the posterior
given the data and z = z_spec); right panel: the marginal posterior
against the photo-z posterior median, as for a catalogue without spectra. Points are coloured by the
width of their 68% host-mass interval (dex), best constrained on top.

Marginal masses of objects with a catastrophic photo-z follow the wrong redshift.

usage: python scripts/plot_mass_redshift.py data/sims/val_milliquas_v12_cond.parquet \
           data/sims/val_cosmos-specz_v12_cond.parquet data/sims/val_cdfs-xray_v12_cond.parquet \
           --out data/sims/plots/mass_redshift_variable_agn_v12.pdf
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

MASS_RANGE = (6.5, 11.5)


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
        if "cond_log_stellar_mass_med" not in d:
            raise SystemExit("the z_spec panel needs masses conditioned on z_spec: run scripts/condition_specz.py "
                             "on these tables and plot its *_cond.parquet outputs")
        x_axes.insert(0, ("ref_z_ref", r"$z_{\rm spec}$", "cond"))
    with use_style():
        fig, axes = plt.subplots(1, len(x_axes), figsize=(6.4 * len(x_axes), 4.6), sharey=True, squeeze=False)
        for ax, (col, label, pre) in zip(axes[0], x_axes):
            m = d[f"{pre}_log_stellar_mass_med"].to_numpy(float)
            width = (d[f"{pre}_log_stellar_mass_hi"] - d[f"{pre}_log_stellar_mass_lo"]).to_numpy(float)
            constrained = width < args.mass_width
            z = d[col].to_numpy(float)
            order = np.argsort(-np.nan_to_num(width, nan=np.inf))  # best constrained drawn last, on top
            art = ax.scatter(z[order], m[order], c=width[order], s=4, lw=0, cmap="viridis_r", vmin=0.0, vmax=2.5,
                             rasterized=True)
            ax.set_xlim(0, 4.5)
            ax.set_xticks(np.arange(0.0, 4.01, 1.0))
            ax.set_ylim(*MASS_RANGE)
            ax.set_xlabel(label)
        axes[0, 0].set_ylabel(r"host $\log M_\star / M_\odot$")
        cax = axes[0, -1].inset_axes((0.45, 0.2, 0.5, 0.04))
        cb = fig.colorbar(art, cax=cax, orientation="horizontal")
        cb.set_label(r"68% $\log M_\star$ interval (dex)", fontsize=10)
        cb.ax.tick_params(labelsize=9)
        fig.tight_layout()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
    print(f"{out}: {len(d)} variable objects, sources {d['source'].value_counts().to_dict()}; "
          f"last panel median 68% mass interval {np.nanmedian(width):.2f} dex, {100 * constrained.mean():.0f}% < {args.mass_width:g} dex")


if __name__ == "__main__":
    main()
