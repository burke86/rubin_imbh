"""Photo-z vs reference redshift for validate_real.py outputs, galaxies and AGN in separate panels.

One figure per model: columns are the reference data sets (COSMOS2020, CDF-S X-ray, Milliquas),
rows are galaxies and AGN. Each panel shows the 1:1 line, the outlier boundary
|dz|/(1+z) = 0.15, and N, NMAD, outlier fraction and median bias.

AGN labels: CDF-S uses the X-ray catalogue's AGN classification; COSMOS2020 has no AGN flag,
so AGN there are variable (all_varSigClip >= 5) or Gaia QSO candidates; every Milliquas source
is an AGN. CDF-S panels use spectroscopic references only. COSMOS2020 references are its
photometric redshifts.

usage: python scripts/plot_photoz.py --val-dir data/sims --model v6c [--model v6w ...] --out data/sims/plots
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

DATASETS = [("cosmos2020", "COSMOS2020", r"$z_{\rm COSMOS2020\ phot}$"),
            ("cdfs-xray", "CDF-S", r"$z_{\rm spec}$"),
            ("milliquas", "Milliquas", r"$z_{\rm spec}$")]
Z_MAX = 4.5
OUTLIER = 0.15


def split(ref: str, d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Galaxy and AGN subsets of one validation table (an empty frame where a class is absent)."""
    if ref == "cdfs-xray":
        d = d[d["ref_z_type"] == "spec"]
        agn = d["ref_is_xray_agn"].astype(bool)
    elif ref == "milliquas":
        agn = pd.Series(True, index=d.index)
    else:
        agn = (d["all_varSigClip"] >= 5) | d["gaia_qso"].fillna(False).astype(bool)
    return {"galaxies": d[~agn], "AGN": d[agn]}


def metrics(z_ref: np.ndarray, z_phot: np.ndarray) -> dict[str, float]:
    """N, NMAD and outlier fraction of dz / (1 + z_ref), and the median bias."""
    dz = (z_phot - z_ref) / (1 + z_ref)
    return {"n": len(dz), "nmad": 1.4826 * np.median(np.abs(dz - np.median(dz))),
            "outlier": np.mean(np.abs(dz) > OUTLIER), "bias": np.median(dz)}


def panel(ax, z_ref, z_phot, color):
    """Density (many objects) or scatter (few) of z_phot vs z_ref, with reference lines and metrics."""
    if len(z_ref) > 3000:
        ax.hexbin(z_ref, z_phot, gridsize=70, extent=(0, Z_MAX, 0, Z_MAX), bins="log", mincnt=1, cmap="Greys")
    elif len(z_ref):
        ax.scatter(z_ref, z_phot, s=6, alpha=0.6, color=color, lw=0)
    grid = np.linspace(0, Z_MAX, 50)
    ax.plot(grid, grid, color="k", lw=0.8)
    for sign in (-1, 1):
        ax.plot(grid, grid + sign * OUTLIER * (1 + grid), color="k", lw=0.6, ls="--")
    if len(z_ref):
        m = metrics(z_ref, z_phot)
        ax.text(0.04, 0.96, f"N = {m['n']}\nNMAD = {m['nmad']:.3f}\noutliers = {100 * m['outlier']:.1f}%\n"
                f"bias = {m['bias']:+.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=11,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    ax.set_xlim(0, Z_MAX)
    ax.set_ylim(0, Z_MAX)
    ax.set_aspect("equal")


def plot_model(val_dir: Path, model: str, out: Path) -> Path:
    """Write the galaxies/AGN x data set figure for one model and return its path."""
    with use_style():
        fig, axes = plt.subplots(2, len(DATASETS), figsize=(4.6 * len(DATASETS), 9.6), sharex=True, sharey=True)
        for j, (ref, title, ref_label) in enumerate(DATASETS):
            path = val_dir / f"val_{ref}_{model}.parquet"
            parts = split(ref, pd.read_parquet(path)) if path.exists() else {}
            for i, (cls, color) in enumerate([("galaxies", "C0"), ("AGN", "C3")]):
                ax = axes[i, j]
                sub = parts.get(cls)
                if sub is None or not len(sub):
                    reason = "no file" if not parts else ("all sources\nare AGN" if ref == "milliquas" else "none")
                    ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha="center", va="center", color="0.5")
                    ax.set_xlim(0, Z_MAX)
                    ax.set_ylim(0, Z_MAX)
                    ax.set_aspect("equal")
                else:
                    panel(ax, sub["ref_z_ref"].to_numpy(float), sub["post_redshift_med"].to_numpy(float), color)
                if i == 0:
                    ax.set_title(title)
                if i == 1:
                    ax.set_xlabel(ref_label)
                if j == 0:
                    ax.set_ylabel(rf"{cls}: $z_{{\rm phot}}$")
        fig.suptitle(f"jaxpz {model}: photo-z vs reference redshift")
        fig.tight_layout()
        out.mkdir(parents=True, exist_ok=True)
        fn = out / f"photoz_{model}.png"
        fig.savefig(fn, dpi=150)
        plt.close(fig)
    return fn


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-dir", default="data/sims", help="directory with val_<reference>_<model>.parquet files")
    p.add_argument("--model", action="append", required=True, help="model label (repeatable), e.g. v6c")
    p.add_argument("--out", default="data/sims/plots")
    args = p.parse_args(argv)
    for model in args.model:
        print(plot_model(Path(args.val_dir), model, Path(args.out)))


if __name__ == "__main__":
    main()
