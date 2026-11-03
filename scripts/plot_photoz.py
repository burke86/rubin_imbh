"""Photo-z vs reference redshift for validate_real.py outputs, galaxies and AGN in separate panels.

One figure per model: columns are the reference samples, rows are galaxies and AGN. By default the
columns are the DDFs (COSMOS and CDF-S pooled) and DESI DR1; --columns picks others (see COLUMNS). Each panel shows the 1:1 line, the outlier boundary
|dz|/(1+z) = 0.15, and N, NMAD, outlier fraction and median bias. Points are coloured by the
width of their 95% credible interval in units of (1 + z_phot) (the 68% interval for older
validation files without ``post_redshift_lo95``), with the best constrained on top; a second
block of statistics covers the well-constrained objects (width below ``--well-constrained``,
default 0.3, i.e. the interval fits inside the outlier window) and the fraction they make up.
A second figure, photoz_<model>_kept.pdf, shows the outlier fraction of the objects kept when
cutting on that width, against the fraction kept.

All references are spectroscopic: COSMOS uses the COSMOS Spectroscopic Redshift Compilation
(Khostovan et al. 2026; flags 3/4 and broad-line 13/14), CDF-S its spec-z subset, DESI DR1 the AGN/galaxy
value-added catalog (scripts/desi_reference.py), and Milliquas.
AGN labels: COSMOS broad-line objects plus variable (all_varSigClip >= 5) or Gaia QSO sources;
the CDF-S X-ray catalog's AGN classification; DESI Redrock QSOs plus broad-line galaxies; every
Milliquas source is an AGN.

usage: python scripts/plot_photoz.py --val-dir data/sims --model v6c [--model v6w ...] --out data/sims/plots

Needs validate_real.py outputs val_<reference>_<model>.parquet for the references in the chosen columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jaxsedfit.mplstyle import use_style  # noqa: E402

# column key: (title, validation references pooled into the column)
COLUMNS = {"ddf": ("DDFs", ("cosmos-specz", "cdfs-xray")), "desi": ("DESI DR1", ("desi",)),
           "cosmos": ("COSMOS2020", ("cosmos-specz",)), "cdfs": ("CDF-S", ("cdfs-xray",)),
           "milliquas": ("Milliquas", ("milliquas",))}
DEFAULT_COLUMNS = ("ddf", "desi")
Z_MAX = 4.5
OUTLIER = 0.15
WIDTH_NORM = LogNorm(0.03, 3.0)
WIDTH_CMAP = "viridis_r"


def split(ref: str, d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Galaxy and AGN subsets of one validation table (an empty frame where a class is absent)."""
    if ref == "cdfs-xray":
        d = d[d["ref_z_type"] == "spec"]
        agn = d["ref_is_xray_agn"].astype(bool)
    elif ref == "milliquas":
        agn = pd.Series(True, index=d.index)
    elif ref == "desi":
        agn = d["ref_is_agn"].astype(bool)
    else:
        variable = (d["all_varSigClip"] >= 5) | d["gaia_qso"].fillna(False).astype(bool)
        agn = variable | d["ref_is_bl_agn"].astype(bool) if "ref_is_bl_agn" in d else variable
    return {"galaxies": d[~agn], "AGN": d[agn]}


def load_column(val_dir: Path, model: str, refs) -> dict[str, pd.DataFrame]:
    """Galaxy and AGN subsets pooled over the validation tables of ``refs`` ({} if none exist)."""
    parts = {}
    for ref in refs:
        path = val_dir / f"val_{ref}_{model}.parquet"
        if path.exists():
            for cls, sub in split(ref, pd.read_parquet(path)).items():
                parts.setdefault(cls, []).append(sub)
    return {cls: pd.concat(subs, ignore_index=True) for cls, subs in parts.items()}


def metrics(z_ref: np.ndarray, z_phot: np.ndarray) -> dict[str, float]:
    """N, NMAD and outlier fraction of dz / (1 + z_ref), and the median bias."""
    dz = (z_phot - z_ref) / (1 + z_ref)
    return {"n": len(dz), "nmad": 1.4826 * np.median(np.abs(dz - np.median(dz))),
            "outlier": np.mean(np.abs(dz) > OUTLIER), "bias": np.median(dz)}


def ci_width(d: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Redshift credible-interval width / (1 + z_phot): 95% if available, else 68%, and its label."""
    med = d["post_redshift_med"].to_numpy(float)
    if "post_redshift_lo95" in d:
        return (d["post_redshift_hi95"] - d["post_redshift_lo95"]).to_numpy(float) / (1 + med), "95%"
    return (d["post_redshift_hi"] - d["post_redshift_lo"]).to_numpy(float) / (1 + med), "68%"


def panel(ax, z_ref, z_phot, width, ci_label, well):
    """z_phot vs z_ref coloured by credible-interval width, with reference lines and metrics for all
    objects and for the well-constrained ones (width < ``well``). Returns the scatter artist."""
    art = None
    if len(z_ref):
        order = np.argsort(-np.nan_to_num(width, nan=np.inf))  # best constrained drawn last, on top
        art = ax.scatter(z_ref[order], z_phot[order], c=np.clip(width[order], WIDTH_NORM.vmin, WIDTH_NORM.vmax),
                         s=3 if len(z_ref) > 3000 else 8, lw=0, cmap=WIDTH_CMAP, norm=WIDTH_NORM, rasterized=True)
    grid = np.linspace(0, Z_MAX, 50)
    ax.plot(grid, grid, color="k", lw=0.8)
    for sign in (-1, 1):
        ax.plot(grid, grid + sign * OUTLIER * (1 + grid), color="k", lw=0.6, ls="--")
    if len(z_ref):
        m = metrics(z_ref, z_phot)
        text = (f"N = {m['n']}\nNMAD = {m['nmad']:.3f}\noutliers = {100 * m['outlier']:.1f}%\n"
                f"bias = {m['bias']:+.3f}")
        good = width < well
        if good.sum() >= 5:
            g = metrics(z_ref[good], z_phot[good])
            text += (f"\n\n{ci_label} CI < {well:g}(1+z): {100 * good.mean():.0f}%\n"
                     f"NMAD = {g['nmad']:.3f}\noutliers = {100 * g['outlier']:.1f}%")
        ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    ax.set_xlim(0, Z_MAX)
    ax.set_ylim(0, Z_MAX)
    ax.set_aspect("equal")
    return art


def plot_model(val_dir: Path, model: str, out: Path, well: float = 0.3, columns=DEFAULT_COLUMNS) -> Path:
    """Write the galaxies/AGN x reference-sample figure for one model and return its path."""
    art, ci_label = None, "95%"
    with use_style():
        fig, axes = plt.subplots(2, len(columns), figsize=(4.6 * len(columns) + 1.0, 9.6), sharex=True, sharey=True,
                                 squeeze=False)
        for j, key in enumerate(columns):
            title, refs = COLUMNS[key]
            ref_label = r"$z_{\rm spec}$"
            parts = load_column(val_dir, model, refs)
            for i, (cls, color) in enumerate([("galaxies", "C0"), ("AGN", "C3")]):
                ax = axes[i, j]
                sub = parts.get(cls)
                if sub is None or not len(sub):
                    reason = "no file" if not parts else ("all sources\nare AGN" if refs == ("milliquas",) else "none")
                    ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha="center", va="center", color="0.5")
                    ax.set_xlim(0, Z_MAX)
                    ax.set_ylim(0, Z_MAX)
                    ax.set_aspect("equal")
                else:
                    width, ci_label = ci_width(sub)
                    art = panel(ax, sub["ref_z_ref"].to_numpy(float), sub["post_redshift_med"].to_numpy(float),
                                width, ci_label, well) or art
                if i == 0:
                    ax.set_title(title)
                if i == 1:
                    ax.set_xlabel(ref_label)
                if j == 0:
                    ax.set_ylabel(rf"{cls}: $z_{{\rm phot}}$")
        fig.tight_layout(rect=(0, 0, 0.93, 1))
        if art is not None:
            cax = fig.add_axes((0.94, 0.25, 0.015, 0.5))
            fig.colorbar(art, cax=cax, label=rf"{ci_label} credible interval / $(1+z_{{\rm phot}})$")
        out.mkdir(parents=True, exist_ok=True)
        fn = out / f"photoz_{model}.pdf"
        fig.savefig(fn)
        plt.close(fig)
    return fn


def plot_kept(val_dir: Path, model: str, out: Path, well: float = 0.3, columns=DEFAULT_COLUMNS) -> Path:
    """Outlier fraction and NMAD of the objects kept when cutting on credible-interval width, against
    the fraction kept (galaxies and AGN panels, one line per data set; dots mark ``well``)."""
    with use_style():
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
        for j, key in enumerate(columns):
            title, refs = COLUMNS[key]
            parts = load_column(val_dir, model, refs)
            for ax, cls in zip(axes, ("galaxies", "AGN")):
                sub = parts.get(cls)
                if sub is None or len(sub) < 20:
                    continue
                width, ci_label = ci_width(sub)
                zr, zp = sub["ref_z_ref"].to_numpy(float), sub["post_redshift_med"].to_numpy(float)
                order = np.argsort(width)
                out_flag = (np.abs(zp - zr) / (1 + zr) > OUTLIER)[order]
                n = np.arange(1, len(order) + 1)
                kept, frac_out = n / len(order), np.cumsum(out_flag) / n
                ax.plot(kept, frac_out, color=f"C{j}", label=title)
                k = int(np.sum(width < well))
                if k:
                    ax.plot(kept[k - 1], frac_out[k - 1], "o", color=f"C{j}")
        for ax, cls in zip(axes, ("galaxies", "AGN")):
            ax.set_title(cls)
            ax.set_xlabel(f"fraction kept (narrowest {ci_label} CI first)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, None)
        axes[0].set_ylabel("outlier fraction of kept objects")
        axes[1].legend(fontsize=10, loc="upper left")
        fig.tight_layout()
        out.mkdir(parents=True, exist_ok=True)
        fn = out / f"photoz_{model}_kept.pdf"
        fig.savefig(fn)
        plt.close(fig)
    return fn


MASS_RANGE = (7.0, 12.5)
MASS_OUTLIER_DEX = 0.5


def mass_metrics(ref: np.ndarray, ours: np.ndarray) -> dict[str, float]:
    """N, median offset, NMAD scatter (dex) and fraction off by more than MASS_OUTLIER_DEX."""
    d = ours - ref
    return {"n": len(d), "offset": np.median(d), "scatter": 1.4826 * np.median(np.abs(d - np.median(d))),
            "outlier": np.mean(np.abs(d) > MASS_OUTLIER_DEX)}


def match_masses(val: pd.DataFrame, masses: pd.DataFrame, radius_arcsec: float = 1.0) -> pd.DataFrame:
    """Attach the nearest fixed-spec-z reference mass within ``radius_arcsec`` to each validated object."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    a = SkyCoord(val["coord_ra"].to_numpy(float) * u.deg, val["coord_dec"].to_numpy(float) * u.deg)
    b = SkyCoord(masses["ra"].to_numpy(float) * u.deg, masses["dec"].to_numpy(float) * u.deg)
    idx, sep, _ = a.match_to_catalog_sky(b)
    ok = sep.arcsec < radius_arcsec
    out = val[ok].copy()
    for c in ("logM_ref", "is_bl_agn", "z_spec"):
        out[c] = masses[c].to_numpy()[idx[ok]]
    return out


def plot_mass(val_dir: Path, model: str, masses_path: Path, out: Path) -> Path | None:
    """Posterior-median stellar mass vs COSMOS LePhare masses fitted at the spectroscopic redshift.

    Galaxies and AGN in separate panels (AGN: broad-line, variable or Gaia QSO). Each panel gives N,
    the median offset and NMAD scatter of log M_ours - log M_ref, the fraction off by more than
    MASS_OUTLIER_DEX, and the same statistics for objects whose photo-z is not an outlier.
    LePhare uses BC03 with a Chabrier IMF; our masses are surviving stellar masses (Kroupa IMF),
    so a few hundredths of a dex offset is expected. AGN reference masses come from galaxy-only fits.
    """
    path = val_dir / f"val_cosmos-specz_{model}.parquet"
    if not path.exists() or not masses_path.exists():
        return None
    d = match_masses(pd.read_parquet(path), pd.read_parquet(masses_path))
    agn = d["is_bl_agn"].astype(bool) | (d["all_varSigClip"] >= 5) | d["gaia_qso"].fillna(False).astype(bool)
    with use_style():
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.2), sharex=True, sharey=True)
        grid = np.linspace(*MASS_RANGE, 50)
        for ax, (label, sub, color) in zip(axes, [("galaxies", d[~agn], "C0"), ("AGN", d[agn], "C3")]):
            ref = sub["logM_ref"].to_numpy(float)
            ours = sub["post_log_stellar_mass_med"].to_numpy(float)
            if len(sub) > 3000:
                ax.hexbin(ref, ours, gridsize=60, extent=MASS_RANGE + MASS_RANGE, bins="log", mincnt=1, cmap="Greys")
            elif len(sub):
                ax.scatter(ref, ours, s=6, alpha=0.6, color=color, lw=0)
            ax.plot(grid, grid, color="k", lw=0.8)
            for sign in (-1, 1):
                ax.plot(grid, grid + sign * MASS_OUTLIER_DEX, color="k", lw=0.6, ls="--")
            if len(sub):
                m = mass_metrics(ref, ours)
                zp = sub["post_redshift_med"].to_numpy(float)
                good = np.abs(zp - sub["z_spec"].to_numpy(float)) / (1 + sub["z_spec"].to_numpy(float)) <= OUTLIER
                g = mass_metrics(ref[good], ours[good]) if good.any() else None
                text = (f"N = {m['n']}\noffset = {m['offset']:+.2f} dex\nscatter = {m['scatter']:.2f} dex\n"
                        f"|Δ| > {MASS_OUTLIER_DEX} dex: {100 * m['outlier']:.1f}%")
                if g:
                    text += f"\nphoto-z ok ({g['n']}): {g['offset']:+.2f} / {g['scatter']:.2f} dex"
                ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left", fontsize=10,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
            ax.set_title(label)
            ax.set_xlim(*MASS_RANGE)
            ax.set_ylim(*MASS_RANGE)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$\log M_\star$ (COSMOS2020 LePhare at $z_{\rm spec}$)")
        axes[0].set_ylabel(r"$\log M_\star$ (jaxpz posterior median)")
        fig.tight_layout()
        out.mkdir(parents=True, exist_ok=True)
        fn = out / f"mass_{model}.pdf"
        fig.savefig(fn)
        plt.close(fig)
    return fn


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-dir", default="data/sims", help="directory with val_<reference>_<model>.parquet files")
    p.add_argument("--model", action="append", required=True, help="model label (repeatable), e.g. v6c")
    p.add_argument("--out", default="data/sims/plots")
    p.add_argument("--masses", default="data/cosmos_specz_lephare_masses.parquet",
                   help="COSMOS fixed-spec-z LePhare masses (ra, dec, z_spec, is_bl_agn, logM_ref) for mass_<model>.pdf")
    p.add_argument("--columns", default=",".join(DEFAULT_COLUMNS),
                   help=f"comma-separated figure columns, from {', '.join(COLUMNS)}")
    p.add_argument("--well-constrained", type=float, default=0.3,
                   help="credible-interval width / (1 + z) below which an object counts as well constrained")
    args = p.parse_args(argv)
    for model in args.model:
        cols = tuple(c.strip() for c in args.columns.split(",") if c.strip())
        print(plot_model(Path(args.val_dir), model, Path(args.out), args.well_constrained, cols))
        print(plot_kept(Path(args.val_dir), model, Path(args.out), args.well_constrained, cols))
        print(plot_mass(Path(args.val_dir), model, Path(args.masses), Path(args.out)))


if __name__ == "__main__":
    main()
