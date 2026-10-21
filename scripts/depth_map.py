"""Map DP2 coadd depth to find its survey layers (DDFs, WFD, anything in between).

    python scripts/depth_map.py --bands gri --order 7 --out ~/dp2_depth

For each HEALPix pixel (order 7 ~ 0.21 deg^2) the 5-sigma point-source depth is estimated
from the coadd PSF-flux errors of faint, sky-limited Objects (S/N < --snr-max):
m5 = 31.4 - 2.5 log10(5 * median(psfFluxErr / nJy)). Writes ``<out>.parquet`` (one row per
pixel) and ``<out>.png`` (area-weighted depth histogram per band and a sky map). Distinct
peaks in the histogram are the layers; the script also lists them.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lsdb
import numpy as np
import pandas as pd
from upath import UPath

AB_ZP_NJY = 31.4


def per_pixel(df, order, bands, ra_c, dec_c, snr_max):
    """Median coadd flux error of faint Objects per HEALPix pixel (one partition)."""
    pix = df.index.to_numpy().astype(np.uint64) >> np.uint64(2 * (29 - order))
    out = {"pix": pix, ra_c: df[ra_c].to_numpy(float), dec_c: df[dec_c].to_numpy(float)}
    for b in bands:
        f = df[f"{b}_psfFlux"].to_numpy(dtype=float, na_value=np.nan)
        e = df[f"{b}_psfFluxErr"].to_numpy(dtype=float, na_value=np.nan)
        ok = np.isfinite(f) & np.isfinite(e) & (e > 0) & (np.abs(f / np.where(e > 0, e, np.nan)) < snr_max)
        out[f"err_{b}"] = np.where(ok, e, np.nan)
    t = pd.DataFrame(out)
    g = t.groupby("pix", sort=False)
    res = g[[ra_c, dec_c]].median()
    res["n_obj"] = g.size()
    for b in bands:
        res[f"err_{b}"] = g[f"err_{b}"].median()
        res[f"n_{b}"] = g[f"err_{b}"].count()
    return res.reset_index()


def find_layers(m5: np.ndarray, area: float, min_area_frac: float = 0.005, width: float = 0.1):
    """Peaks of the area-weighted depth histogram (smoothed), as (depth, area deg^2)."""
    m5 = m5[np.isfinite(m5)]
    if len(m5) < 10:
        return []
    # pad so peaks at the extremes are interior bins
    bins = np.arange(np.floor(m5.min() / width) * width - 3 * width, m5.max() + 4 * width, width)
    h, edges = np.histogram(m5, bins)
    k = np.array([1, 2, 3, 2, 1], float)
    hs = np.convolve(h, k / k.sum(), mode="same")
    peaks = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1]
             and hs[i] * area >= min_area_frac * len(m5) * area]
    centers = 0.5 * (edges[1:] + edges[:-1])
    return [(centers[i], h[max(i - 2, 0): i + 3].sum() * area) for i in peaks]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="/rubin/lsdb_data/dp2/object_collection")
    p.add_argument("--bands", default="gri")
    p.add_argument("--order", type=int, default=7, help="HEALPix order of the depth map (7 ~ 0.21 deg^2)")
    p.add_argument("--snr-max", type=float, default=20.0, help="use Objects with S/N below this (sky-limited)")
    p.add_argument("--cone", nargs=3, type=float, metavar=("RA", "DEC", "RADIUS_ARCSEC"), help="for a quick test")
    p.add_argument("--out", required=True, help="output prefix (.parquet and .png are added)")
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--memory-limit", default="20GB")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bands = list(args.bands)

    full = lsdb.open_catalog(UPath(args.catalog))
    allc = list(getattr(full, "all_columns", full.columns))
    info = full.hc_structure.catalog_info
    ra_c, dec_c = info.ra_column, info.dec_column
    need = [c for b in bands for c in (f"{b}_psfFlux", f"{b}_psfFluxErr")]
    missing = [c for c in need if c not in allc]
    if missing:
        flux_like = [c for c in allc if "flux" in c.lower()]
        raise SystemExit(f"missing columns {missing}; flux-like columns available: {flux_like}")
    search = lsdb.ConeSearch(ra=args.cone[0], dec=args.cone[1], radius_arcsec=args.cone[2]) if args.cone else None
    cat = lsdb.open_catalog(UPath(args.catalog), columns=[ra_c, dec_c, *need], search_filter=search)

    from dask.distributed import Client

    kw = dict(order=args.order, bands=bands, ra_c=ra_c, dec_c=dec_c, snr_max=args.snr_max)
    empty = cat.meta if hasattr(cat, "meta") else cat._ddf._meta
    with Client(n_workers=args.n_workers, threads_per_worker=1, memory_limit=args.memory_limit) as client:
        logging.info("dask dashboard: %s", client.dashboard_link)
        parts = cat.map_partitions(per_pixel, meta=per_pixel(empty, **kw), **kw).compute()

    # a pixel split across partitions: combine with count-weighted medians (approximate)
    def combine(g):
        r = {ra_c: g[ra_c].median(), dec_c: g[dec_c].median(), "n_obj": g["n_obj"].sum()}
        for b in bands:
            w = g[f"n_{b}"].to_numpy(float)
            v = g[f"err_{b}"].to_numpy(float)
            ok = np.isfinite(v) & (w > 0)
            r[f"err_{b}"] = np.average(v[ok], weights=w[ok]) if ok.any() else np.nan
            r[f"n_{b}"] = w.sum()
        return pd.Series(r)

    px = parts.groupby("pix")[[c for c in parts.columns if c != "pix"]].apply(combine).reset_index()
    area = 41253.0 / (12 * 4**args.order)
    for b in bands:
        with np.errstate(invalid="ignore", divide="ignore"):
            px[f"m5_{b}"] = AB_ZP_NJY - 2.5 * np.log10(5 * px[f"err_{b}"])
        px.loc[px[f"n_{b}"] < 20, f"m5_{b}"] = np.nan  # too few faint Objects for a stable median
    out = Path(args.out).expanduser()
    px.to_parquet(out.with_suffix(".parquet"))
    logging.info("wrote %d pixels (%.0f deg^2) to %s", len(px), len(px) * area, out.with_suffix(".parquet"))

    for b in bands:
        m5 = px[f"m5_{b}"].to_numpy()
        q = np.nanpercentile(m5, [5, 50, 95])
        print(f"\n{b}: m5 5/50/95th percentile {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} over "
              f"{np.isfinite(m5).sum() * area:.0f} deg^2")
        for depth, a in find_layers(m5, area):
            print(f"   layer near m5 = {depth:.2f}: ~{a:.0f} deg^2")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for b in bands:
        m5 = px[f"m5_{b}"].dropna().to_numpy()
        ax[0].hist(m5, bins=np.arange(np.floor(m5.min()), np.ceil(m5.max()) + 0.1, 0.1),
                   weights=np.full(len(m5), area), histtype="step", lw=1.5, label=b)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("coadd 5σ point-source depth (AB)")
    ax[0].set_ylabel("area per 0.1 mag (deg²)")
    ax[0].legend()
    b0 = "i" if "i" in bands else bands[-1]
    sc = ax[1].scatter(px[ra_c], px[dec_c], c=px[f"m5_{b0}"], s=2, cmap="viridis")
    ax[1].invert_xaxis()
    ax[1].set_xlabel("RA (deg)")
    ax[1].set_ylabel("Dec (deg)")
    fig.colorbar(sc, ax=ax[1], label=f"{b0}-band m5")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    logging.info("wrote %s", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
