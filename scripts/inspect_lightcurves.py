"""Look at the light curves behind a selection: per-object plots and shared-night diagnostics.

    python scripts/inspect_lightcurves.py ~/dp2_variables_v2_all.parquet --tiles o3_p761 o3_p753 \
        --n 300 --plot 24 --out ~/lc_check

For a random sample of selected (envClean) objects in each tile, fetches the forced light
curves, bins them per night and band as the pipeline does, and reports:
  - per object: nights per band, the significance and sign of each night's deviation;
  - per tile: the nights on which sampled objects deviate (|z| > 4) and how many objects share
    each such night. Excursions concentrated on a few shared nights point to night/visit-level
    problems (calibration, templates, conditions) rather than real variability; asteroids hit
    one object per night and are positive.
Writes ``<out>_<tile>.png`` (light curves) and ``<out>_<tile>_nights.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lsdb
import numpy as np
import pandas as pd
from upath import UPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jaxpz.lightcurve import night_index  # noqa: E402

BANDS = "griz"
FIELDS = ["band", "psfDiffFlux", "psfDiffFluxErr", "psfFlux", "midpointMjdTai", "psfDiffFlux_flag",
          "pixelFlags_bad", "pixelFlags_edge", "pixelFlags_saturatedCenter", "pixelFlags_crCenter",
          "pixelFlags_suspectCenter", "pixelFlags_interpolatedCenter", "invalidPsfFlag", "psfFlux_flag"]


def fetch(catalog: str, rows: pd.DataFrame, lc_col: str) -> pd.DataFrame:
    """Flat light curves (one row per epoch) for the given objects, via small cone searches."""
    full = lsdb.open_catalog(UPath(catalog))
    have = set(full.dtypes[lc_col].field_names)
    fields = [f for f in FIELDS if f in have]
    cols = ["objectId", "coord_ra", "coord_dec"] + [f"{lc_col}.{f}" for f in fields]
    out = []
    for r in rows.itertuples():
        cat = lsdb.open_catalog(UPath(catalog), columns=cols,
                                search_filter=lsdb.ConeSearch(ra=r.coord_ra, dec=r.coord_dec, radius_arcsec=0.5))
        nf = cat.compute()
        nf = nf[nf.objectId == r.objectId]
        if len(nf) == 0:
            continue
        flat = nf[lc_col].nest.to_flat()
        flat["objectId"] = r.objectId
        out.append(flat.reset_index(drop=True))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def nightly(lc: pd.DataFrame, err_scale: float, sys_frac: float) -> pd.DataFrame:
    flags = [c for c in FIELDS[5:] if c in lc]
    good = np.isfinite(lc.psfDiffFlux) & np.isfinite(lc.psfDiffFluxErr) & (lc.psfDiffFluxErr > 0)
    for c in flags:
        good &= ~lc[c].fillna(False).astype(bool)
    x = lc[good & lc.band.isin(list(BANDS))].copy()
    e = np.hypot(err_scale * x.psfDiffFluxErr, sys_frac * x.psfFlux.abs().fillna(0))
    x["w"] = 1 / e**2
    x["wf"] = x.w * x.psfDiffFlux
    x["night"] = night_index(x.midpointMjdTai)
    g = x.groupby(["objectId", "band", "night"], observed=True)
    n = g.agg(sw=("w", "sum"), swf=("wf", "sum"), mjd=("midpointMjdTai", "mean"), nvis=("w", "size")).reset_index()
    n["f"] = n.swf / n.sw
    n["e"] = 1 / np.sqrt(n.sw)
    # deviation of each night from the object's weighted mean in that band
    gb = n.groupby(["objectId", "band"], observed=True)
    mu = gb.swf.transform("sum") / gb.sw.transform("sum")
    n["z"] = (n.f - mu) / n.e
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("selected", help="combined selection parquet (summarize_selection.py output)")
    p.add_argument("--tiles", nargs="+", required=True)
    p.add_argument("--catalog", default="/rubin/lsdb_data/dp2/object_collection")
    p.add_argument("--lc-column", default="objectForcedSource")
    p.add_argument("--n", type=int, default=300, help="objects sampled per tile for the night statistics")
    p.add_argument("--plot", type=int, default=24, help="light curves plotted per tile")
    p.add_argument("--route", choices=["any", "combined", "per-band"], default="any")
    p.add_argument("--err-scale", type=float, default=1.0)
    p.add_argument("--sys-frac", type=float, default=0.0075)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    pd.set_option("display.width", 200)
    d = pd.read_parquet(Path(args.selected).expanduser())
    if "envClean" in d:
        d = d[d.envClean.astype(bool)]
    if args.route == "combined":
        d = d[d.varCombo & (d.nVarBands == 0)]
    elif args.route == "per-band":
        d = d[d.nVarBands >= 1]
    rng = np.random.default_rng(0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for tile in args.tiles:
        sel = d[d.tile == tile]
        if len(sel) == 0:
            print(f"\n{tile}: no selected objects")
            continue
        sample = sel.iloc[rng.choice(len(sel), min(args.n, len(sel)), replace=False)]
        lc = fetch(args.catalog, sample, args.lc_column)
        if len(lc) == 0:
            print(f"\n{tile}: no light curves fetched")
            continue
        n = nightly(lc, args.err_scale, args.sys_frac)
        per_obj = n.groupby("objectId").agg(nights=("night", "nunique"), n_dev=("z", lambda z: int((z.abs() > 4).sum())),
                                            n_pos=("z", lambda z: int((z > 4).sum())), n_neg=("z", lambda z: int((z < -4).sum())))
        dev = n[n.z.abs() > 4]
        shared = dev.groupby("night").agg(n_obj=("objectId", "nunique"), n_pos=("z", lambda z: int((z > 0).sum())),
                                          n_neg=("z", lambda z: int((z < 0).sum())), bands=("band", lambda b: "".join(sorted(set(b)))))
        shared = shared.sort_values("n_obj", ascending=False)
        shared["frac_of_sample"] = shared.n_obj / per_obj.shape[0]
        print(f"\n=== {tile}: {len(sel)} selected; {per_obj.shape[0]} sampled light curves fetched")
        print(f"median nights per object {per_obj.nights.median():.0f}; deviating nights (|z|>4) per object: "
              f"median {per_obj.n_dev.median():.0f}, positive fraction {dev.z.gt(0).mean():.2f}")
        print("objects by number of deviating nights:", per_obj.n_dev.clip(upper=6).value_counts().sort_index().to_dict(),
              "(6 = 6 or more)")
        print("nights shared by the most deviating objects (a real-variable sample shares few):")
        print(shared.head(10).round(3).to_string())
        shared.to_csv(f"{Path(args.out).expanduser()}_{tile}_nights.csv")

        ids = sample.objectId.to_numpy()[: args.plot]
        ncol = 4
        nrow = int(np.ceil(len(ids) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow), squeeze=False)
        colors = dict(g="tab:green", r="tab:red", i="tab:orange", z="tab:purple")
        bad_nights = set(shared.index[shared.n_obj >= max(3, 0.05 * per_obj.shape[0])])
        for ax, oid in zip(axes.ravel(), ids):
            o = n[n.objectId == oid]
            for b, ob in o.groupby("band"):
                ax.errorbar(ob.mjd, ob.f, ob.e, fmt="o", ms=3, color=colors.get(b, "k"), label=b)
            for night in bad_nights & set(o.night):
                ax.axvline(night + 0.5, color="0.7", lw=0.8, zorder=0)
            row = sel[sel.objectId == oid].iloc[0]
            ax.set_title(f"{oid} r={31.4 - 2.5 * np.log10(row.r_psfFlux):.1f} "
                         f"{'ext' if row.refExtendedness >= 0.5 else 'pt'} σc={row.all_varSigClip:.1f}", fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(ids):]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=6)
        fig.suptitle(f"{tile}: nightly difference flux (nJy); grey lines = nights shared by many deviating objects", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{Path(args.out).expanduser()}_{tile}.png", dpi=110)
        print(f"wrote {args.out}_{tile}.png")


if __name__ == "__main__":
    main()
