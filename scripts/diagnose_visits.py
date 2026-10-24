"""Visit-level false-positive diagnostics on one cone (loads its light curves into memory).

    python scripts/diagnose_visits.py --catalog $CAT --cone 53.13 -28.10 1800 \
        --err-scale 1.07 --out-visits ~/ecdfs_visits.csv

1. Per-visit ensemble statistics of reference (1-2 detection) objects: robust scatter of
   normalized residuals, 4-sigma outlier rate, and the coherent fractional offset of
   bright reference objects (zeropoint-like errors).
2. 5-sigma false-positive fractions vs magnitude after each candidate fix: removing
   per-visit offsets, dropping bad visits, and averaging visits within a night.
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
from rubin_imbh.pipeline import needed_columns, resolve_columns  # noqa: E402
from jaxpz.lightcurve import DIA, OBJECT  # noqa: E402
from rubin_imbh.visits import (  # noqa: E402
    add_residuals,
    apply_visit_offsets,
    bin_nightly,
    flatten,
    frac5_by_mag,
    mag_to_njy,
    visit_table,
)

MAG_BINS = [-np.inf, 19, 21, 23, 25]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["object", "dia"], default="object")
    p.add_argument("--catalog", help="default: the DP2 collection for --source")
    p.add_argument("--cone", nargs=3, type=float, required=True, metavar=("RA", "DEC", "RADIUS_ARCSEC"))
    p.add_argument("--err-scale", type=float, default=1.07)
    p.add_argument("--bright-mag", type=float, default=21.0, help="reference objects brighter than this set offsets")
    p.add_argument("--bad-sd", type=float, default=1.5, help="visits with robust sd_z above this are 'bad'")
    p.add_argument("--out-visits", help="write the per-visit table to this CSV")
    args = p.parse_args(argv)
    args.catalog = args.catalog or f"/rubin/lsdb_data/dp2/{'object' if args.source == 'object' else 'dia_object'}_collection"
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    full = lsdb.open_catalog(UPath(args.catalog))
    cols = resolve_columns(full, OBJECT if args.source == "object" else DIA)
    search = lsdb.ConeSearch(ra=args.cone[0], dec=args.cone[1], radius_arcsec=args.cone[2])
    nf = lsdb.open_catalog(UPath(args.catalog), search_filter=search, columns=needed_columns(full, cols)).compute()
    n_obj = len(nf)
    df = flatten(nf, cols, args.err_scale)
    print(f"objects: {n_obj}   good griz epochs: {len(df)}   visits: {df.visit.nunique()}   nights: {df.night.nunique()}")

    if cols.n_dia_sources:
        ref_obj = nf[cols.n_dia_sources].to_numpy() <= 2
        var_obj = nf[cols.n_dia_sources].to_numpy() > 5
        groups = {"ref(1-2)": ref_obj, "var(>5)": var_obj}
    else:  # every Object is a reference; split by morphology instead
        ext = nf["refExtendedness"].to_numpy(float) if "refExtendedness" in nf else np.full(len(nf), np.nan)
        groups = {"point": ext < 0.5, "extended": ext >= 0.5}
    ref = (df.nDia <= 2) & (df.n_ep >= 10)
    bright = df.F.abs() > mag_to_njy(args.bright_mag)

    # 1. per-visit tables. Pass 1 measures coherent offsets; pass 2 measures the scatter
    #    left after removing them, so a visit with only an offset is not called "bad".
    vt = visit_table(df, ref.to_numpy(), bright.to_numpy())
    df["f_off"] = apply_visit_offsets(df, vt)
    df_off = add_residuals(df.assign(f=df["f_off"]))
    vt2 = visit_table(df_off, ref.to_numpy(), bright.to_numpy())
    vt = vt.join(vt2[["sd_z", "f4"]].rename(columns={"sd_z": "sd_z_corr", "f4": "f4_corr"}))
    if args.out_visits:
        vt.to_csv(Path(args.out_visits).expanduser())
        print(f"wrote per-visit table to {args.out_visits}")
    print("\n=== per-visit ensemble statistics (reference objects), by band")
    for b, t in vt.groupby("band"):
        sig_off = (t.off_frac.abs() / t.off_err) > 5
        print(f"{b}: {len(t)} visits | |offset| median {t.off_frac.abs().median():.4f}, p90 "
              f"{t.off_frac.abs().quantile(0.9):.4f}, significant (>5 sigma): {sig_off.sum()} | "
              f"after offsets: sd_z median {t.sd_z_corr.median():.2f}, >{args.bad_sd}: {(t.sd_z_corr > args.bad_sd).sum()}, "
              f"f4 median {t.f4_corr.median():.4f} (expect ~6e-5)")
    print("\nlargest 15 coherent offsets (bright reference objects):")
    print(vt.reindex(vt.off_frac.abs().sort_values(ascending=False).index).head(15).round(4).to_string())
    print("\nworst 15 visits by scatter after removing offsets:")
    print(vt.sort_values("sd_z_corr", ascending=False).head(15).round(4).to_string())

    # same-night correlation of offsets: do visits on one night share an offset?
    t = vt.dropna(subset=["off_frac"])
    if len(t):
        night_mean = t.groupby(["band", "night"]).off_frac.transform("mean")
        resid = t.off_frac - night_mean
        print(f"\noffset scatter: total {t.off_frac.std():.4f}, within-night {resid.std():.4f} "
              f"(within << total means offsets are shared by a night)")

    # 2. false-positive fractions under candidate fixes
    variants = {}
    variants["raw"] = frac5_by_mag(df, n_obj, groups, MAG_BINS)
    variants["visit offsets removed"] = frac5_by_mag(df, n_obj, groups, MAG_BINS, flux="f_off")
    bad = vt.index[vt.sd_z_corr > args.bad_sd]
    keep = ~df.visit.isin(bad)
    variants[f"bad visits dropped ({len(bad)})"] = frac5_by_mag(df[keep], n_obj, groups, MAG_BINS)
    variants["offsets + bad dropped"] = frac5_by_mag(df[keep], n_obj, groups, MAG_BINS, flux="f_off")
    nightly = bin_nightly(df[keep].assign(f=df.loc[keep, "f_off"]))
    variants["offsets + bad dropped + nightly"] = frac5_by_mag(nightly, n_obj, groups, MAG_BINS, min_epochs=5)

    frac = pd.DataFrame({k: v[0] for k, v in variants.items()}).T.sort_index(axis=1)
    count = pd.DataFrame({k: v[1] for k, v in variants.items()}).T.sort_index(axis=1)
    for b in "griz":
        for gname in groups:
            sub = frac.loc[:, (b, gname)].reindex(columns=[f"{lo}-{hi}" for lo, hi in zip(MAG_BINS[:-1], MAG_BINS[1:])])
            n = count.loc["raw", (b, gname)]
            sizes = ", ".join(f"{k}: {int(v)}" for k, v in n.items())
            print(f"\n=== {b} {gname}: 5-sigma fraction by magnitude  (raw n per bin: {sizes})")
            print(sub.round(3).to_string())
    print("\nnightly binning requires >= 5 nights per band instead of >= 10 epochs")


if __name__ == "__main__":
    main()
