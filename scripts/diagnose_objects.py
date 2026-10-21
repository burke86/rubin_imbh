"""Object-level false-positive diagnostics on one cone (loads its light curves into memory).

    python scripts/diagnose_objects.py --catalog $CAT --cone 53.13 -28.10 1800 --err-scale 1.07

For reference (1-2 detection) objects flagged at 5 sigma, tests which picture fits:
  - outlier tails: are flags driven by a few |z| > 4 epochs, and are those concentrated in
    a subset of objects or spread over all?
  - airmass: do residuals correlate with airmass (differential chromatic refraction)?
  - morphology: are flagged objects mostly extended (galaxy subtraction residuals)?
  - environment: do they cluster near bright stars (halos, spikes)?
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
from rubin_imbh.objects import airmass, match_environment, tail_stats  # noqa: E402
from rubin_imbh.pipeline import needed_columns, resolve_columns  # noqa: E402
from rubin_imbh.varstats import Columns  # noqa: E402
from rubin_imbh.visits import AB_ZP_NJY, flatten, mag_to_njy, stats_from_flat  # noqa: E402

BANDS = "griz"
SIG = 5.0


def pick(columns, *candidates, contains=None):
    for c in candidates:
        if c in columns:
            return c
    if contains:
        hits = [c for c in columns if contains.lower() in c.lower()]
        if hits:
            return hits[0]
    return None


def table(mask_groups: dict, flagged: pd.Series) -> pd.DataFrame:
    rows = {k: (int(m.sum()), flagged[m].mean() if m.sum() else np.nan) for k, m in mask_groups.items()}
    return pd.DataFrame(rows, index=["n", "frac_flagged"]).T


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="/rubin/lsdb_data/dp2/dia_object_collection")
    p.add_argument("--object-catalog", default="/rubin/lsdb_data/dp2/object_collection",
                   help="Object collection for extendedness and bright stars ('none' to skip)")
    p.add_argument("--cone", nargs=3, type=float, required=True, metavar=("RA", "DEC", "RADIUS_ARCSEC"))
    p.add_argument("--err-scale", type=float, default=1.07)
    p.add_argument("--bright-star-mag", type=float, default=18.0)
    args = p.parse_args(argv)
    pd.set_option("display.width", 200)
    ra0, dec0, rad = args.cone
    search = lsdb.ConeSearch(ra=ra0, dec=dec0, radius_arcsec=rad)

    full = lsdb.open_catalog(UPath(args.catalog))
    cols = resolve_columns(full, Columns())
    nf = lsdb.open_catalog(UPath(args.catalog), search_filter=search, columns=needed_columns(full, cols)).compute()
    n_obj = len(nf)
    df = flatten(nf, cols, args.err_scale, BANDS)
    df["airmass"] = airmass(df["mjd"].to_numpy(), ra0, dec0)
    print(f"objects: {n_obj}   griz epochs: {len(df)}   airmass range {df.airmass.min():.2f}-{df.airmass.max():.2f}")

    st = stats_from_flat(df, n_obj)
    st_no4 = stats_from_flat(df[df.z.abs() <= 4], n_obj)
    tails = tail_stats(df)
    ndia = nf[cols.n_dia_sources].to_numpy()

    env = None
    if args.object_catalog.lower() != "none":
        ofull = lsdb.open_catalog(UPath(args.object_catalog))
        allc = list(getattr(ofull, "all_columns", ofull.columns))
        info = ofull.hc_structure.catalog_info
        ra_c, dec_c = info.ra_column, info.dec_column
        ext_c = pick(allc, "refExtendedness", "r_extendedness", "i_extendedness", contains="xtendedness")
        flux_c = pick(allc, "r_psfFlux", "i_psfFlux")
        load = [c for c in dict.fromkeys([ra_c, dec_c, ext_c, flux_c]) if c]
        print(f"object catalog columns used: {load}")
        obj = lsdb.open_catalog(UPath(args.object_catalog), search_filter=search, columns=load).compute()
        env = match_environment(nf["ra"].to_numpy(), nf["dec"].to_numpy(), obj, ra_c, dec_c, ext_c, flux_c,
                                mag_to_njy(args.bright_star_mag))
        print(f"Objects in cone: {len(obj)}; DiaObjects matched within 1\": {(env.sep_obj <= 1).mean():.2f}")

    for b in BANDS:
        ok = (st[f"{b}_nEpoch"] >= 10).to_numpy()
        ref = ok & (ndia <= 2)
        var = ok & (ndia > 5)
        flag = pd.Series(st[f"{b}_varSigClip"].to_numpy() >= SIG)
        with np.errstate(invalid="ignore", divide="ignore"):
            mag = AB_ZP_NJY - 2.5 * np.log10(st[f"{b}_medFlux"].to_numpy())
        tb = tails.xs(b, level="band").reindex(range(n_obj))
        n_out = tb["n_out"].fillna(0).to_numpy()
        print(f"\n================ {b} band: reference objects {ref.sum()}, flagged {flag[ref].mean():.3f}")

        # outlier tails
        fl = ref & flag.to_numpy()
        rescued = fl & (st_no4[f"{b}_varSig"].to_numpy() < SIG)
        print(f"flagged reference objects: {fl.sum()}; no longer significant after dropping all |z|>4 epochs: "
              f"{rescued.sum()} ({rescued.sum() / max(fl.sum(), 1):.2f})")
        bins = pd.cut(n_out[fl], [-1, 0, 1, 2, 5, 10, np.inf], labels=["0", "1", "2", "3-5", "6-10", ">10"])
        print("  |z|>4 epochs per flagged object:", pd.Series(bins).value_counts(sort=False).to_dict())
        srt = np.sort(n_out[ref])[::-1]
        top = srt[: max(len(srt) // 10, 1)].sum() / max(srt.sum(), 1)
        print(f"  reference objects with >=1 outlier: {(n_out[ref] >= 1).mean():.3f}; share of all outliers in the "
              f"top 10% of objects: {top:.2f} (spread evenly ~0.1, concentrated -> 1)")

        # airmass correlation
        if "r_airmass" in tb:
            r = tb["r_airmass"].to_numpy()
            print(f"  corr(z, airmass): flagged median |r| {np.nanmedian(np.abs(r[fl])):.2f}, |r|>0.3: "
                  f"{np.nanmean(np.abs(r[fl]) > 0.3):.2f} | unflagged median |r| {np.nanmedian(np.abs(r[ref & ~fl])):.2f}, "
                  f"|r|>0.3: {np.nanmean(np.abs(r[ref & ~fl]) > 0.3):.2f}")

        if env is not None:
            if "extendedness" in env:
                e = env["extendedness"].to_numpy()
                for label, base in (("reference", ref), ("var(>5)", var)):
                    t = table({"point (ext<0.5)": base & (e < 0.5), "extended (ext>=0.5)": base & (e >= 0.5),
                               "no Object within 1\"": base & np.isnan(e)}, flag)
                    print(f"  by morphology, {label}:\n" + t.round(3).to_string())
            if "d_bright" in env:
                dmin = env["d_bright"].to_numpy()
                groups = {f"{max(lo, 0)}-{hi}\"": ref & (dmin > lo) & (dmin <= hi)
                          for lo, hi in [(-1, 2), (2, 10), (10, 30), (30, 60), (60, np.inf)]}
                print(f"  by distance to nearest r<{args.bright_star_mag} point source, reference "
                      f"(0-2\" is the star itself):\n" + table(groups, flag).round(3).to_string())

        # magnitude dependence of the rescue, to see whether tails explain the faint excess
        rows = {}
        for lo, hi in [(-np.inf, 19), (19, 21), (21, 23), (23, 25)]:
            m = ref & (mag > lo) & (mag <= hi)
            rows[f"{lo}-{hi}"] = (int(m.sum()), flag[m].mean() if m.sum() else np.nan,
                                  (st_no4[f"{b}_varSig"].to_numpy()[m] >= SIG).mean() if m.sum() else np.nan)
        print("  reference flagged fraction by magnitude: clipped-one vs all |z|>4 removed\n" +
              pd.DataFrame(rows, index=["n", "varSigClip>=5", "no-|z|>4 varSig>=5"]).T.round(3).to_string())


if __name__ == "__main__":
    main()
