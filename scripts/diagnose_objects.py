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


def cut_sequence(b, ref, var, flag, mag, env, args, tb):
    """Reference false-positive rate and variable retention as environment cuts are applied."""
    area = np.pi * (args.cone[2] / 3600.0) ** 2
    e, dmin = env["extendedness"].to_numpy(), env["d_bright"].to_numpy()
    far = dmin > args.veto_arcsec
    matched = ~np.isnan(e)
    stages = {
        "all": np.ones_like(ref),
        f"bright-star veto (>{args.veto_arcsec:g}\")": far,
        "+ Object within 1\"": far & matched,
        "+ extended": far & matched & (e >= 0.5),
        "+ point-like": far & matched & (e < 0.5),
    }
    rows = {}
    for k, m in stages.items():
        r, v = ref & m, var & m
        rows[k] = (int(r.sum()), flag[r].mean() if r.sum() else np.nan, (flag & r).sum() / area,
                   int(v.sum()), flag[v].mean() if v.sum() else np.nan)
    print(f"  cut sequence (cone area {area:.2f} deg^2):\n" + pd.DataFrame(
        rows, index=["n_ref", "ref_flagged", "ref_flagged_per_deg2", "n_var(>5)", "var_flagged"]).T.round(3).to_string())

    rows = {}
    for lo, hi in [(-np.inf, 19), (19, 21), (21, 23), (23, 25)]:
        mm = (mag > lo) & (mag <= hi)
        for label, sel in (("extended", stages["+ extended"]), ("point", stages["+ point-like"])):
            r, v = ref & sel & mm, var & sel & mm
            rows[(f"{lo}-{hi}", label)] = (int(r.sum()), flag[r].mean() if r.sum() >= 10 else np.nan,
                                           int(v.sum()), flag[v].mean() if v.sum() >= 10 else np.nan)
    print("  after veto + Object match, by magnitude:\n" + pd.DataFrame(
        rows, index=["n_ref", "ref_flagged", "n_var", "var_flagged"]).T.round(3).to_string())

    # colour test of the airmass correlation (DCR / colour-dependent extinction predict that
    # the sign of corr(z, airmass) follows colour)
    if "color" in env and "r_airmass" in tb:
        r_am = tb["r_airmass"].to_numpy()
        c = env["color"].to_numpy()
        sel = ref & far & matched & np.isfinite(c) & np.isfinite(r_am)
        if sel.sum() >= 30:
            x = pd.DataFrame({"color": c[sel], "r": r_am[sel], "flag": flag[sel]})
            rho_all = x["color"].corr(x["r"], method="spearman")
            rho_fl = x[x.flag]["color"].corr(x[x.flag]["r"], method="spearman") if x.flag.sum() >= 30 else np.nan
            x["tercile"] = pd.qcut(x["color"], 3, labels=["blue", "mid", "red"])
            t = x.groupby("tercile", observed=True).agg(n=("r", "size"), g_i=("color", "median"),
                                                        median_signed_r=("r", "median"), flagged=("flag", "mean"))
            print(f"  airmass vs colour (g-i), matched reference objects: Spearman(colour, signed r) "
                  f"all {rho_all:.2f}, flagged {rho_fl:.2f}\n" + t.round(3).to_string())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="/rubin/lsdb_data/dp2/dia_object_collection")
    p.add_argument("--object-catalog", default="/rubin/lsdb_data/dp2/object_collection",
                   help="Object collection for extendedness and bright stars ('none' to skip)")
    p.add_argument("--cone", nargs=3, type=float, required=True, metavar=("RA", "DEC", "RADIUS_ARCSEC"))
    p.add_argument("--err-scale", type=float, default=1.07)
    p.add_argument("--bright-star-mag", type=float, default=18.0)
    p.add_argument("--veto-arcsec", type=float, default=10.0, help="bright-star veto radius")
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
        g_c, i_c = pick(allc, "g_psfFlux"), pick(allc, "i_psfFlux")
        color_cols = (g_c, i_c) if g_c and i_c else None
        load = [c for c in dict.fromkeys([ra_c, dec_c, ext_c, flux_c, g_c, i_c]) if c]
        print(f"object catalog columns used: {load}")
        obj = lsdb.open_catalog(UPath(args.object_catalog), search_filter=search, columns=load).compute()
        env = match_environment(nf["ra"].to_numpy(), nf["dec"].to_numpy(), obj, ra_c, dec_c, ext_c, flux_c,
                                mag_to_njy(args.bright_star_mag), color_cols=color_cols)
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

        if env is not None and "extendedness" in env and "d_bright" in env:
            cut_sequence(b, ref, var, flag.to_numpy(), mag, env, args, tb)

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
