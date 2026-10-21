"""Validate the variability selection against Chandra X-ray AGN in the (E-)CDF-S.

    python scripts/validate_xray.py --stats ~/ecdfs_env_stats.parquet \
        --object-catalog /rubin/lsdb_data/dp2/object_collection

``--stats`` is a ``run --no-select --stats-only --min-dia-sources 0 --object-catalog ...``
output (all Object-matched DiaObjects with statistics and environment flags).

X-ray catalogs (VizieR): CDF-S 7 Ms, Luo et al. 2017 (J/ApJS/228/2/sources, ``OType``) and
E-CDF-S 250 ks, Xue et al. 2016 (J/ApJS/224/15/table9 + table12, ``Type``); E-CDF-S sources
within the match radius of a 7 Ms source are dropped.

Reported for several selection variants (bands, number of bands), all requiring
``envClean`` (Object match, no r<18 star within 10"):
  - completeness: fraction of X-ray AGN with a coadd Object counterpart that are selected,
    by magnitude (an AGN with no DiaObject at all counts as missed);
  - X-ray match fraction of selected objects inside the X-ray footprint, a lower limit on
    AGN purity (not every variable AGN is X-ray detected), with the chance-coincidence rate
    from shifted X-ray positions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIG = 5.0
MIN_EPOCHS = 10
VARIANTS = {
    "r": ("r", 1), "i": ("i", 1), "z": ("z", 1), "g": ("g", 1),
    "riz >=1": ("riz", 1), "riz >=2": ("riz", 2), "rz >=1": ("rz", 1), "rz both": ("rz", 2), "griz >=2": ("griz", 2),
}
MAG_BINS = [-np.inf, 21, 22, 23, 24]
XRAY = [("J/ApJS/228/2/sources", "OType", "CDF-S 7Ms"),
        ("J/ApJS/224/15/table9", "Type", "E-CDF-S 250ks"),
        ("J/ApJS/224/15/table12", "Type", "E-CDF-S 250ks supp")]


def fetch_vizier(table: str) -> pd.DataFrame:
    """Whole VizieR table with decimal J2000 positions as ``ra``/``dec``."""
    try:
        from astroquery.vizier import Vizier

        t = Vizier(columns=["**", "_RAJ2000", "_DEJ2000"], row_limit=-1).get_catalogs(table)[0]
        df = t.to_pandas()
        return df.rename(columns={"_RAJ2000": "ra", "_DEJ2000": "dec"})
    except ImportError:
        import pyvo

        tap = pyvo.dal.TAPService("https://tapvizier.cds.unistra.fr/TAPVizieR/tap")
        df = tap.search(f'SELECT * FROM "{table}"').to_table().to_pandas()
        ra, dec = df["RAJ2000"], df["DEJ2000"]
        if ra.dtype == object:  # sexagesimal strings
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            c = SkyCoord(ra.astype(str), dec.astype(str), unit=(u.hourangle, u.deg))
            ra, dec = c.ra.deg, c.dec.deg
        return df.assign(ra=np.asarray(ra, float), dec=np.asarray(dec, float))


def load_xray(match_arcsec: float) -> pd.DataFrame:
    parts = []
    for table, type_col, label in XRAY:
        df = fetch_vizier(table)
        cls = df[type_col].astype(str).str.strip()
        parts.append(pd.DataFrame({"ra": df["ra"], "dec": df["dec"], "cls": cls, "survey": label}))
        print(f"{label}: {len(df)} sources, {cls.value_counts().to_dict()}")
    deep, wide = parts[0], pd.concat(parts[1:], ignore_index=True)
    d = sky(deep)
    idx, sep, _ = sky(wide).match_to_catalog_sky(d)
    wide = wide[sep.arcsec > match_arcsec]
    x = pd.concat([deep, wide], ignore_index=True)
    print(f"combined X-ray catalog: {len(x)} sources, {x.cls.value_counts().to_dict()}")
    return x


def sky(df, ra="ra", dec="dec"):
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    return SkyCoord(df[ra].to_numpy(float) * u.deg, df[dec].to_numpy(float) * u.deg)


def nearest(a, b, radius):
    """For each position in ``a``, index into ``b`` of the nearest one within ``radius`` arcsec, else -1."""
    idx, sep, _ = a.match_to_catalog_sky(b)
    return np.where(sep.arcsec <= radius, idx, -1)


def in_footprint(pos, xray_pos, radius_arcmin=2.0, min_sources=3):
    """At least ``min_sources`` of ``xray_pos`` within ``radius`` (survey-area membership)."""
    import astropy.units as u

    i, _, _, _ = xray_pos.search_around_sky(pos, radius_arcmin * u.arcmin)
    return np.bincount(i, minlength=len(pos)) >= min_sources


def selected(st: pd.DataFrame, bands: str, k: int) -> np.ndarray:
    n = np.zeros(len(st), int)
    for b in bands:
        n += ((st[f"{b}_varSigClip"] >= SIG) & (st[f"{b}_nEpoch"] >= MIN_EPOCHS)).to_numpy()
    return (n >= k) & st["envClean"].to_numpy(dtype=bool)


def floor_scan(st, a, foot, is_agn, chance, ext, floors):
    """Completeness and X-ray purity vs a common systematic floor, from one floor-free stats run."""
    from rubin_imbh.calib import rescaled_sig

    rows = {}
    for s in floors:
        x = st.copy()
        for b in "griz":
            if f"{b}_rmsErr" in x:
                x[f"{b}_varSigClip"] = rescaled_sig(st, b, s)
        for name in ("riz >=1", "r", "rz both"):
            bands, kk = VARIANTS[name]
            sel = selected(x, bands, kk)
            hit = np.where(a.st_row >= 0, sel[np.clip(a.st_row, 0, None)], False)
            m24, m22 = a.rmag <= 24, a.rmag <= 22
            fs = sel & foot
            rows[(s, name)] = {
                "compl r<=22": hit[m22].mean(), "compl r<=24": hit[m24].mean(),
                "n_sel": int(fs.sum()), "xray_AGN": is_agn[fs].mean() if fs.sum() else np.nan,
                "xray_AGN_ext": is_agn[fs & ext].mean() if (fs & ext).sum() else np.nan,
                "n_ext": int((fs & ext).sum()), "chance": chance[fs].mean() if fs.sum() else np.nan,
            }
    print("\n=== floor scan (common floor in all bands; rows = floor, variant)")
    print(pd.DataFrame(rows).T.round(3).to_string())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", required=True)
    p.add_argument("--object-catalog", default="/rubin/lsdb_data/dp2/object_collection")
    p.add_argument("--match-arcsec", type=float, default=1.5, help="X-ray to optical match radius")
    p.add_argument("--out", help="CSV of selected objects (riz >=1) with X-ray matches")
    p.add_argument("--floors", help="comma-separated trial floors (fraction of medFlux, all bands) to compare; "
                   "the stats must be computed with --sys-frac 0, e.g. 0,0.005,0.01,0.015,0.025")
    args = p.parse_args(argv)
    pd.set_option("display.width", 200)

    st = pd.read_parquet(Path(args.stats).expanduser()).reset_index(drop=True)
    ora = next(c for c in ("coord_ra", "ra_obj") if c in st)
    odec = next(c for c in ("coord_dec", "dec_obj") if c in st)
    # position of each row: the DiaObject (dia source) or the Object itself (object source)
    pra, pdec = ("ra", "dec") if "ra" in st else (ora, odec)
    ext = next((c for c in ("refExtendedness", "r_extendedness") if c in st), None)
    print(f"stats: {len(st)} rows, envClean {st.envClean.sum()}")

    x = load_xray(args.match_arcsec)
    agn = x[x.cls == "AGN"].reset_index(drop=True)
    # completeness only where the stats run has coverage (its cone may not span the whole
    # X-ray survey): at least 5 Object-matched DiaObjects within 2'
    covered = in_footprint(sky(agn), sky(st, pra, pdec), radius_arcmin=2.0, min_sources=5)
    print(f"X-ray AGN inside the stats coverage: {covered.sum()} of {len(agn)}")
    agn = agn[covered].reset_index(drop=True)
    xpos, apos = sky(x), sky(agn)

    # --- completeness: X-ray AGN with an Object counterpart
    import lsdb
    from upath import UPath

    from rubin_imbh.environment import resolve_object_columns

    oc = resolve_object_columns(lsdb.open_catalog(UPath(args.object_catalog)))
    ra0, dec0 = agn.ra.mean(), agn.dec.mean()
    rad = 3600 * np.max(sky(pd.DataFrame({"ra": [ra0], "dec": [dec0]}))[0].separation(apos).deg) + 10
    cols = [c for c in dict.fromkeys([oc.ra, oc.dec, oc.ext, "r_psfFlux", "i_psfFlux"]) if c]
    obj = lsdb.open_catalog(UPath(args.object_catalog), columns=cols,
                            search_filter=lsdb.ConeSearch(ra=ra0, dec=dec0, radius_arcsec=rad)).compute()
    obj = obj.reset_index(drop=True)
    j = nearest(apos, sky(obj, oc.ra, oc.dec), args.match_arcsec)
    has_obj = j >= 0
    print(f"X-ray AGN: {len(agn)}; with a coadd Object within {args.match_arcsec}\": {has_obj.sum()}")
    a = agn[has_obj].copy()
    o = obj.iloc[j[has_obj]].reset_index(drop=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        a["rmag"] = 31.4 - 2.5 * np.log10(o["r_psfFlux"].to_numpy(float)) if "r_psfFlux" in o else np.nan
    a["extended"] = o[oc.ext].to_numpy(float) >= 0.5 if oc.ext else np.nan
    # link each counterpart Object to the stats row whose matched Object it is
    k = nearest(sky(o, oc.ra, oc.dec), sky(st, ora, odec), 0.3)
    a["st_row"] = k
    print(f"  ...of which are in the stats file (dia source: have a DiaObject): {(k >= 0).sum()}")

    rows = {}
    for name, (bands, kk) in VARIANTS.items():
        sel = selected(st, bands, kk)
        hit = np.where(a.st_row >= 0, sel[np.clip(a.st_row, 0, None)], False)
        r = {}
        for lo, hi in zip(MAG_BINS[:-1], MAG_BINS[1:]):
            m = (a.rmag > lo) & (a.rmag <= hi)
            r[f"r {lo}-{hi}"] = hit[m].mean() if m.sum() >= 5 else np.nan
        m = a.rmag <= 24
        r["all r<=24"] = hit[m].mean()
        r["n r<=24"] = int(m.sum())
        rows[name] = r
    print("\n=== completeness: fraction of X-ray AGN (with Object counterpart) selected")
    print(pd.DataFrame(rows).T.round(3).to_string())

    # --- X-ray match fraction of selected objects inside the footprint
    spos = sky(st, pra, pdec)
    foot = in_footprint(spos, xpos)
    any_x = nearest(spos, xpos, args.match_arcsec) >= 0
    is_agn = nearest(spos, apos, args.match_arcsec) >= 0
    shifted = x.assign(ra=x.ra + 60 / 3600 / np.cos(np.deg2rad(x.dec)), dec=x.dec + 60 / 3600)
    chance = nearest(spos, sky(shifted), args.match_arcsec) >= 0
    e = st[ext].to_numpy(float) >= 0.5 if ext else np.zeros(len(st), bool)
    rows = {}
    for name, (bands, kk) in VARIANTS.items():
        sel = selected(st, bands, kk) & foot
        r = {"n_sel": int(sel.sum()), "xray_AGN": is_agn[sel].mean() if sel.sum() else np.nan,
             "any_xray": any_x[sel].mean() if sel.sum() else np.nan,
             "chance": chance[sel].mean() if sel.sum() else np.nan}
        for lab, mm in (("ext", e), ("pt", ~e)):
            s2 = sel & mm
            r[f"n_{lab}"] = int(s2.sum())
            r[f"xray_AGN_{lab}"] = is_agn[s2].mean() if s2.sum() else np.nan
        rows[name] = r
    ref = foot & st["envClean"].to_numpy(dtype=bool)
    print(f"\n=== selected objects inside the X-ray footprint ({foot.sum()} rows; "
          f"X-ray AGN match rate among all envClean ones: {is_agn[ref].mean():.3f})")
    print(pd.DataFrame(rows).T.round(3).to_string())
    print("xray_AGN is a lower limit on purity; 'chance' is the match rate with X-ray positions shifted by 85\"")

    if args.floors:
        floor_scan(st, a, foot, is_agn, chance, e, [float(v) for v in args.floors.split(",")])

    if args.out:
        sel = selected(st, "riz", 1)
        out = st[sel].assign(in_xray_footprint=foot[sel], xray_AGN=is_agn[sel], any_xray=any_x[sel])
        out.to_csv(Path(args.out).expanduser(), index=False)
        print(f"wrote {len(out)} selected objects to {args.out}")


if __name__ == "__main__":
    main()
