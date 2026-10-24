"""Export an empirical survey library (noise + cadence) from DP2 object_collection.

    python scripts/export_survey_library.py --out ~/survey_library.parquet

For each region (a cone with a layer label), samples random Objects and writes, per band:
the coadd flux and error (cModel by default; --flux-template), and the nightly epochs and nightly-mean difference-flux
errors after the pipeline's flag cuts (format in rubin_imbh/amortized/library.py). The
training-set generator borrows noise and cadence from these rows.

Regions: the DP2 DDFs (layer "ddf") plus WFD cones (layer "wfd"). With --depth-map (the
depth_map.py parquet), WFD cones are centred on sky pixels of typical WFD depth (between the
25th and 75th percentile of i-band m5), spread over ecliptic latitude; otherwise a fixed list
is used. Regions without coverage are skipped. Override everything with
--region NAME LAYER RA DEC (repeatable).

--strip-ids drops objectId and coordinates (the generator needs neither) and shuffles rows,
for a library that is to leave the RSP (see DATA_POLICY.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import lsdb
import numpy as np
import pandas as pd
from upath import UPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.pipeline import resolve_columns  # noqa: E402
from jaxpz.lightcurve import OBJECT, night_index  # noqa: E402

BANDS = "ugrizy"
DEFAULT_REGIONS = [
    ("ECDFS", "ddf", 53.13, -28.10),
    ("COSMOS", "ddf", 150.10, 2.18),
    ("ELAIS-S1", "ddf", 9.45, -44.00),
    ("EDFS", "ddf", 58.90, -49.32),
    ("wfd_753", "wfd", 320.6, -30.0),     # ecliptic latitude ~ -14
    ("wfd_724", "wfd", 353.6, -48.1),     # ~ -40
    ("wfd_735", "wfd", 337.5, -24.6),     # ~ -15
    ("wfd_290", "wfd", 331.9, -9.6),      # on the ecliptic, densely sampled
    ("wfd_401", "wfd", 208.1, -9.6),      # on the ecliptic, b = +50
]


def region_rows(catalog: str, name: str, layer: str, ra: float, dec: float, radius: float, n: int,
                rng: np.random.Generator, flux_template: str = "{b}_cModelFlux") -> pd.DataFrame:
    full = lsdb.open_catalog(UPath(catalog))
    cols = resolve_columns(full, OBJECT)
    allc = list(getattr(full, "all_columns", full.columns))
    info = full.hc_structure.catalog_info
    top = [cols.obj_id, info.ra_column, info.dec_column, "refExtendedness"]
    flux_cols = {b: (flux_template.format(b=b), flux_template.format(b=b) + "Err") for b in BANDS}
    top += [c for b in BANDS for c in flux_cols[b] if c in allc]
    fields = [cols.band, cols.flux_err, cols.time, *cols.flags]
    columns = [c for c in top if c in allc] + [f"{cols.lc}.{f}" for f in dict.fromkeys(fields)]
    try:
        cat = lsdb.open_catalog(UPath(catalog), columns=columns,
                                search_filter=lsdb.ConeSearch(ra=ra, dec=dec, radius_arcsec=radius))
    except ValueError as err:
        if "no coverage" not in str(err):
            raise
        logging.warning("%s: no coverage at (%.2f, %.2f); skipped", name, ra, dec)
        return pd.DataFrame()
    nf = cat.query("refExtendedness >= 0").compute()
    if len(nf) == 0:
        logging.warning("%s: no Objects", name)
        return pd.DataFrame()
    nf = nf.iloc[np.sort(rng.choice(len(nf), min(n, len(nf)), replace=False))]
    lc = nf[cols.lc]
    lengths = np.asarray(lc.nest.list_lengths)
    obj = np.repeat(np.arange(len(nf)), lengths)
    flat = lc.nest.to_flat()
    e = flat[cols.flux_err].to_numpy(float, na_value=np.nan)
    t = flat[cols.time].to_numpy(float, na_value=np.nan)
    good = np.isfinite(e) & (e > 0) & np.isfinite(t)
    for f in cols.flags:  # the pipeline's epoch flag cuts
        good &= ~flat[f].fillna(False).to_numpy(bool)
    df = pd.DataFrame({"obj": obj[good], "band": flat[cols.band].to_numpy()[good], "w": 1.0 / e[good] ** 2,
                       "t": t[good], "night": night_index(t[good])})
    nights = df.groupby(["obj", "band", "night"], observed=True).agg(sw=("w", "sum"), t=("t", "mean")).reset_index()
    nights["err"] = 1.0 / np.sqrt(nights.sw)
    nights = nights.sort_values(["obj", "band", "t"])

    out = pd.DataFrame({
        "objectId": nf[cols.obj_id].to_numpy(), "region": name, "layer": layer,
        "coord_ra": nf[info.ra_column].to_numpy(), "coord_dec": nf[info.dec_column].to_numpy(),
        "refExtendedness": nf["refExtendedness"].to_numpy(),
    })
    for b in BANDS:
        fc, ec = flux_cols[b]
        out[f"{b}_coadd_flux"] = nf[fc].to_numpy(float) if fc in nf else np.nan
        out[f"{b}_coadd_err"] = nf[ec].to_numpy(float) if ec in nf else np.nan
        nb = nights[nights.band == b]
        g = nb.groupby("obj")
        t_lists = g["t"].apply(lambda x: x.to_numpy()).reindex(range(len(nf)))
        e_lists = g["err"].apply(lambda x: x.to_numpy()).reindex(range(len(nf)))
        empty = np.zeros(0)
        out[f"{b}_night_mjd"] = [v if isinstance(v, np.ndarray) else empty for v in t_lists]
        out[f"{b}_night_err"] = [v if isinstance(v, np.ndarray) else empty for v in e_lists]
    nn = {b: np.median([len(v) for v in out[f"{b}_night_mjd"]]) for b in BANDS}
    logging.info("%s (%s): %d Objects sampled; median nights per band %s", name, layer, len(out),
                 {b: int(v) for b, v in nn.items()})
    return out


def wfd_regions_from_depth_map(path, n_per_bin: int = 3, band: str = "i", seed: int = 0):
    """WFD cone centres on pixels of typical depth, stratified by |ecliptic latitude|."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    px = pd.read_parquet(Path(path).expanduser())
    ra_c = next(c for c in ("coord_ra", "ra") if c in px)
    dec_c = next(c for c in ("coord_dec", "dec") if c in px)
    m5 = px[f"m5_{band}"]
    lo, hi = np.nanpercentile(m5, [25, 75])
    px = px[(m5 >= lo) & (m5 <= hi)].copy()
    c = SkyCoord(px[ra_c].to_numpy() * u.deg, px[dec_c].to_numpy() * u.deg)
    px["ecl"] = np.abs(c.barycentrictrueecliptic.lat.deg)
    px["gal_b"] = np.abs(c.galactic.b.deg)
    px = px[px.gal_b >= 25]
    rng = np.random.default_rng(seed)
    out = []
    for blo, bhi in ((0, 10), (10, 30), (30, 90)):
        sel = px[(px.ecl >= blo) & (px.ecl < bhi)]
        for _, r in sel.iloc[rng.choice(len(sel), min(n_per_bin, len(sel)), replace=False)].iterrows():
            out.append((f"wfd_ecl{blo}-{bhi}_{r[ra_c]:.1f}_{r[dec_c]:.1f}", "wfd", float(r[ra_c]), float(r[dec_c])))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="/rubin/lsdb_data/dp2/object_collection")
    p.add_argument("--region", nargs=4, action="append", metavar=("NAME", "LAYER", "RA", "DEC"),
                   help="replace the default regions (repeatable)")
    p.add_argument("--depth-map", help="depth_map.py parquet: choose WFD regions from typical-depth pixels")
    p.add_argument("--radius", type=float, default=600.0, help="cone radius per region (arcsec)")
    p.add_argument("--per-region", type=int, default=15000, help="Objects sampled per region")
    p.add_argument("--flux-template", default="{b}_cModelFlux",
                   help="coadd flux column (error = <name>Err); must match what validate_real.py feeds the model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strip-ids", action="store_true",
                   help="drop objectId and coordinates and shuffle rows (for copies leaving the RSP; DATA_POLICY.md)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.region:
        regions = [(r[0], r[1], float(r[2]), float(r[3])) for r in args.region]
    elif args.depth_map:
        regions = [r for r in DEFAULT_REGIONS if r[1] == "ddf"] + wfd_regions_from_depth_map(args.depth_map, seed=args.seed)
    else:
        regions = DEFAULT_REGIONS
    logging.info("regions: %s", [(r[0], r[1]) for r in regions])
    rng = np.random.default_rng(args.seed)
    parts = [region_rows(args.catalog, *r, args.radius, args.per_region, rng, args.flux_template) for r in regions]
    lib = pd.concat([x for x in parts if len(x)], ignore_index=True)
    if args.strip_ids:
        lib = lib.drop(columns=["objectId", "coord_ra", "coord_dec"])
        lib = lib.iloc[rng.permutation(len(lib))].reset_index(drop=True)
    lib.to_parquet(Path(args.out).expanduser())
    print(f"wrote {len(lib)} library rows ({lib.groupby('layer').size().to_dict()}) to {args.out}")


if __name__ == "__main__":
    main()
