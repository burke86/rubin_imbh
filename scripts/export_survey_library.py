"""Export an empirical survey library (noise + cadence) from DP2 object_collection.

    python scripts/export_survey_library.py --out ~/survey_library.parquet

For each region (a cone with a layer label), samples random Objects and writes, per band:
the coadd PSF flux and error, and the nightly epochs and nightly-mean difference-flux
errors after the pipeline's flag cuts (format in rubin_imbh/amortized/library.py). The
training-set generator borrows noise and cadence from these rows.

Default regions: the DP2 DDFs (layer "ddf") and WFD cones at several ecliptic latitudes,
including the densely sampled ecliptic patch (layer "wfd"). Override with
--region NAME LAYER RA DEC (repeatable).
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
from rubin_imbh.varstats import OBJECT, night_index  # noqa: E402

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
                rng: np.random.Generator) -> pd.DataFrame:
    full = lsdb.open_catalog(UPath(catalog))
    cols = resolve_columns(full, OBJECT)
    allc = list(getattr(full, "all_columns", full.columns))
    info = full.hc_structure.catalog_info
    top = [cols.obj_id, info.ra_column, info.dec_column, "refExtendedness"]
    top += [c for b in BANDS for c in (f"{b}_psfFlux", f"{b}_psfFluxErr") if c in allc]
    fields = [cols.band, cols.flux_err, cols.time, *cols.flags]
    columns = [c for c in top if c in allc] + [f"{cols.lc}.{f}" for f in dict.fromkeys(fields)]
    cat = lsdb.open_catalog(UPath(catalog), columns=columns,
                            search_filter=lsdb.ConeSearch(ra=ra, dec=dec, radius_arcsec=radius))
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
        out[f"{b}_coadd_flux"] = nf[f"{b}_psfFlux"].to_numpy(float) if f"{b}_psfFlux" in nf else np.nan
        out[f"{b}_coadd_err"] = nf[f"{b}_psfFluxErr"].to_numpy(float) if f"{b}_psfFluxErr" in nf else np.nan
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="/rubin/lsdb_data/dp2/object_collection")
    p.add_argument("--region", nargs=4, action="append", metavar=("NAME", "LAYER", "RA", "DEC"),
                   help="replace the default regions (repeatable)")
    p.add_argument("--radius", type=float, default=600.0, help="cone radius per region (arcsec)")
    p.add_argument("--per-region", type=int, default=15000, help="Objects sampled per region")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    regions = [(r[0], r[1], float(r[2]), float(r[3])) for r in args.region] if args.region else DEFAULT_REGIONS
    rng = np.random.default_rng(args.seed)
    parts = [region_rows(args.catalog, *r, args.radius, args.per_region, rng) for r in regions]
    lib = pd.concat([x for x in parts if len(x)], ignore_index=True)
    lib.to_parquet(Path(args.out).expanduser())
    print(f"wrote {len(lib)} library rows ({lib.groupby('layer').size().to_dict()}) to {args.out}")


if __name__ == "__main__":
    main()
