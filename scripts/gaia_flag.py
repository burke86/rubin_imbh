"""Flag Gaia DR3 stars (and quasar candidates) among selected variables.

    python scripts/gaia_flag.py ~/dp2_variables_v2_all.parquet --out ~/dp2_variables_v2_gaia.parquet

Crossmatches each selected object with Gaia DR3 (public HATS catalog, read remotely) within
--radius and adds:
  gaia_sep, gaia_G, gaia_plx_snr, gaia_pm_snr   match separation (arcsec), G mag, parallax and
                                                 proper-motion significance
  gaia_qso    Gaia quasar candidate (in_qso_candidates, or DSC quasar probability > 0.5)
  gaia_var    Gaia photometric variable (phot_variable_flag == VARIABLE)
  gaia_star   parallax or proper motion > --snr sigma, and not gaia_qso

High-proper-motion stars also fake variability in difference images (the star moves
between template and epochs). Gaia reaches G ~ 20.7, so fainter stars are not flagged.
"""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path

import lsdb
import numpy as np
import pandas as pd
from upath import UPath

GAIA = "https://linea.data.lsdb.io/hats/gaia_dr3"
GAIA_COLS = ["source_id", "ra", "dec", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error",
             "phot_g_mean_mag", "in_qso_candidates", "classprob_dsc_combmod_quasar", "phot_variable_flag"]


def gaia_flags(x: pd.DataFrame, snr: float) -> pd.DataFrame:
    """Flag columns from matched Gaia columns (NaN where unmatched)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        plx = (x["parallax"] / x["parallax_error"]).astype(float)
        pm = np.hypot((x["pmra"] / x["pmra_error"]).astype(float), (x["pmdec"] / x["pmdec_error"]).astype(float))
    qso = x["in_qso_candidates"].fillna(False).astype(bool) | (x["classprob_dsc_combmod_quasar"].astype(float) > 0.5)
    var = x["phot_variable_flag"].astype(str).str.strip().eq("VARIABLE")
    star = ((plx > snr) | (pm > snr)) & ~qso
    return pd.DataFrame({
        "gaia_G": x["phot_g_mean_mag"].astype(float), "gaia_plx_snr": plx, "gaia_pm_snr": pm,
        "gaia_qso": qso.to_numpy(), "gaia_var": var.to_numpy(), "gaia_star": star.fillna(False).to_numpy(),
    }, index=x.index)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("selected", help="selection parquet with objectId, coord_ra, coord_dec")
    p.add_argument("--out", required=True)
    p.add_argument("--gaia", default=GAIA, help="Gaia DR3 HATS catalog (path or URL)")
    p.add_argument("--radius", type=float, default=1.0, help="match radius (arcsec)")
    p.add_argument("--snr", type=float, default=5.0, help="parallax / proper-motion significance for a star")
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--memory-limit", default="12GB")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    d = pd.read_parquet(Path(args.selected).expanduser()).reset_index(drop=True)
    pos = d[["objectId", "coord_ra", "coord_dec"]].copy()
    key = "partition_rows" if "partition_rows" in inspect.signature(lsdb.from_dataframe).parameters else "threshold"
    left = lsdb.from_dataframe(pos, ra_column="coord_ra", dec_column="coord_dec", lowest_order=0, highest_order=8,
                               margin_threshold=None, **{key: 200_000})
    gaia = lsdb.open_catalog(UPath(args.gaia) if "://" not in args.gaia else args.gaia, columns=GAIA_COLS)

    from dask.distributed import Client

    with Client(n_workers=args.n_workers, threads_per_worker=1, memory_limit=args.memory_limit):
        x = left.crossmatch(gaia, radius_arcsec=args.radius, n_neighbors=1, how="left",
                            suffix_method="overlapping_columns", suffixes=("", "_gaia")).compute()
    x = x.rename(columns={"_dist_arcsec": "gaia_sep"}).reset_index(drop=True)
    flags = gaia_flags(x, args.snr)
    m = pd.concat([x[["objectId", "gaia_sep", "source_id"]].rename(columns={"source_id": "gaia_source_id"}), flags], axis=1)
    m = m.drop_duplicates("objectId")
    out = d.merge(m, on="objectId", how="left")
    for c in ("gaia_qso", "gaia_var", "gaia_star"):
        out[c] = out[c].fillna(False).astype(bool)
    out.to_parquet(Path(args.out).expanduser())

    matched = out.gaia_sep.notna()
    ext = out.get("refExtendedness", pd.Series(np.nan, index=out.index)) >= 0.5
    print(f"{len(out)} objects; Gaia match within {args.radius}\": {matched.sum()} ({matched.mean():.2f})")
    print(f"  gaia_star {out.gaia_star.sum()} ({out.gaia_star.mean():.2f}) | gaia_qso {out.gaia_qso.sum()} | "
          f"gaia_var {out.gaia_var.sum()}")
    print(f"  point-like: {(~ext).sum()}, of which gaia_star {(out.gaia_star & ~ext).sum()} "
          f"({(out.gaia_star & ~ext).sum() / max((~ext).sum(), 1):.2f}); extended gaia_star {(out.gaia_star & ext).sum()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
