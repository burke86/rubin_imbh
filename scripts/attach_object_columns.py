"""Attach Object columns (default: cModel fluxes) to a selection, without recomputing statistics.

    python scripts/attach_object_columns.py ~/dp2_variables_v2_gaia.parquet \
        --out ~/dp2_variables_v2_cmodel.parquet

Crossmatches the selection (objectId, coord_ra, coord_dec) with object_collection within
--radius and keeps the match only where the objectId agrees, so the result is a pure lookup
by objectId. Adds --columns (default: ugrizy cModel flux and error, the coadd photometry the
amortized model uses) to every selected row; rows without a match get NaN.
"""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path

import lsdb
import pandas as pd
from upath import UPath

DEFAULT_COLUMNS = ",".join(f"{b}_cModelFlux,{b}_cModelFluxErr" for b in "ugrizy")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("selected", help="selection parquet with objectId, coord_ra, coord_dec")
    p.add_argument("--out", required=True)
    p.add_argument("--catalog", default="/rubin/lsdb_data/dp2/object_collection")
    p.add_argument("--columns", default=DEFAULT_COLUMNS, help="comma-separated Object columns to attach")
    p.add_argument("--radius", type=float, default=0.2, help="match radius (arcsec); objectIds must also agree")
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--memory-limit", default="16GB")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    d = pd.read_parquet(Path(args.selected).expanduser()).reset_index(drop=True)
    cols = [c for c in args.columns.split(",") if c]
    have = [c for c in cols if c in d]
    if have:
        logging.info("replacing columns already in the selection: %s", have)
        d = d.drop(columns=have)
    pos = d[["objectId", "coord_ra", "coord_dec"]].copy()
    key = "partition_rows" if "partition_rows" in inspect.signature(lsdb.from_dataframe).parameters else "threshold"
    left = lsdb.from_dataframe(pos, ra_column="coord_ra", dec_column="coord_dec", lowest_order=0, highest_order=8,
                               margin_threshold=None, **{key: 200_000})
    objects = lsdb.open_catalog(UPath(args.catalog), columns=["objectId", "coord_ra", "coord_dec", *cols])

    from dask.distributed import Client

    with Client(n_workers=args.n_workers, threads_per_worker=1, memory_limit=args.memory_limit):
        x = left.crossmatch(objects, radius_arcsec=args.radius, n_neighbors=3,  # blends: keep the same-id one
                            suffix_method="overlapping_columns", suffixes=("", "_obj")).compute()
    x = x.reset_index(drop=True)
    same = x["objectId"].to_numpy() == x["objectId_obj"].to_numpy()
    logging.info("%d matches within %.2f\", %d with the same objectId", len(x), args.radius, int(same.sum()))
    x = x[same].drop_duplicates("objectId")
    out = d.merge(x[["objectId", *cols]], on="objectId", how="left")
    out.to_parquet(Path(args.out).expanduser())
    found = out["objectId"].isin(x["objectId"])
    filled = ", ".join(f"{c} {out[c].notna().mean():.2f}" for c in cols)
    print(f"{len(out)} selected objects; {int(found.sum())} ({found.mean():.3f}) matched by objectId; wrote {args.out}")
    print(f"non-null fraction per column (missing coadds are NaN): {filled}")


if __name__ == "__main__":
    main()
