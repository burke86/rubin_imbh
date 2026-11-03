"""Match a list of target positions (e.g. DESI spectra) to DP2 coadd Objects, on the RSP.

    python scripts/match_targets.py ~/desi_dr1_reference_positions.parquet --out ~/desi_object_ids.parquet

The targets need ra and dec (deg) and an id column (--id, default targetid). Each target gets its nearest
Object within --radius; the output has the target id, objectId, the Object position and the separation
(arcsec), one row per matched target. Feed the objectIds to
``select_variables.py run --object-ids ... --no-select --stats-only`` to compute light-curve statistics
for only those Objects.
"""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path

import lsdb
import pandas as pd
from upath import UPath


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", help="parquet with ra, dec and --id")
    p.add_argument("--out", required=True)
    p.add_argument("--id", default="targetid")
    p.add_argument("--catalog", default="/rubin/lsdb_data/dp2/object_collection")
    p.add_argument("--radius", type=float, default=1.0, help="match radius (arcsec)")
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--memory-limit", default="8GB")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    t = pd.read_parquet(Path(args.targets).expanduser())[[args.id, "ra", "dec"]]
    key = "partition_rows" if "partition_rows" in inspect.signature(lsdb.from_dataframe).parameters else "threshold"
    left = lsdb.from_dataframe(t, ra_column="ra", dec_column="dec", lowest_order=0, highest_order=8,
                               margin_threshold=None, **{key: 200_000})
    objects = lsdb.open_catalog(UPath(args.catalog), columns=["objectId", "coord_ra", "coord_dec"])

    from dask.distributed import Client

    with Client(n_workers=args.n_workers, threads_per_worker=1, memory_limit=args.memory_limit):
        x = left.crossmatch(objects, radius_arcsec=args.radius, n_neighbors=1,
                            suffix_method="overlapping_columns", suffixes=("", "_obj")).compute()
    x = x.reset_index(drop=True).rename(columns={"_dist_arcsec": "sep_arcsec"})
    keep = [args.id, "objectId", "coord_ra", "coord_dec", "sep_arcsec"]
    x = x[[c for c in keep if c in x]].sort_values("sep_arcsec").drop_duplicates("objectId")
    x.to_parquet(Path(args.out).expanduser())
    print(f"{len(t)} targets, {len(x)} matched to a distinct Object within {args.radius}\"; wrote {args.out}")


if __name__ == "__main__":
    main()
