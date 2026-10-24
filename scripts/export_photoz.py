"""Export DP2 photo-z (object_photoz) for a cone, for comparison in validate_real.py.

    python scripts/export_photoz.py --cone 150.1 2.2 1800 --out ~/cosmos_photoz.parquet

Writes objectId plus every top-level object_photoz column in the cone, and prints the column
names so the redshift estimate can be chosen with validate_real.py --photoz-column.
"""

import argparse
from pathlib import Path

import lsdb
from upath import UPath


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="/rubin/lsdb_data/dp2/object_photoz")
    p.add_argument("--cone", nargs=3, type=float, required=True, metavar=("RA", "DEC", "RADIUS_ARCSEC"))
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    full = lsdb.open_catalog(UPath(args.catalog))
    cols = [c for c in getattr(full, "all_columns", full.columns) if c not in full.nested_columns]
    print("object_photoz columns:", cols)
    cat = lsdb.open_catalog(UPath(args.catalog), columns=cols,
                            search_filter=lsdb.ConeSearch(ra=args.cone[0], dec=args.cone[1], radius_arcsec=args.cone[2]))
    df = cat.compute().reset_index(drop=True)
    df.to_parquet(Path(args.out).expanduser())
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
