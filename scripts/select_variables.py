"""Select variable DIA objects in Rubin DP2 with LSDB.

Run on the RSP (or another DP2 HATS site). Typical sequence:

    # 1. confirm column names in the DP2 collection
    python scripts/select_variables.py inspect

    # 2. statistics for every object in a test cone (no selection), to calibrate
    #    err_scale and the significance threshold on the non-variable population
    python scripts/select_variables.py run --cone 28.5 0.4 1800 --no-select \
        --out ~/rubin_imbh/test_cone_stats.parquet

    # 3. full-sky selection, split into order-3 HEALPix tiles (~54 deg^2 each); each tile
    #    is written to <out>/o3_p<pixel>/ and finished tiles are skipped on a rerun
    python scripts/select_variables.py tiles --order 3
    python scripts/select_variables.py run --err-scale 1.2 --tile-order 3 \
        --out ~/rubin_imbh/dp2_variables
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import hats
import lsdb
from upath import UPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.pipeline import needed_columns, nested_fields, select_variables, tile_pixels  # noqa: E402
from rubin_imbh.varstats import Columns, Cuts  # noqa: E402

DEFAULT_CATALOG = "/rubin/lsdb_data/dia_object_collection"


def open_dia(path: str, cone=None, columns=None):
    search = lsdb.ConeSearch(ra=cone[0], dec=cone[1], radius_arcsec=cone[2]) if cone else None
    return lsdb.open_catalog(UPath(path), search_filter=search, columns=columns)


def cmd_inspect(args):
    cat = open_dia(args.catalog)
    print(cat)
    print(f"\npartitions: {cat.npartitions}")
    print(f"default-loaded columns ({len(cat.columns)}):")
    for c in cat.columns:
        print(f"  {c:40s} {cat.dtypes[c]}")
    for nc in cat.nested_columns:
        print(f"\nnested column '{nc}' fields:")
        for f in nested_fields(cat, nc):
            print(f"  {f}")
    if hasattr(cat, "all_columns"):
        extra = [c for c in cat.all_columns if c not in cat.columns]
        print(f"\nother available columns, not loaded by default ({len(extra)}):")
        for c in extra:
            print(f"  {c}")


def cmd_run(args):
    cols = Columns(
        lc=args.lc_column,
        flux=args.flux_column,
        flux_err=args.flux_column + "Err",
        sci_flux=args.sci_flux_column,
    )
    cuts = Cuts(
        min_dia_sources=args.min_dia_sources,
        min_epochs=args.min_epochs,
        min_sig=args.min_sig,
        min_bands=args.min_bands,
        err_scale=args.err_scale,
        sys_frac=args.sys_frac,
    )
    logging.info("columns: %s", cols)
    logging.info("cuts: %s", cuts)

    if args.cone and (args.tile or args.tile_order is not None):
        sys.exit("--cone cannot be combined with --tile/--tile-order")

    from dask.distributed import Client

    with Client(n_workers=args.n_workers, threads_per_worker=1, memory_limit=args.memory_limit) as client:
        logging.info("dask dashboard: %s", client.dashboard_link)
        # open once for the schema (no data read), then reopen with only the needed columns
        extra = ("diaSource",) if args.keep_dia_source else ()
        columns = needed_columns(open_dia(args.catalog), cols, extra)
        logging.info("loading columns: %s", columns)
        cat = open_dia(args.catalog, args.cone, columns)
        out = Path(args.out).expanduser()

        if args.tile_order is None:
            tile = tuple(args.tile) if args.tile else None
            dest = out / f"o{tile[0]}_p{tile[1]}" if tile else out
            run_one(cat, cols, cuts, args, dest, tile)
            return

        tiles = tile_pixels(cat, args.tile_order)
        out.mkdir(parents=True, exist_ok=True)
        for i, pix in enumerate(tiles, 1):
            dest = out / f"o{args.tile_order}_p{pix}"
            if (dest / DONE).exists():
                logging.info("[%d/%d] %s already done, skipping", i, len(tiles), dest.name)
                continue
            logging.info("[%d/%d] %s", i, len(tiles), dest.name)
            run_one(cat, cols, cuts, args, dest, (args.tile_order, pix))


DONE = "_DONE"


def run_one(cat, cols, cuts, args, dest: Path, tile):
    """Run one region and write it to ``dest``: a .parquet file, or a HATS directory.

    In tile mode ``dest`` is a directory holding the output plus a ``_DONE`` marker,
    written last so an interrupted tile is redone on the next run.
    """
    result = select_variables(cat, cols, cuts, select=not args.no_select, tile=tile)
    if args.no_select or args.stats_only:
        result = result[[c for c in result.columns if c not in cat.nested_columns]]

    if tile is None and dest.suffix == ".parquet":
        df = result.compute()
        df.to_parquet(dest)
        logging.info("wrote %d rows to %s", len(df), dest)
        return

    if tile is not None and (args.no_select or args.stats_only):
        # light-curve-free tiles are small: one parquet file each, easy to concatenate
        dest.mkdir(parents=True, exist_ok=True)
        df = result.compute()
        df.to_parquet(dest / "stats.parquet")
        n = len(df)
    else:
        try:
            # a tile dir without _DONE is a leftover from an interrupted run: safe to replace
            result.to_hats(dest, catalog_name=dest.name, overwrite=tile is not None)
            written = hats.read_hats(dest)  # a collection (main + margin) in lsdb >= 0.7
            n = getattr(written, "main_catalog", written).catalog_info.total_rows
        except RuntimeError as err:
            if "empty" not in str(err):
                raise
            dest.mkdir(parents=True, exist_ok=True)  # no variables in this region; just mark done
            n = 0
    if tile is not None:
        (dest / DONE).write_text(f"{n}\n")
    logging.info("wrote %d rows to %s", n, dest)


def cmd_tiles(args):
    cat = open_dia(args.catalog)
    tiles = tile_pixels(cat, args.order)
    area = 41253.0 / (12 * 4**args.order)
    print(f"{len(tiles)} order-{args.order} tiles (~{area:.1f} deg^2 each) overlap {cat.npartitions} partitions")
    print(" ".join(map(str, tiles)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=DEFAULT_CATALOG, help="path to the DP2 dia_object_collection")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inspect", help="print the catalog schema")

    t = sub.add_parser("tiles", help="list HEALPix tiles covering the catalog")
    t.add_argument("--order", type=int, default=3)

    r = sub.add_parser("run", help="compute variability statistics and select variables")
    r.add_argument("--out", required=True, help="*.parquet (computed in memory) or a directory (HATS)")
    r.add_argument("--cone", nargs=3, type=float, metavar=("RA", "DEC", "RADIUS_ARCSEC"))
    r.add_argument("--tile", nargs=2, type=int, metavar=("ORDER", "PIXEL"), help="run one HEALPix tile")
    r.add_argument("--tile-order", type=int, help="loop over all tiles at this order, skipping done ones")
    r.add_argument("--no-select", action="store_true", help="keep all objects (for calibration)")
    r.add_argument("--stats-only", action="store_true", help="drop the nested light curves from the output")
    r.add_argument("--keep-dia-source", action="store_true", help="also load and keep the nested diaSource column")
    r.add_argument("--min-dia-sources", type=int, default=Cuts.min_dia_sources)
    r.add_argument("--min-epochs", type=int, default=Cuts.min_epochs)
    r.add_argument("--min-sig", type=float, default=Cuts.min_sig)
    r.add_argument("--min-bands", type=int, default=Cuts.min_bands)
    r.add_argument("--err-scale", type=float, default=Cuts.err_scale)
    r.add_argument("--sys-frac", type=float, default=Cuts.sys_frac)
    r.add_argument("--lc-column", default=Columns.lc)
    r.add_argument("--flux-column", default=Columns.flux, help="error column is <name>Err")
    r.add_argument("--sci-flux-column", default=Columns.sci_flux)
    r.add_argument("--n-workers", type=int, default=4)
    r.add_argument("--memory-limit", default="6GB", help="per worker")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"inspect": cmd_inspect, "tiles": cmd_tiles, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
