"""Select variable objects in Rubin DP2 with LSDB.

Two sources of light curves:
  --source object (default): forced photometry on every coadd Object (object_collection).
      No DIA detection is required, so low-amplitude nuclear variability (dwarf AGN, TDEs)
      is kept; the bright-star veto runs against the same catalog.
  --source dia: forced photometry on DIA-detected objects (dia_object_collection), with an
      nDiaSources cut and optional matching to coadd Objects (--object-catalog). Use it for
      off-nuclear or hostless sources.

Run on the RSP (or another DP2 HATS site). Typical sequence:

    # 1. confirm column names
    python scripts/select_variables.py inspect

    # 2. statistics for every object in a test cone (no selection), to calibrate
    python scripts/select_variables.py run --cone 53.13 -28.10 1800 --no-select --stats-only \
        --out ~/rubin_imbh/test_cone_stats.parquet

    # 3. full-sky selection, split into order-3 HEALPix tiles (~54 deg^2 each); each tile
    #    is written to <out>/o3_p<pixel>/ and finished tiles are skipped on a rerun
    python scripts/select_variables.py tiles --order 3
    python scripts/select_variables.py run --err-scale 1.07 --tile-order 3 \
        --out ~/rubin_imbh/dp2_variables
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import hats
import lsdb
import numpy as np
import pandas as pd
from upath import UPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.pipeline import needed_columns, nested_fields, select_variables, tile_pixels  # noqa: E402
from jaxpz.lightcurve import DIA, OBJECT, Cuts  # noqa: E402

DEFAULT_CATALOG = {
    "object": "/rubin/lsdb_data/dp2/object_collection",
    "dia": "/rubin/lsdb_data/dp2/dia_object_collection",
}
PRESETS = {"object": OBJECT, "dia": DIA}


def parse_sys_frac(text: str):
    """``0.02`` for all bands, or ``g=0.03,r=0.02,i=0.02,z=0.02`` per band."""
    if "=" not in text:
        return float(text)
    return {k.strip(): float(v) for k, v in (item.split("=") for item in text.split(","))}


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


def build_columns(args):
    from dataclasses import replace

    cols = PRESETS[args.source]
    over = {k: v for k, v in (("lc", args.lc_column), ("flux", args.flux_column),
                              ("sci_flux", args.sci_flux_column)) if v}
    if "flux" in over:
        over["flux_err"] = over["flux"] + "Err"
    return replace(cols, **over)


def build_prefilter(args) -> str | None:
    parts = [args.prefilter] if args.prefilter else []
    if args.extended_only:
        parts.append("refExtendedness >= 0.5")
    return " and ".join(f"({p})" for p in parts) or None


def cmd_run(args):
    cols = build_columns(args)
    cuts = Cuts(
        min_dia_sources=args.min_dia_sources,
        min_epochs=args.min_epochs,
        min_sig=args.min_sig,
        min_bands=args.min_bands,
        err_scale=args.err_scale,
        sys_frac=args.sys_frac,
        select_bands=tuple(args.select_bands),
        robust=not args.no_robust,
        combo_bands=tuple(args.combo_bands),
        combined=args.combined,
        min_combo_dof=args.min_combo_dof,
        nightly=not args.no_nightly,
    )
    logging.info("source: %s   catalog: %s", args.source, args.catalog)
    logging.info("columns: %s", cols)
    logging.info("cuts: %s   prefilter: %s", cuts, build_prefilter(args))
    if args.source == "object" and args.object_catalog:
        sys.exit("--object-catalog is for --source dia; the object source vetoes against its own catalog")

    if args.cone and (args.tile or args.tile_order is not None):
        sys.exit("--cone cannot be combined with --tile/--tile-order")

    from dask.distributed import Client

    import contextlib

    from dask.distributed import performance_report

    report = performance_report(filename=str(Path(args.profile).expanduser())) if args.profile else contextlib.nullcontext()
    with Client(n_workers=args.n_workers, threads_per_worker=1, memory_limit=args.memory_limit) as client, report:
        logging.info("dask dashboard: %s", client.dashboard_link)
        # open once for the schema (no data read), then reopen with only the needed columns
        extra = ("diaSource",) if args.keep_dia_source and args.source == "dia" else ()
        if args.keep_columns:
            avail = set(getattr(open_dia(args.catalog), "all_columns", []))
            want = [c.strip() for c in args.keep_columns.split(",") if c.strip()]
            missing = [c for c in want if avail and c not in avail]
            if missing:
                logging.warning("--keep-columns not in catalog, skipped: %s", missing)
            extra = extra + tuple(c for c in want if c not in missing)
        columns = needed_columns(open_dia(args.catalog), cols, extra, keep_lc=not (args.stats_only or args.no_select))
        logging.info("loading columns: %s", columns)
        cat = open_dia(args.catalog, args.cone, columns)
        out = Path(args.out).expanduser()
        if args.source == "object":
            env = None if args.no_veto else load_environment(args, out, args.catalog)
        else:
            env = load_environment(args, out, args.object_catalog) if args.object_catalog else None

        if args.tile_order is None:
            tile = tuple(args.tile) if args.tile else None
            dest = out / f"o{tile[0]}_p{tile[1]}" if tile else out
            run_one(cat, cols, cuts, args, dest, tile, env)
            return

        tiles = tile_pixels(cat, args.tile_order)
        if args.min_abs_b:
            b = tile_galactic_latitude(tiles, args.tile_order)
            skipped = [t for t, bb in zip(tiles, b) if abs(bb) < args.min_abs_b]
            tiles = [t for t, bb in zip(tiles, b) if abs(bb) >= args.min_abs_b]
            logging.info("skipping %d tiles with |b| < %g deg at their centre: %s", len(skipped), args.min_abs_b, skipped)
        out.mkdir(parents=True, exist_ok=True)
        todo = [p for p in tiles if not (out / f"o{args.tile_order}_p{p}" / DONE).exists()]
        logging.info("%d tiles, %d already done, %d to run", len(tiles), len(tiles) - len(todo), len(todo))
        t_start = time.time()
        for i, pix in enumerate(todo, 1):
            dest = out / f"o{args.tile_order}_p{pix}"
            t0 = time.time()
            logging.info("[%d/%d] %s", i, len(todo), dest.name)
            run_one(cat, cols, cuts, args, dest, (args.tile_order, pix), env)
            per = (time.time() - t_start) / i
            logging.info("[%d/%d] %s took %.0f s; mean %.0f s/tile, ~%.1f h remaining",
                         i, len(todo), dest.name, time.time() - t0, per, per * (len(todo) - i) / 3600)


DONE = "_DONE"


def tile_galactic_latitude(tiles, order):
    """Galactic latitude (deg) of each HEALPix tile centre."""
    import astropy.units as u
    import cdshealpix.nested as cn
    import numpy as np
    from astropy.coordinates import SkyCoord

    lon, lat = cn.healpix_to_lonlat(np.asarray(tiles, dtype=np.uint64), order)
    return SkyCoord(lon, lat).galactic.b.to(u.deg).value


def load_environment(args, out: Path, object_catalog: str):
    """Object catalog plus a bright-star catalog with a margin wider than the veto radius.

    Bright stars are computed once (restricted to the cone, padded by the veto radius, when
    --cone is given) and cached as parquet so tiled runs and reruns reuse them.
    """
    from rubin_imbh.environment import (
        bright_star_catalog,
        bright_stars,
        require_crossmatch_support,
        resolve_object_columns,
    )

    require_crossmatch_support()
    oc = resolve_object_columns(lsdb.open_catalog(UPath(object_catalog)))
    objects = lsdb.open_catalog(UPath(object_catalog), columns=oc.load)
    logging.info("object columns: %s", oc)

    cache = Path(args.bright_stars).expanduser() if args.bright_stars else None
    if cache is None:
        base = out.parent if out.suffix == ".parquet" else out
        tag = f"_cone{args.cone[0]:g}_{args.cone[1]:g}_{args.cone[2]:g}" if args.cone else ""
        cache = base / f"bright_stars_r{args.bright_star_mag:g}{tag}.parquet"
    if cache.exists():
        stars = pd.read_parquet(cache)
        logging.info("read %d bright stars from %s", len(stars), cache)
    else:
        src = objects
        if args.cone:
            pad = args.cone[2] + 2 * args.veto_arcsec
            src = lsdb.open_catalog(UPath(object_catalog), columns=oc.load,
                                    search_filter=lsdb.ConeSearch(ra=args.cone[0], dec=args.cone[1], radius_arcsec=pad))
        logging.info("computing r < %g point sources from %s (one pass over the object catalog)",
                     args.bright_star_mag, object_catalog)
        stars = bright_stars(src, oc, args.bright_star_mag)
        cache.parent.mkdir(parents=True, exist_ok=True)
        stars.to_parquet(cache)
        logging.info("cached %d bright stars to %s", len(stars), cache)
    if args.cone:  # an all-sky cache would otherwise ship every star to the workers
        ra0, dec0 = np.radians(args.cone[0]), np.radians(args.cone[1])
        ra, dec = np.radians(stars[oc.ra].to_numpy(float)), np.radians(stars[oc.dec].to_numpy(float))
        cos_sep = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
        pad = np.radians((args.cone[2] + 2 * args.veto_arcsec) / 3600)
        stars = stars[cos_sep >= np.cos(pad)]
        logging.info("%d bright stars within the cone", len(stars))
    bright = bright_star_catalog(stars, oc, margin_arcsec=args.veto_arcsec + 5.0)
    return dict(objects=objects, oc=oc, bright=bright)


def run_one(cat, cols, cuts, args, dest: Path, tile, env=None):
    """Run one region and write it to ``dest``: a .parquet file, or a HATS directory.

    In tile mode ``dest`` is a directory holding the output plus a ``_DONE`` marker,
    written last so an interrupted tile is redone on the next run.
    """
    result = select_variables(cat, cols, cuts, select=not args.no_select, tile=tile, prefilter=build_prefilter(args),
                              chunk_objects=args.chunk_objects)
    if args.no_select or args.stats_only:
        result = result[[c for c in result.columns if c not in cat.nested_columns]]
    if env is not None:
        from rubin_imbh.environment import add_environment

        result = add_environment(result, env["objects"], env["oc"], env["bright"], match_arcsec=args.match_arcsec,
                                 veto_arcsec=args.veto_arcsec, keep_unmatched=args.keep_unmatched,
                                 match_objects=args.source == "dia")

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
    p.add_argument("--source", choices=["object", "dia"], default="object",
                   help="object: forced photometry on every coadd Object (default; nuclear AGN/TDEs); "
                        "dia: DIA-detected objects (off-nuclear/hostless)")
    p.add_argument("--catalog", help="HATS collection path (default: the DP2 collection for --source)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inspect", help="print the catalog schema")

    t = sub.add_parser("tiles", help="list HEALPix tiles covering the catalog")
    t.add_argument("--order", type=int, default=3)

    r = sub.add_parser("run", help="compute variability statistics and select variables")
    r.add_argument("--out", required=True, help="*.parquet (computed in memory) or a directory (HATS)")
    r.add_argument("--cone", nargs=3, type=float, metavar=("RA", "DEC", "RADIUS_ARCSEC"))
    r.add_argument("--tile", nargs=2, type=int, metavar=("ORDER", "PIXEL"), help="run one HEALPix tile")
    r.add_argument("--tile-order", type=int, help="loop over all tiles at this order, skipping done ones")
    r.add_argument("--min-abs-b", type=float, default=0.0,
                   help="with --tile-order: skip tiles whose centre is within this Galactic latitude (deg)")
    r.add_argument("--no-select", action="store_true", help="keep all objects (for calibration)")
    r.add_argument("--stats-only", action="store_true", help="drop the nested light curves from the output")
    r.add_argument("--prefilter", help="pandas query on top-level columns before any light curve is read, "
                   "e.g. 'r_psfFlux > 1000' (object source: host selection)")
    r.add_argument("--extended-only", action="store_true", help="add refExtendedness >= 0.5 to the prefilter")
    r.add_argument("--keep-dia-source", action="store_true", help="dia source: also keep the nested diaSource column")
    r.add_argument("--keep-columns", help="comma-separated extra top-level columns to carry into the output "
                   "(e.g. coadd total fluxes and errors for validate_real.py)")
    r.add_argument("--min-dia-sources", type=int, default=Cuts.min_dia_sources, help="dia source only")
    r.add_argument("--min-epochs", type=int, default=Cuts.min_epochs)
    r.add_argument("--min-sig", type=float, default=Cuts.min_sig)
    r.add_argument("--min-bands", type=int, default=Cuts.min_bands)
    r.add_argument("--err-scale", type=float, default=Cuts.err_scale)
    r.add_argument("--sys-frac", type=parse_sys_frac, default=Cuts.sys_frac,
                   help="error floor as a fraction of direct-image flux: 0.02 or g=0.03,r=0.02,...")
    r.add_argument("--select-bands", default="".join(Cuts.select_bands), help="bands that count toward selection")
    r.add_argument("--no-robust", action="store_true", help="select on varSig instead of varSigClip")
    r.add_argument("--combo-bands", default="".join(Cuts.combo_bands),
                   help="bands in the combined-band chi^2 for sparse (WFD) light curves")
    r.add_argument("--combined", choices=["either", "only", "off"], default=Cuts.combined,
                   help="either: per-band OR combined criterion; only: combined; off: per band only")
    r.add_argument("--no-nightly", action="store_true",
                   help="use individual epochs instead of nightly means (single-night artefacts then count)")
    r.add_argument("--min-combo-dof", type=int, default=Cuts.min_combo_dof,
                   help="minimum sum over combo bands of (epochs - 1)")
    r.add_argument("--lc-column", help="nested light-curve column (default: auto-detected)")
    r.add_argument("--flux-column", help=f"default {DIA.flux}; error column is <name>Err")
    r.add_argument("--sci-flux-column", help=f"default {DIA.sci_flux}")
    r.add_argument("--no-veto", action="store_true", help="object source: skip the bright-star veto")
    r.add_argument("--object-catalog", help="dia source: object_collection path to match coadd Objects and flag "
                   "bright-star neighbourhoods (e.g. /rubin/lsdb_data/dp2/object_collection; needs lsdb >= 0.10)")
    r.add_argument("--match-arcsec", type=float, default=1.0, help="DiaObject-Object match radius")
    r.add_argument("--keep-unmatched", action="store_true", help="keep variables with no Object match (objSep NaN)")
    r.add_argument("--veto-arcsec", type=float, default=10.0, help="bright-star neighbourhood radius")
    r.add_argument("--bright-star-mag", type=float, default=18.0, help="r-band limit for veto stars")
    r.add_argument("--bright-stars", help="bright-star parquet cache (default: next to --out)")
    r.add_argument("--chunk-objects", type=int, default=10_000,
                   help="objects per light-curve chunk within a partition (lower to cut peak memory)")
    r.add_argument("--profile", help="write a dask performance report (HTML) here: where the time goes")
    r.add_argument("--n-workers", type=int, default=4)
    r.add_argument("--memory-limit", default="6GB", help="per worker")

    args = p.parse_args(argv)
    args.catalog = args.catalog or DEFAULT_CATALOG[args.source]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"inspect": cmd_inspect, "tiles": cmd_tiles, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
