"""LSDB wrapper: attach variability statistics to a DIA object catalog and select variables."""

from __future__ import annotations

import logging
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from .varstats import BANDS, Columns, Cuts, lightcurve_stats, summarize

log = logging.getLogger(__name__)


def nested_fields(catalog, column: str) -> list[str]:
    return list(catalog.dtypes[column].field_names)


# top-level and light-curve fields carried into the output alongside what the statistics need
# (host morphology and coadd fluxes exist only in object_collection)
KEEP_TOP = ("tract", "refExtendedness", *(f"{b}_psfFlux" for b in "ugrizy"))
KEEP_LC = ("midpointMjdTai", "visit", "psfFluxErr")


def available_columns(catalog) -> list[str]:
    """All top-level columns, including those not loaded by default."""
    return list(getattr(catalog, "all_columns", catalog.columns))


def needed_columns(catalog, cols: Columns, extra: tuple[str, ...] = (), keep_lc: bool = True) -> list[str]:
    """Columns to load, with nested fields as ``"lc.field"``.

    The DP2 collection is ~1.1 TB, mostly nested data the pipeline never touches (the
    whole ``diaSource`` column and most forced-photometry fields), so loading only these
    cuts I/O substantially. ``catalog`` should be opened without ``columns`` so its full
    schema is visible; nothing is read either way. ``keep_lc=False`` (stats-only runs) skips
    the light-curve fields that are only carried into the output.
    """
    cols = resolve_columns(catalog, cols)
    info = catalog.hc_structure.catalog_info
    allc = available_columns(catalog)
    want = (cols.obj_id, info.ra_column, info.dec_column, *KEEP_TOP, cols.n_dia_sources)
    top = [c for c in dict.fromkeys(want) if c and c in allc]
    fields = nested_fields(catalog, cols.lc)
    lc = [cols.band, cols.flux, cols.flux_err, cols.sci_flux, *cols.flags]
    if cols.time in fields:
        lc.append(cols.time)  # needed for nightly binning, not only for the output
    if keep_lc:
        lc += [f for f in KEEP_LC if f in fields]
    return top + [f"{cols.lc}.{f}" for f in dict.fromkeys(lc)] + list(extra)


def resolve_columns(catalog, cols: Columns) -> Columns:
    """Check the configured columns against the catalog schema.

    If ``cols.lc`` is not a nested column, the nested column holding ``cols.flux`` is used
    instead. Missing required columns raise; missing flag columns are dropped with a warning,
    since the DP2 forced-photometry schema is slimmer than DP1's.
    """
    nested = list(catalog.nested_columns)
    if cols.lc not in nested:
        hits = [nc for nc in nested if cols.flux in nested_fields(catalog, nc)]
        if not hits:
            raise KeyError(
                f"no nested column has '{cols.flux}' (difference-image forced photometry); nested "
                f"columns: { {nc: nested_fields(catalog, nc) for nc in nested} }"
            )
        if len(hits) > 1:
            raise KeyError(f"several nested columns have '{cols.flux}': {hits}; pass --lc-column")
        warnings.warn(f"light-curve column '{cols.lc}' not found; using '{hits[0]}'", stacklevel=2)
        cols = replace(cols, lc=hits[0])
    if cols.n_dia_sources is not None and cols.n_dia_sources not in available_columns(catalog):
        raise KeyError(f"'{cols.n_dia_sources}' not in catalog; for object_collection use the object source")
    fields = nested_fields(catalog, cols.lc)
    missing = [c for c in (cols.band, cols.flux, cols.flux_err, cols.sci_flux) if c not in fields]
    if missing:
        raise KeyError(f"Nested columns {missing} not in {cols.lc}; available: {fields}")
    absent = [f for f in cols.flags if f not in fields]
    if absent:
        warnings.warn(f"Flag columns not in {cols.lc}, skipping: {absent}", stacklevel=2)
    return replace(cols, flags=tuple(f for f in cols.flags if f in fields))


def tile_pixels(catalog, order: int) -> list[int]:
    """HEALPix pixels at ``order`` that overlap the catalog's partitions."""
    pix = set()
    for hp in catalog.get_healpix_pixels():
        if hp.order >= order:
            pix.add(hp.pixel >> (2 * (hp.order - order)))
        else:
            shift = 2 * (order - hp.order)
            pix.update(range(hp.pixel << shift, (hp.pixel + 1) << shift))
    return sorted(pix)


def in_tile(nf, tile: tuple[int, int]) -> np.ndarray:
    """Rows whose ``_healpix_29`` index falls in ``tile = (order, pixel)``."""
    order, pixel = tile
    return (nf.index.to_numpy().astype(np.uint64) >> np.uint64(2 * (29 - order))) == pixel


CHUNK_OBJECTS = 10_000


def add_variability(nf, cols: Columns, cuts: Cuts, select: bool = True, bands=BANDS, tile=None,
                    chunk_objects: int = CHUNK_OBJECTS):
    """Partition function: append ``{band}_{stat}`` and summary columns, optionally filter.

    Light curves are flattened ``chunk_objects`` objects at a time, and only the fields the
    statistics use, so peak memory stays bounded for dense object_collection partitions.
    """
    if tile is not None:
        # a partition coarser than the tile spans several tiles; keep only this tile's rows
        nf = nf.iloc[np.flatnonzero(in_tile(nf, tile))]
    present = set(nf[cols.lc].nest.fields) if hasattr(nf[cols.lc].nest, "fields") else set()
    fields = [cols.band, cols.flux, cols.flux_err, cols.sci_flux, *cols.flags]
    if cols.time in present:
        fields.append(cols.time)
    fields = list(dict.fromkeys(fields))
    parts = []
    for a in range(0, max(len(nf), 1), chunk_objects):
        lc = nf[cols.lc].iloc[a : a + chunk_objects]
        lengths = np.asarray(lc.nest.list_lengths)
        obj = np.repeat(np.arange(len(lc)), lengths)
        flat = lc.nest.to_flat(fields)  # positional: the keyword differs across nested-pandas versions
        parts.append(lightcurve_stats(obj, flat, len(lc), cols, cuts, flags=cols.flags, bands=bands))
        del flat
    stats = pd.concat(parts, ignore_index=True)
    stats = pd.concat([stats, summarize(stats, cuts)], axis=1)
    stats.index = nf.index
    out = nf.assign(**{c: stats[c] for c in stats.columns})
    if select:
        # positional take: an empty mask would otherwise be read as a column selector
        out = out.iloc[np.flatnonzero(stats["isVariable"].to_numpy(dtype=bool))]
    return out


def select_variables(
    catalog, cols: Columns = Columns(), cuts: Cuts = Cuts(), select: bool = True, tile=None,
    prefilter: str | None = None, chunk_objects: int = CHUNK_OBJECTS,
):
    """Lazy LSDB pipeline: cheap row cuts, then light-curve statistics per partition.

    ``prefilter`` is a pandas query on top-level columns (e.g. host selection such as
    ``"refExtendedness >= 0.5"``); for DIA catalogs ``nDiaSources >= min_dia_sources`` is
    applied as well. ``tile = (order, pixel)`` restricts the run to one HEALPix pixel, so a
    full-sky run can be split into independently restartable pieces.
    """
    cols = resolve_columns(catalog, cols)
    if tile is not None:
        catalog = catalog.pixel_search([tuple(tile)])
    coarse = catalog
    if prefilter:
        coarse = coarse.query(prefilter)
    if cols.n_dia_sources is not None:
        coarse = coarse.query(f"{cols.n_dia_sources} >= {cuts.min_dia_sources}")
    kw = dict(cols=cols, cuts=cuts, select=select, tile=tile, chunk_objects=chunk_objects)
    # dask cannot infer the new columns from an empty frame on its own, so build meta explicitly
    empty = coarse.meta if hasattr(coarse, "meta") else coarse._ddf._meta  # public in lsdb >= 0.7
    meta = add_variability(empty, **kw)
    return coarse.map_partitions(add_variability, meta=meta, **kw)
