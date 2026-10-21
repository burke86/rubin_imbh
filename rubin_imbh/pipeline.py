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
KEEP_TOP = ("diaObjectId", "ra", "dec", "tract")
KEEP_LC = ("midpointMjdTai", "visit", "psfFluxErr")


def needed_columns(catalog, cols: Columns, extra: tuple[str, ...] = ()) -> list[str]:
    """Columns to load, with nested fields as ``"lc.field"``.

    The DP2 collection is ~1.1 TB, mostly nested data the pipeline never touches (the
    whole ``diaSource`` column and most forced-photometry fields), so loading only these
    cuts I/O substantially. ``catalog`` should be opened without ``columns`` so its full
    schema is visible; nothing is read either way.
    """
    cols = resolve_columns(catalog, cols)
    top = [c for c in (*KEEP_TOP, cols.n_dia_sources) if c in catalog.columns]
    fields = nested_fields(catalog, cols.lc)
    lc = [cols.band, cols.flux, cols.flux_err, cols.sci_flux, *cols.flags]
    lc += [f for f in KEEP_LC if f in fields]
    return top + [f"{cols.lc}.{f}" for f in dict.fromkeys(lc)] + list(extra)


def resolve_columns(catalog, cols: Columns) -> Columns:
    """Check the configured columns against the catalog schema.

    Missing required columns raise; missing flag columns are dropped with a warning,
    since the DP2 forced-photometry schema is slimmer than DP1's.
    """
    top = list(catalog.columns)
    missing = [c for c in (cols.lc, cols.n_dia_sources) if c not in top]
    if missing:
        raise KeyError(f"Top-level columns {missing} not in catalog; available: {top}")
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


def add_variability(nf, cols: Columns, cuts: Cuts, select: bool = True, bands=BANDS, tile=None):
    """Partition function: append ``{band}_{stat}`` and summary columns, optionally filter."""
    if tile is not None:
        # a partition coarser than the tile spans several tiles; keep only this tile's rows
        nf = nf.iloc[np.flatnonzero(in_tile(nf, tile))]
    lc = nf[cols.lc]
    lengths = np.asarray(lc.nest.list_lengths)
    obj = np.repeat(np.arange(len(nf)), lengths)
    flat = lc.nest.to_flat()
    stats = lightcurve_stats(obj, flat, len(nf), cols, cuts, flags=cols.flags, bands=bands)
    stats = pd.concat([stats, summarize(stats, cuts)], axis=1)
    stats.index = nf.index
    out = nf.assign(**{c: stats[c] for c in stats.columns})
    if select:
        # positional take: an empty mask would otherwise be read as a column selector
        out = out.iloc[np.flatnonzero(stats["isVariable"].to_numpy(dtype=bool))]
    return out


def select_variables(
    catalog, cols: Columns = Columns(), cuts: Cuts = Cuts(), select: bool = True, tile=None
):
    """Lazy LSDB pipeline: coarse DiaObject cut, then light-curve statistics per partition.

    ``tile = (order, pixel)`` restricts the run to one HEALPix pixel, so a full-sky run can
    be split into independently restartable pieces.
    """
    cols = resolve_columns(catalog, cols)
    if tile is not None:
        catalog = catalog.pixel_search([tuple(tile)])
    coarse = catalog.query(f"{cols.n_dia_sources} >= {cuts.min_dia_sources}")
    kw = dict(cols=cols, cuts=cuts, select=select, tile=tile)
    # dask cannot infer the new columns from an empty frame on its own, so build meta explicitly
    empty = coarse.meta if hasattr(coarse, "meta") else coarse._ddf._meta  # public in lsdb >= 0.7
    meta = add_variability(empty, **kw)
    return coarse.map_partitions(add_variability, meta=meta, **kw)
