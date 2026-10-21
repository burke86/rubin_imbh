"""Host and environment cuts: match variables to coadd Objects and flag bright-star surroundings.

ECDFS calibration showed most false positives are DiaObjects with no coadd Object within
1" (noise peaks, artifacts) or within ~10" of an r < 18 point source (halos, spikes,
residuals). Both are cut here with LSDB crossmatches, which needs lsdb >= 0.10.

The bright-star veto radius usually exceeds object_collection's margin, which LSDB refuses
to crossmatch beyond, so bright stars (sparse: ~10^6 over DP2) are pulled into memory once
and rebuilt as a small catalog with a margin larger than the veto radius.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import lsdb
import numpy as np
import pandas as pd

from .visits import mag_to_njy


@dataclass(frozen=True)
class ObjectColumns:
    ra: str
    dec: str
    ext: str | None
    fluxes: tuple[str, ...]
    obj_id: str | None

    @property
    def load(self) -> list[str]:
        return [c for c in dict.fromkeys([self.obj_id, self.ra, self.dec, self.ext, *self.fluxes]) if c]


def require_crossmatch_support():
    params = inspect.signature(lsdb.catalog.Catalog.crossmatch).parameters
    missing = [p for p in ("how", "suffix_method") if p not in params]
    if missing:
        raise RuntimeError(f"environment cuts need lsdb >= 0.10 (crossmatch lacks {missing}); have {lsdb.__version__}")


def resolve_object_columns(objects) -> ObjectColumns:
    """Pick object_collection column names (DP2: coord_ra/dec, refExtendedness, {band}_psfFlux)."""
    allc = list(getattr(objects, "all_columns", objects.columns))
    info = objects.hc_structure.catalog_info
    ext = next((c for c in ("refExtendedness", "r_extendedness", "i_extendedness") if c in allc), None)
    if ext is None:
        ext = next((c for c in allc if "xtendedness" in c.lower()), None)
    fluxes = tuple(f"{b}_psfFlux" for b in "ugrizy" if f"{b}_psfFlux" in allc)
    obj_id = next((c for c in ("objectId", "id") if c in allc), None)
    return ObjectColumns(info.ra_column, info.dec_column, ext, fluxes, obj_id)


def bright_stars(objects, oc: ObjectColumns, mag: float = 18.0) -> pd.DataFrame:
    """Positions of point sources brighter than ``mag`` in r (i if r is absent)."""
    flux = next((f for f in ("r_psfFlux", "i_psfFlux") if f in oc.fluxes), None)
    if flux is None or oc.ext is None:
        raise KeyError(f"need an r/i psfFlux and an extendedness column; have {oc}")
    q = objects[[oc.ra, oc.dec, oc.ext, flux]].query(f"{flux} > {mag_to_njy(mag)} and {oc.ext} < 0.5")
    return q.compute()[[oc.ra, oc.dec]].reset_index(drop=True)


def bright_star_catalog(stars: pd.DataFrame, oc: ObjectColumns, margin_arcsec: float):
    key = "partition_rows" if "partition_rows" in inspect.signature(lsdb.from_dataframe).parameters else "threshold"
    return lsdb.from_dataframe(
        stars, ra_column=oc.ra, dec_column=oc.dec, margin_threshold=margin_arcsec,
        lowest_order=0, highest_order=7, **{key: 500_000},
    )


def add_environment(cat, objects, oc: ObjectColumns, bright, match_arcsec: float = 1.0,
                    veto_arcsec: float = 10.0, keep_unmatched: bool = False, match_objects: bool = True):
    """Attach the nearest Object within ``match_arcsec`` and the distance to the nearest bright
    star within ``veto_arcsec``.

    Adds ``objSep`` (NaN if unmatched; unmatched rows are dropped unless ``keep_unmatched``),
    the Object columns, ``brightStarDist`` (NaN if no bright star within the veto radius) and
    ``envClean`` (matched and not near a bright star). With ``match_objects=False`` (the input
    already is object_collection) only the bright-star step runs; a bright star's own row is
    flagged at distance ~0.
    """
    require_crossmatch_support()
    if match_objects:
        x = cat.crossmatch(
            objects[oc.load], radius_arcsec=match_arcsec, n_neighbors=1, how="left" if keep_unmatched else "inner",
            suffix_method="overlapping_columns", suffixes=("", "_obj"),
        ).rename(columns={"_dist_arcsec": "objSep"})
    else:
        x = cat
    y = x.crossmatch(
        bright, radius_arcsec=veto_arcsec, n_neighbors=1, how="left",
        suffix_method="overlapping_columns", suffixes=("", "_bright"),
    ).rename(columns={"_dist_arcsec": "brightStarDist"})
    y = y[[c for c in y.columns if not c.endswith("_bright")]]

    def flag(df):
        clean = df["brightStarDist"].isna().to_numpy()
        if match_objects:
            clean &= df["objSep"].notna().to_numpy()
        return df.assign(envClean=clean)

    return y.map_partitions(flag, meta=flag(y.meta))
