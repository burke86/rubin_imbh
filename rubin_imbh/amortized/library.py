"""Empirical survey library: real DP2 noise and cadence for the simulator.

Library format (one row per real Object; written by ``scripts/export_survey_library.py``):

  objectId, coord_ra, coord_dec      (absent with export --strip-ids; not used here)
  region, layer (e.g. "wfd", "ddf"), refExtendedness
  {b}_coadd_flux, {b}_coadd_err      coadd flux (cModel by default) and error, nJy
  {b}_night_mjd, {b}_night_err       lists: nightly epochs (mean MJD) and nightly-mean
                                     difference-flux errors (nJy), after flag cuts

A simulated object borrows the coadd errors, cadence and nightly errors of a library
object of similar r-band brightness in the requested layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = tuple("ugrizy")


def load_library(path) -> pd.DataFrame:
    lib = pd.read_parquet(path)
    missing = [c for b in BANDS for c in (f"{b}_coadd_flux", f"{b}_coadd_err", f"{b}_night_mjd", f"{b}_night_err")
               if c not in lib]
    if missing:
        raise KeyError(f"survey library lacks columns {missing}")
    return lib.reset_index(drop=True)


def mock_library(n: int, rng: np.random.Generator, layer: str = "wfd") -> pd.DataFrame:
    """Synthetic library for tests: sky-limited errors at a fixed depth, random cadences.

    ``layer="wfd"``: m5 ~ 24.5, 1-6 nights per band; ``"ddf"``: m5 ~ 26, 30-60 nights.
    """
    m5, (lo, hi) = (24.5, (1, 6)) if layer == "wfd" else (26.0, (30, 60))
    sigma5 = 10 ** ((31.4 - m5) / 2.5) / 5.0  # nJy, 1-sigma sky-limited coadd error
    rmag = rng.uniform(17.0, 26.5, n)
    rows = {"objectId": np.arange(n), "layer": layer, "refExtendedness": (rng.random(n) < 0.6).astype(float)}
    for b in BANDS:
        flux = 10 ** ((31.4 - (rmag + rng.normal(0, 0.3, n))) / 2.5)
        rows[f"{b}_coadd_flux"] = flux
        rows[f"{b}_coadd_err"] = np.sqrt(sigma5**2 + 0.004 * flux)
        k = rng.integers(lo, hi + 1, n)
        rows[f"{b}_night_mjd"] = [np.sort(60800.0 + rng.choice(250, kk, replace=False) + 0.2) for kk in k]
        # nightly errors: a single-visit-depth error, a few times the coadd error
        rows[f"{b}_night_err"] = [np.full(kk, np.sqrt((sigma5 * 3) ** 2 + 0.01 * f)) * rng.uniform(0.8, 1.2, kk)
                                  for kk, f in zip(k, flux)]
    return pd.DataFrame(rows)


def match_library(lib: pd.DataFrame, r_flux_njy: np.ndarray, rng: np.random.Generator,
                  layer: str | None = None, k: int = 25) -> np.ndarray:
    """Library row index for each simulated object: a random one of the ``k`` library objects
    nearest in log r-band coadd flux (within ``layer`` if given)."""
    pool = np.flatnonzero((lib["layer"] == layer).to_numpy()) if layer is not None else np.arange(len(lib))
    if len(pool) == 0:
        raise ValueError(f"no library objects in layer {layer!r}")
    ref = np.log10(np.clip(lib["r_coadd_flux"].to_numpy(float)[pool], 1e-3, None))
    order = np.argsort(ref)
    ref_sorted = ref[order]
    x = np.log10(np.clip(np.asarray(r_flux_njy, float), 1e-3, None))
    pos = np.searchsorted(ref_sorted, x)
    k = min(k, len(pool))
    start = np.clip(pos - k // 2, 0, len(pool) - k)
    pick = start + rng.integers(0, k, len(x))
    return pool[order[pick]]
