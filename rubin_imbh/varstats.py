"""Per-object, per-band variability statistics from nested DIA forced photometry.

Everything here is plain pandas/numpy so it can be unit-tested without LSDB and
applied partition-by-partition with ``Catalog.map_partitions``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BANDS = tuple("ugrizy")
STATS = ("nEpoch", "chi2dof", "varSig", "excessVar", "fvar")


@dataclass(frozen=True)
class Columns:
    """Column names in the DP2 ``dia_object_collection``.

    Defaults follow the DP1/DP2 ForcedSourceOnDiaObject schema; confirm them with
    ``scripts/select_variables.py inspect`` before a full run.
    """

    lc: str = "diaObjectForcedSource"  # nested forced-photometry light curve
    band: str = "band"
    flux: str = "psfDiffFlux"  # difference-image flux [nJy]
    flux_err: str = "psfDiffFluxErr"
    sci_flux: str = "psfFlux"  # direct-image flux, used to normalize excess variance
    n_dia_sources: str = "nDiaSources"
    flags: tuple[str, ...] = field(
        default=(
            "diff_PixelFlags_nodataCenter",
            "psfDiffFlux_flag",
            "psfFlux_flag",
            "invalidPsfFlag",
            "pixelFlags_bad",
            "pixelFlags_edge",
            "pixelFlags_saturatedCenter",
            "pixelFlags_crCenter",
            "pixelFlags_suspectCenter",
            "pixelFlags_interpolatedCenter",
        )
    )


@dataclass(frozen=True)
class Cuts:
    min_dia_sources: int = 3  # coarse cut on the DiaObject summary row
    min_epochs: int = 10  # good forced-photometry epochs required in a band
    min_sig: float = 5.0  # chi^2 significance (Gaussian sigma) for "variable" in a band
    min_bands: int = 1  # number of bands that must pass min_sig
    err_scale: float = 1.0  # multiplicative error rescaling (calibrate on a test field)
    sys_frac: float = 0.0  # systematic floor as a fraction of |direct-image flux|


def chi2_to_sigma(chi2: np.ndarray, dof: np.ndarray) -> np.ndarray:
    """Gaussian-equivalent significance of chi^2 exceeding its expectation.

    Wilson-Hilferty cube-root transform: accurate for dof >~ 5 and, unlike
    ``norm.isf(chi2.sf(...))``, does not saturate for very large chi^2.
    """
    chi2 = np.asarray(chi2, dtype=float)
    dof = np.asarray(dof, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = 2.0 / (9.0 * dof)
        z = (np.cbrt(chi2 / dof) - (1.0 - a)) / np.sqrt(a)
    return np.where(dof > 0, z, np.nan)


def good_epoch_mask(flat: pd.DataFrame, cols: Columns, flags: tuple[str, ...]) -> np.ndarray:
    f = flat[cols.flux].to_numpy(dtype=float, na_value=np.nan)
    e = flat[cols.flux_err].to_numpy(dtype=float, na_value=np.nan)
    good = np.isfinite(f) & np.isfinite(e) & (e > 0)
    for flag in flags:
        good &= ~flat[flag].fillna(False).to_numpy(dtype=bool)
    return good


def lightcurve_stats(
    obj: np.ndarray,
    flat: pd.DataFrame,
    n_obj: int,
    cols: Columns,
    cuts: Cuts,
    flags: tuple[str, ...] = (),
    bands: tuple[str, ...] = BANDS,
) -> pd.DataFrame:
    """Variability statistics for each object and band.

    Parameters
    ----------
    obj
        Integer position (0..n_obj-1) of the parent object for each row of ``flat``.
        Positions rather than index labels, because the HEALPix index of an LSDB
        partition is not guaranteed unique.
    flat
        One row per epoch with the band, flux, error, direct flux and flag columns.

    Returns
    -------
    DataFrame with ``n_obj`` rows and columns ``{band}_{stat}`` for ``STATS``.
    """
    good = good_epoch_mask(flat, cols, flags)
    f = flat[cols.flux].to_numpy(dtype=float, na_value=np.nan)[good]
    e = cuts.err_scale * flat[cols.flux_err].to_numpy(dtype=float, na_value=np.nan)[good]
    s = flat[cols.sci_flux].to_numpy(dtype=float, na_value=np.nan)[good]
    if cuts.sys_frac > 0:
        e = np.hypot(e, cuts.sys_frac * np.nan_to_num(np.abs(s)))
    e2 = e**2
    w = 1.0 / e2
    df = pd.DataFrame(
        {
            "obj": obj[good],
            "band": flat[cols.band].to_numpy()[good],
            "f": f,
            "e2": e2,
            "s": s,
            "w": w,
            "wf": w * f,
            "wf2": w * f * f,
        }
    )
    df = df[df["band"].isin(bands)]

    g = df.groupby(["obj", "band"], sort=False, observed=True)
    agg = g.agg(
        n=("f", "size"),
        sw=("w", "sum"),
        swf=("wf", "sum"),
        swf2=("wf2", "sum"),
        var_f=("f", "var"),
        mean_e2=("e2", "mean"),
        med_s=("s", "median"),
    )
    n = agg["n"].to_numpy(dtype=float)
    dof = n - 1.0
    # chi^2 about the weighted mean, written in moment form so it groups cleanly
    chi2 = np.maximum(agg["swf2"] - agg["swf"] ** 2 / agg["sw"], 0.0).to_numpy()
    xs = ((agg["var_f"] - agg["mean_e2"]) / agg["med_s"] ** 2).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame(
            {
                "nEpoch": agg["n"].to_numpy(),
                "chi2dof": np.where(dof > 0, chi2 / dof, np.nan),
                "varSig": chi2_to_sigma(chi2, dof),
                "excessVar": xs,
                "fvar": np.sqrt(np.clip(xs, 0.0, None)),
            },
            index=agg.index,
        )

    wide = out.unstack("band")
    wide.columns = [f"{b}_{stat}" for stat, b in wide.columns]
    wide = wide.reindex(index=np.arange(n_obj), columns=[f"{b}_{s}" for b in bands for s in STATS])
    nep = [f"{b}_nEpoch" for b in bands]
    wide[nep] = wide[nep].fillna(0).astype("int32")
    return wide


def summarize(stats: pd.DataFrame, cuts: Cuts, bands: tuple[str, ...] = BANDS) -> pd.DataFrame:
    """Cross-band summary: number of significantly variable bands and best significance."""
    sig = np.column_stack([stats[f"{b}_varSig"].to_numpy(dtype=float) for b in bands])
    nep = np.column_stack([stats[f"{b}_nEpoch"].to_numpy() for b in bands])
    usable = nep >= cuts.min_epochs
    sig = np.where(usable, sig, np.nan)
    with np.errstate(invalid="ignore"):
        n_var = np.sum(sig >= cuts.min_sig, axis=1)
    max_sig = np.full(len(stats), np.nan)
    any_usable = usable.any(axis=1)
    max_sig[any_usable] = np.nanmax(sig[any_usable], axis=1)
    return pd.DataFrame(
        {"nVarBands": n_var.astype("int32"), "maxVarSig": max_sig, "isVariable": n_var >= cuts.min_bands},
        index=stats.index,
    )
