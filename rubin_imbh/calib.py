"""Calibration groups for stats tables from either source."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jaxpz.lightcurve import clip_chi2_to_sigma


def calibration_groups(d: pd.DataFrame) -> tuple[pd.Series, str, str | None]:
    """Group label per row, plus the reference and likely-variable labels.

    dia source: DIA detection count (1-2 = mostly non-variable reference, >5 = mostly real
    variables). object source: no detection count, so morphology (point / extended); every
    Object is tested and nearly all are constant, so the reference is all of them and there is
    no likely-variable group.
    """
    if "nDiaSources" in d:
        g = pd.cut(d["nDiaSources"], [0, 2, 5, np.inf], labels=["1-2", "3-5", ">5"])
        return g, "1-2", ">5"
    if "refExtendedness" in d:
        g = pd.Series(np.where(d["refExtendedness"] >= 0.5, "extended",
                               np.where(d["refExtendedness"] < 0.5, "point", "unknown")), index=d.index)
    else:
        g = pd.Series("all", index=d.index)
    return g, None, None


def rescaled_sig(x: pd.DataFrame, b: str, s: float) -> np.ndarray:
    """``{b}_varSigClip`` as if a floor ``s`` x |medFlux| had been added in quadrature.

    Approximation (median flux and rms error instead of per-epoch values), validated against
    exact recomputation on mocks; the stats must have been computed with ``--sys-frac 0``.
    """
    n = x[f"{b}_nEpoch"].to_numpy(dtype=float)
    r2 = x[f"{b}_rmsErr"].to_numpy(dtype=float) ** 2
    f2 = (s * x[f"{b}_medFlux"].abs().to_numpy(dtype=float)) ** 2
    chi2c = x[f"{b}_chi2dofClip"].to_numpy(dtype=float) * (n - 2) * r2 / (r2 + f2)
    return clip_chi2_to_sigma(chi2c, n)
