"""Observables for real objects, in the same form as the simulated training set.

``select_variables.py run --no-select --stats-only`` produces per-object statistics; this turns
them (plus coadd flux columns kept with ``--keep-columns``) into the columns
:func:`rubin_imbh.amortized.npe.build_features` expects: ``coadd_<b>``, ``coadd_err_<b>`` (nJy),
``<b>_nEpoch``, ``<b>_varSigClip``, ``<b>_rmsErr``, ``<b>_xsAbs`` (excess variance x median
flux^2, as in ``observe()``), ``all_varSigClip``, ``all_dof``.

For consistency with the simulator, the real statistics should come from a run with the
same floor as ``ObserveConfig.sys_frac`` (default 0.75% in r, i, z) and nightly binning.
Stars are not modelled: drop ``gaia_star`` objects (``scripts/gaia_flag.py``); ``pointlike``
marks the remaining point sources (quasars, or stars fainter than Gaia's G ~ 21).

The simulated coadd fluxes are total fluxes, so use a total-flux measurement (cModel /
GAaP / Kron) rather than ``psfFlux``, which misses extended host light.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

BANDS = tuple("ugrizy")
VAR_BANDS = tuple("griz")


def real_observables(stats: pd.DataFrame, flux_template: str = "{b}_psfFlux",
                     err_template: str = "{b}_psfFluxErr") -> pd.DataFrame:
    if "psfFlux" in flux_template:
        warnings.warn("using PSF fluxes as coadd photometry: the simulator models total fluxes, so extended "
                      "hosts will look too faint; prefer a total-flux column (--flux-template)", stacklevel=2)
    keep = [c for c in ("objectId", "coord_ra", "coord_dec", "refExtendedness", "envClean", "tile",
                        "gaia_star", "gaia_qso", "gaia_sep") if c in stats]
    out = stats[keep].copy()
    # point sources that Gaia does not confirm as stars: quasars, or stars too faint for Gaia
    ext = stats["refExtendedness"].to_numpy(float) if "refExtendedness" in stats else np.full(len(stats), np.nan)
    gstar = stats["gaia_star"].fillna(False).to_numpy(bool) if "gaia_star" in stats else np.zeros(len(stats), bool)
    out["pointlike"] = (ext < 0.5) & ~gstar
    for b in BANDS:
        fc, ec = flux_template.format(b=b), err_template.format(b=b)
        missing = [c for c in (fc, ec) if c not in stats]
        if missing:
            raise KeyError(f"stats lack {missing}; keep them with select_variables.py --keep-columns")
        out[f"coadd_{b}"] = stats[fc].to_numpy(float)
        out[f"coadd_err_{b}"] = stats[ec].to_numpy(float)
    for b in VAR_BANDS:
        out[f"{b}_nEpoch"] = stats[f"{b}_nEpoch"].to_numpy(float)
        out[f"{b}_varSigClip"] = stats[f"{b}_varSigClip"].to_numpy(float)
        out[f"{b}_rmsErr"] = stats[f"{b}_rmsErr"].to_numpy(float)
        out[f"{b}_xsAbs"] = (stats[f"{b}_excessVar"] * stats[f"{b}_medFlux"] ** 2).to_numpy(float)
    out["all_varSigClip"] = stats["all_varSigClip"].to_numpy(float)
    out["all_dof"] = stats["all_dof"].to_numpy(float)
    return out.reset_index(drop=True)
