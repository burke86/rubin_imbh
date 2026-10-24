"""Reference redshifts (and X-ray luminosities) for validating posteriors on real objects.

Each builder returns a DataFrame with ``ra``, ``dec``, ``z_ref``, ``z_type`` (``spec`` or
``photo``) and, where available, ``logLX`` (log10 absorption-corrected 0.5-7 keV luminosity in
erg/s) and ``is_xray_agn``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def cosmos2020_reference(path, r_max: float = 25.5) -> pd.DataFrame:
    """COSMOS2020 CLASSIC LePhare photo-z (~35 bands): an accurate reference for r <~ 24."""
    d = pd.read_parquet(Path(path).expanduser())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 23.9 - 2.5 * np.log10(d["hsc_r_flux_auto"].to_numpy(float))
    keep = r < r_max
    return pd.DataFrame({"ra": d["ra"][keep].to_numpy(), "dec": d["dec"][keep].to_numpy(),
                         "z_ref": d["lp_zbest"][keep].to_numpy(), "z_type": "photo",
                         "z_ref_lo": d["lp_zpdf_l68"][keep].to_numpy(), "z_ref_hi": d["lp_zpdf_u68"][keep].to_numpy()})


def cdfs_xray_reference() -> pd.DataFrame:
    """CDF-S 7 Ms sources (Luo et al. 2017): adopted redshift (spec-z where available), L_X, class."""
    from astroquery.vizier import Vizier

    cols = ["RAJ2000", "DEJ2000", "zF", "r_zF", "zspec", "q_zspec", "LXc", "OType"]
    t = Vizier(columns=cols, row_limit=-1).get_catalogs("J/ApJS/228/2/sources")[0].to_pandas()
    is_spec = t["r_zF"].astype(str).str.strip().str.lower().str.startswith("zspec")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lx = np.log10(t["LXc"].to_numpy(float))
    return pd.DataFrame({"ra": t["RAJ2000"].to_numpy(float), "dec": t["DEJ2000"].to_numpy(float),
                         "z_ref": t["zF"].to_numpy(float), "z_type": np.where(is_spec, "spec", "photo"),
                         "logLX": log_lx, "is_xray_agn": t["OType"].astype(str).str.strip().eq("AGN").to_numpy()})


def match_reference(objects: pd.DataFrame, ref: pd.DataFrame, radius_arcsec: float = 1.0,
                    ra: str = "coord_ra", dec: str = "coord_dec") -> pd.DataFrame:
    """Nearest reference within ``radius_arcsec`` for each object (columns prefixed ``ref_``)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    a = SkyCoord(objects[ra].to_numpy(float) * u.deg, objects[dec].to_numpy(float) * u.deg)
    b = SkyCoord(ref["ra"].to_numpy(float) * u.deg, ref["dec"].to_numpy(float) * u.deg)
    idx, sep, _ = a.match_to_catalog_sky(b)
    ok = sep.arcsec <= radius_arcsec
    out = objects.copy()
    for c in ref.columns:
        if c in ("ra", "dec"):
            continue
        vals = ref[c].to_numpy()[idx]
        out[f"ref_{c}"] = np.where(ok, vals, np.nan) if vals.dtype.kind in "fi" else np.where(ok, vals, None)
    out["ref_sep"] = np.where(ok, sep.arcsec, np.nan)
    return out
