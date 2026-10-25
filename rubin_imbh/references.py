"""Reference redshifts (and X-ray luminosities) for validating posteriors on real objects.

Each builder returns a DataFrame with ``ra``, ``dec``, ``z_ref``, ``z_type`` (``spec`` or
``photo``) and, where available, ``logLX`` (log10 absorption-corrected 0.5-7 keV luminosity in
erg/s), ``is_xray_agn`` and ``is_star``.
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


def cosmos_specz_reference(path, reliable_flags=(3, 4, 13, 14)) -> pd.DataFrame:
    """COSMOS Spectroscopic Redshift Compilation (Khostovan et al. 2026, ApJS 282, 6; DR1.1 unique file).

    Keeps redshifts with quality flags of at least 95% confidence (3, 4; +10 for broad-line
    objects, which are marked ``is_bl_agn``). Positions are the Gaia-corrected ones; z < 0.003
    is treated as a star. Source: github.com/cosmosastro/speczcompilation.
    """
    from astropy.table import Table

    t = Table.read(Path(path).expanduser())
    flag = np.asarray(t["flag"], dtype=int)
    z = np.asarray(t["specz"], dtype=float)
    keep = np.isin(flag, reliable_flags) & np.isfinite(z) & (z >= 0)
    return pd.DataFrame({"ra": np.asarray(t["ra_corrected"], float)[keep], "dec": np.asarray(t["dec_corrected"], float)[keep],
                         "z_ref": z[keep], "z_type": "spec", "is_bl_agn": flag[keep] >= 10,
                         "is_star": z[keep] < 0.003})


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
                         "logLX": log_lx, "is_xray_agn": t["OType"].astype(str).str.strip().eq("AGN").to_numpy(),
                         "is_star": t["OType"].astype(str).str.strip().str.lower().eq("star").to_numpy()})


def milliquas_reference(cache="data/milliquas.parquet") -> pd.DataFrame:
    """Milliquas (Flesch 2023, VizieR VII/294) quasars and AGN with spectroscopic classification
    (Type starting with Q, A, B, K, N or L) and z > 0, downloaded once and cached as parquet."""
    path = Path(cache).expanduser()
    if path.exists():
        t = pd.read_parquet(path)
    else:
        from astroquery.vizier import Vizier

        t = Vizier(columns=["RAJ2000", "DEJ2000", "Name", "Type", "Rmag", "z"], row_limit=-1).get_catalogs("VII/294")[0].to_pandas()
        path.parent.mkdir(parents=True, exist_ok=True)
        t.to_parquet(path)
    typ = t["Type"].astype(str).str.strip()
    keep = typ.str[:1].isin(list("QABKNL")) & (t["z"].astype(float) > 0)
    t = t[keep]
    return pd.DataFrame({"ra": t["RAJ2000"].to_numpy(float), "dec": t["DEJ2000"].to_numpy(float),
                         "z_ref": t["z"].to_numpy(float), "z_type": "spec", "mq_type": typ[keep].to_numpy(),
                         "is_xray_agn": False})


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
