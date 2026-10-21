"""Object-level false-positive diagnostics: outlier tails, airmass correlation, environment."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Rubin Observatory, Cerro Pachon
SITE_LAT, SITE_LON, SITE_HEIGHT = -30.2446, -70.7494, 2663.0


def airmass(mjd: np.ndarray, ra: float, dec: float) -> np.ndarray:
    """sec(z) at the Rubin site for a fixed position (a cone centre is fine for ~1 deg)."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    uniq, inv = np.unique(np.asarray(mjd, dtype=float), return_inverse=True)
    site = EarthLocation(lat=SITE_LAT * u.deg, lon=SITE_LON * u.deg, height=SITE_HEIGHT * u.m)
    altaz = SkyCoord(ra * u.deg, dec * u.deg).transform_to(AltAz(obstime=Time(uniq, format="mjd", scale="tai"), location=site))
    return (1.0 / np.cos(np.deg2rad(90.0 - altaz.alt.deg)))[inv]


def tail_stats(df: pd.DataFrame, sig: float = 4.0) -> pd.DataFrame:
    """Per (object, band): epochs, |z| > sig outliers, and the correlation of z with airmass."""
    d = df.assign(out=(df["z"].abs() > sig))
    g = d.groupby(["obj", "band"], sort=False)
    t = g.agg(n_ep=("z", "size"), n_out=("out", "sum"))
    if "airmass" in d:
        x, y = d["airmass"], d["z"]
        sx, sy = g["airmass"].transform("mean"), g["z"].transform("mean")
        cov = ((x - sx) * (y - sy)).groupby([d["obj"], d["band"]], sort=False).mean()
        vx = ((x - sx) ** 2).groupby([d["obj"], d["band"]], sort=False).mean()
        vy = ((y - sy) ** 2).groupby([d["obj"], d["band"]], sort=False).mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            t["r_airmass"] = cov / np.sqrt(vx * vy)
    return t


def match_environment(dia_ra, dia_dec, obj: pd.DataFrame, ra_col: str, dec_col: str,
                      ext_col: str | None, bright_flux_col: str | None, bright_njy: float,
                      match_arcsec: float = 1.0, color_cols: tuple[str, str] | None = None) -> pd.DataFrame:
    """Nearest Object within ``match_arcsec`` (its extendedness and optional colour) and
    distance to the nearest bright point source, for each DiaObject."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    dia = SkyCoord(np.asarray(dia_ra) * u.deg, np.asarray(dia_dec) * u.deg)
    cat = SkyCoord(obj[ra_col].to_numpy() * u.deg, obj[dec_col].to_numpy() * u.deg)
    idx, sep, _ = dia.match_to_catalog_sky(cat)
    out = pd.DataFrame({"sep_obj": sep.arcsec})
    if ext_col is not None:
        ext = obj[ext_col].to_numpy(dtype=float)[idx]
        out["extendedness"] = np.where(out["sep_obj"] <= match_arcsec, ext, np.nan)
    if color_cols is not None:
        f1, f2 = (obj[c].to_numpy(dtype=float)[idx] for c in color_cols)
        with np.errstate(invalid="ignore", divide="ignore"):
            color = -2.5 * np.log10(f1 / f2)
        out["color"] = np.where(out["sep_obj"] <= match_arcsec, color, np.nan)
    if bright_flux_col is not None:
        bright = obj[bright_flux_col].to_numpy(dtype=float) > bright_njy
        if ext_col is not None:
            bright &= obj[ext_col].to_numpy(dtype=float) < 0.5
        if bright.any():
            _, bsep, _ = dia.match_to_catalog_sky(cat[bright])
            out["d_bright"] = bsep.arcsec
            # a DiaObject on the bright star itself is distance ~0; that is the star, not its halo
    return out
