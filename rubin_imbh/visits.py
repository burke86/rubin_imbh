"""Visit-level diagnostics: bad visits, coherent per-visit offsets, same-night correlation.

The variability statistics treat epochs as independent with correctly scaled errors. If
whole visits are bad (poor subtraction or a zeropoint error shared by every object in the
exposure), or visits within a night are correlated, non-variable objects pick up excess
chi^2 that no per-object error floor fixes. These functions measure that from an ensemble
of reference objects and test corrections, all on a flat per-epoch table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .varstats import Columns, Cuts, good_epoch_mask, lightcurve_stats

AB_ZP_NJY = 31.4  # m_AB = 31.4 - 2.5 log10(f / nJy)


def mag_to_njy(mag: float) -> float:
    return 10 ** ((AB_ZP_NJY - mag) / 2.5)


def flatten(nf, cols: Columns, err_scale: float, bands: str = "griz") -> pd.DataFrame:
    """One row per good epoch with object position, detection count and scaled errors."""
    lc = nf[cols.lc]
    lengths = np.asarray(lc.nest.list_lengths)
    flat = lc.nest.to_flat()
    obj = np.repeat(np.arange(len(nf)), lengths)
    good = good_epoch_mask(flat, cols, cols.flags)
    band = flat[cols.band].to_numpy()
    good &= flat[cols.band].isin(list(bands)).fillna(False).to_numpy(dtype=bool)  # NA-safe
    df = pd.DataFrame(
        {
            "obj": obj,
            "nDia": np.repeat(nf[cols.n_dia_sources].to_numpy(), lengths),
            "band": band,
            "visit": flat["visit"].to_numpy(),
            "mjd": flat["midpointMjdTai"].to_numpy(dtype=float),
            "f": flat[cols.flux].to_numpy(dtype=float, na_value=np.nan),
            "e": err_scale * flat[cols.flux_err].to_numpy(dtype=float, na_value=np.nan),
            "s": flat[cols.sci_flux].to_numpy(dtype=float, na_value=np.nan),
        }
    )[good].reset_index(drop=True)
    df["night"] = np.floor(df["mjd"] - 0.5).astype("int64")  # Rubin dayObs boundary, 12:00 UTC
    return add_residuals(df)


def add_residuals(df: pd.DataFrame, flux: str = "f") -> pd.DataFrame:
    """Per (object, band) weighted mean, normalized residual z and fractional residual."""
    g = df.groupby(["obj", "band"], sort=False)
    w = 1.0 / df["e"] ** 2
    mean = (w * df[flux]).groupby([df["obj"], df["band"]], sort=False).transform("sum") / w.groupby(
        [df["obj"], df["band"]], sort=False
    ).transform("sum")
    out = df.assign(
        F=g["s"].transform("median"),
        n_ep=g["f"].transform("size"),
        z=(df[flux] - mean) / df["e"],
    )
    out["frac"] = (out[flux] - mean) / out["F"].abs()
    return out


def visit_table(df: pd.DataFrame, ref: np.ndarray, bright: np.ndarray, min_ref: int = 20) -> pd.DataFrame:
    """Per-visit ensemble statistics from reference objects.

    ``sd_z`` (robust, 1.4826 MAD) should be ~1 for a good visit; ``f4`` is the fraction of
    reference epochs beyond 4 sigma (~6e-5 expected); ``off_frac`` is the median fractional
    residual of bright reference objects, i.e. a coherent per-visit offset such as a
    zeropoint error, with ``off_err`` its standard error.
    """
    r = df[ref]
    gz = r.groupby("visit")
    t = pd.DataFrame(
        {
            "band": gz["band"].first(),
            "mjd": gz["mjd"].first(),
            "night": gz["night"].first(),
            "n_ref": gz["z"].size(),
            "med_z": gz["z"].median(),
            "sd_z": gz["z"].agg(lambda z: 1.4826 * np.median(np.abs(z - np.median(z)))),
            "f4": gz["z"].agg(lambda z: np.mean(np.abs(z) > 4)),
        }
    )
    b = df[ref & bright].groupby("visit")["frac"]
    t["n_bright"] = b.size().reindex(t.index).fillna(0).astype(int)
    t["off_frac"] = b.median().reindex(t.index)
    t["off_err"] = (1.2533 * b.std() / np.sqrt(b.size())).reindex(t.index)
    return t[t.n_ref >= min_ref].sort_values("mjd")


def apply_visit_offsets(df: pd.DataFrame, vt: pd.DataFrame, min_bright: int = 10) -> pd.Series:
    """Flux with each visit's coherent fractional offset (x each object's flux) removed."""
    off = vt.loc[vt.n_bright >= min_bright, "off_frac"]
    return df["f"] - df["visit"].map(off).fillna(0.0) * df["F"].abs()


def bin_nightly(df: pd.DataFrame, flux: str = "f") -> pd.DataFrame:
    """Inverse-variance average of each object's epochs within a night, per band."""
    w = 1.0 / df["e"] ** 2
    tmp = df.assign(w=w, wf=w * df[flux])
    g = tmp.groupby(["obj", "band", "night"], sort=False)
    out = g.agg(sw=("w", "sum"), swf=("wf", "sum"), s=("s", "median"), nDia=("nDia", "first")).reset_index()
    out["f"] = out["swf"] / out["sw"]
    out["e"] = 1.0 / np.sqrt(out["sw"])
    return out


def stats_from_flat(df: pd.DataFrame, n_obj: int, flux: str = "f", bands: str = "griz") -> pd.DataFrame:
    """``lightcurve_stats`` on a flattened table whose errors are already scaled."""
    cols = Columns(flux="_flux", flux_err="_err", sci_flux="_sci", band="band", flags=())
    flat = pd.DataFrame({"band": df["band"].to_numpy(), "_flux": df[flux].to_numpy(),
                         "_err": df["e"].to_numpy(), "_sci": df["s"].to_numpy()})
    return lightcurve_stats(df["obj"].to_numpy(), flat, n_obj, cols, Cuts(err_scale=1.0), bands=tuple(bands))


def frac5_by_mag(df, n_obj, groups, mag_bins, flux="f", min_epochs=10, sig=5.0, bands="griz"):
    """Fraction of objects with varSigClip >= sig, per band, detection group and magnitude bin."""
    st = stats_from_flat(df, n_obj, flux, bands)
    rows = {}
    for b in bands:
        ok = st[f"{b}_nEpoch"] >= min_epochs
        with np.errstate(invalid="ignore", divide="ignore"):
            mag = AB_ZP_NJY - 2.5 * np.log10(st[f"{b}_medFlux"])
        hit = st[f"{b}_varSigClip"] >= sig
        for gname, gmask in groups.items():
            for lo, hi in zip(mag_bins[:-1], mag_bins[1:]):
                m = ok & gmask & (mag > lo) & (mag <= hi)
                rows[(b, gname, f"{lo}-{hi}")] = (hit[m].mean() if m.sum() >= 20 else np.nan, int(m.sum()))
    return pd.Series({k: v[0] for k, v in rows.items()}), pd.Series({k: v[1] for k, v in rows.items()})
