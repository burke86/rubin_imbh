"""Per-object, per-band variability statistics from nested DIA forced photometry.

Everything here is plain pandas/numpy so it can be unit-tested without LSDB and
applied partition-by-partition with ``Catalog.map_partitions``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

BANDS = tuple("ugrizy")
STATS = ("nEpoch", "medFlux", "rmsErr", "chi2dof", "varSig", "chi2dofClip", "varSigClip", "excessVar", "fvar")


@dataclass(frozen=True)
class Columns:
    """Column names of a DP2 HATS collection with nested forced photometry.

    Defaults are the ``dia_object_collection`` (DIA-detected objects); ``OBJECT`` below is
    the ``object_collection`` (every coadd Object). Confirm with
    ``scripts/select_variables.py inspect``; ``resolve_columns`` also auto-detects the
    light-curve column as the nested column holding ``flux``.
    """

    lc: str = "diaObjectForcedSource"  # nested forced-photometry light curve
    band: str = "band"
    flux: str = "psfDiffFlux"  # difference-image flux [nJy]
    flux_err: str = "psfDiffFluxErr"
    sci_flux: str = "psfFlux"  # direct-image flux, used to normalize excess variance
    obj_id: str = "diaObjectId"
    n_dia_sources: str | None = "nDiaSources"  # None: no DIA detection count (object_collection)
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


DIA = Columns()
# forced photometry on every coadd Object: no DIA detection needed, so low-amplitude
# nuclear variability (dwarf AGN, early TDEs) is not lost to the 5-sigma DIA threshold
OBJECT = Columns(lc="objectForcedSource", obj_id="objectId", n_dia_sources=None)


@dataclass(frozen=True)
class Cuts:
    min_dia_sources: int = 3  # coarse cut on the DiaObject summary row (dia source only)
    min_epochs: int = 10  # good forced-photometry epochs required in a band
    min_sig: float = 5.0  # chi^2 significance (Gaussian sigma) for "variable" in a band
    min_bands: int = 1  # number of bands that must pass min_sig
    err_scale: float = 1.0  # multiplicative error rescaling (calibrate on a test field)
    # systematic floor as a fraction of |direct-image flux|, added in quadrature per epoch;
    # a float for all bands or a {band: frac} dict (ECDFS: ~0.03 in g, ~0.02 in riz)
    sys_frac: float | dict[str, float] = 0.0
    # bands that count toward selection; stats are still computed for all of BANDS.
    # ECDFS: u and y have ~10 epochs and no variable/reference separation; g has a
    # colour-dependent airmass systematic (DCR or second-order extinction)
    select_bands: tuple[str, ...] = tuple("riz")
    robust: bool = True  # select on varSigClip (most deviant epoch dropped) instead of varSig
    # Combined-band statistic for sparse light curves (DP2 WFD: ~2/4/3 epochs in r/i/z):
    # one chi^2 over all epochs in combo_bands about per-band means. "either": variable if
    # the per-band OR the combined criterion passes; "only": combined only; "off": per band.
    combo_bands: tuple[str, ...] = tuple("riz")
    combined: str = "either"
    min_combo_dof: int = 4  # sum over combo bands of (n_b - 1)


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


# Null distribution of the leave-one-out statistic. Dropping the most deviant epoch
# biases chi^2 low, so (chi^2_clip / (n-2))^(1/3) is standardized with its simulated
# mean and sd for Gaussian noise (scripts/simulate_clip_null.py). Calibrated to
# |mean| < 0.03, sd within 1% of unity and Gaussian 3-4 sigma tails for 8 <= n <= 400,
# with unequal errors.
_CLIP_N = np.array([4, 5, 6, 7, 9, 10, 13, 15, 19, 22, 27, 33, 40, 48, 59, 71, 86, 104, 126, 153, 186, 225, 273, 330, 400])
_CLIP_MEAN = np.array([0.6246, 0.7078, 0.7566, 0.7873, 0.8275, 0.8427, 0.8736, 0.8868, 0.9056, 0.9167, 0.9283, 0.9385, 0.9466, 0.9542, 0.9617, 0.9663, 0.9712, 0.9752, 0.9788, 0.9813, 0.9847, 0.9868, 0.9885, 0.9905, 0.9918])
_CLIP_SD = np.array([0.247, 0.22, 0.1985, 0.1828, 0.1596, 0.1519, 0.131, 0.1217, 0.1081, 0.1005, 0.0899, 0.0818, 0.0742, 0.0673, 0.0609, 0.0554, 0.0506, 0.0458, 0.0415, 0.0381, 0.0341, 0.0313, 0.0286, 0.026, 0.0235])


def _clip_null_moments(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd of the cube-root leave-one-out statistic under the null, for n epochs."""
    n = np.asarray(n, dtype=float)
    mean = np.interp(np.log(n), np.log(_CLIP_N), _CLIP_MEAN)
    sd = np.interp(np.log(n), np.log(_CLIP_N), _CLIP_SD)
    big = n > _CLIP_N[-1]
    if big.any():
        # Wilson-Hilferty for n-2 dof, minus the dropped epoch (~ expected max of n chi^2_1
        # values), scaled to match the simulated table at its last point
        def asym(m):
            d = m - 2.0
            return 1.0 - 2.0 / (9.0 * d) - (2.0 * np.log(m) - np.log(np.log(m))) / (3.0 * d)

        nb = n[big]
        mean[big] = asym(nb) + (_CLIP_MEAN[-1] - asym(_CLIP_N[-1]))
        sd[big] = np.sqrt(2.0 / (9.0 * (nb - 2.0)))
    return mean, sd


def clip_chi2_to_sigma(chi2_clip: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Gaussian-equivalent significance of a leave-one-out chi^2 from n epochs (n >= 4)."""
    chi2_clip = np.asarray(chi2_clip, dtype=float)
    n = np.asarray(n, dtype=float)
    ok = n >= _CLIP_N[0]
    nn = np.where(ok, n, _CLIP_N[0])
    mean, sd = _clip_null_moments(nn)
    z = (np.cbrt(chi2_clip / (nn - 2.0)) - mean) / sd
    return np.where(ok, z, np.nan)


COMBO_SIM_MAX_EPOCHS = 60  # above this, the single-band table is used (split barely matters)


@lru_cache(maxsize=None)
def _combo_null_moments(split: tuple[int, ...], draws: int = 40_000) -> tuple[float, float]:
    """Null mean/sd of (chi2_clip / (dof - 1))^(1/3) for a band split, by simulation.

    chi2 is summed over bands about each band's own mean; the single epoch with the largest
    normalized residual over all bands is removed (exact leave-one-out within its band).
    Equal weights; the single-band case showed unequal errors do not matter.
    """
    rng = np.random.default_rng(abs(hash(split)) % 2**32)
    chi2 = np.zeros(draws)
    best = np.full(draws, -1.0)
    drop = np.zeros(draws)
    for n in split:
        z = rng.standard_normal((draws, n))
        r2 = (z - z.mean(1, keepdims=True)) ** 2
        chi2 += r2.sum(1)
        k = r2.argmax(1)
        rk = r2[np.arange(draws), k]
        eff = rk * n / (n - 1) if n > 1 else np.zeros(draws)
        upd = rk > best
        best = np.where(upd, rk, best)
        drop = np.where(upd, eff, drop)
    dof = sum(n - 1 for n in split)
    y = np.cbrt(np.maximum(chi2 - drop, 0.0) / (dof - 1))
    return float(y.mean()), float(y.std())


def combo_clip_sigma(chi2_clip: np.ndarray, counts: np.ndarray, min_dof_clip: int = 3) -> np.ndarray:
    """Gaussian-equivalent significance of a combined-band leave-one-out chi^2.

    ``counts`` is (n_obj, n_bands) epochs per band. The null depends on how epochs split
    across bands, so it is simulated once per distinct split (cached); for more than
    ``COMBO_SIM_MAX_EPOCHS`` epochs the single-band table at the same dof is used.
    """
    counts = np.asarray(counts, dtype=np.int64)
    dof_c = np.clip(counts - 1, 0, None).sum(1) - 1
    out = np.full(len(counts), np.nan)
    ok = dof_c >= min_dof_clip
    if not ok.any():
        return out
    splits = -np.sort(-counts[ok], axis=1)
    uniq, inv = np.unique(splits, axis=0, return_inverse=True)
    mean, sd = np.empty(len(uniq)), np.empty(len(uniq))
    for j, row in enumerate(uniq):
        split = tuple(int(v) for v in row if v > 0)
        if sum(split) > COMBO_SIM_MAX_EPOCHS:
            dof = sum(n - 1 for n in split)
            m, s = _clip_null_moments(np.array([dof + 1.0]))
            mean[j], sd[j] = m[0], s[0]
        else:
            mean[j], sd[j] = _combo_null_moments(split)
    inv = np.asarray(inv).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = (np.cbrt(np.asarray(chi2_clip, float)[ok] / dof_c[ok]) - mean[inv]) / sd[inv]
    return out


def band_floor(sys_frac: float | dict[str, float], band: np.ndarray) -> np.ndarray:
    """Per-epoch systematic floor fraction from a scalar or a per-band mapping."""
    if isinstance(sys_frac, dict):
        return pd.Series(band).map(sys_frac).fillna(0.0).to_numpy(dtype=float)
    return np.full(len(band), float(sys_frac))


def good_epoch_mask(flat: pd.DataFrame, cols: Columns, flags: tuple[str, ...]) -> np.ndarray:
    f = flat[cols.flux].to_numpy(dtype=float, na_value=np.nan)
    e = flat[cols.flux_err].to_numpy(dtype=float, na_value=np.nan)
    good = np.isfinite(f) & np.isfinite(e) & (e > 0)
    for flag in flags:
        good &= ~flat[flag].fillna(False).to_numpy(dtype=bool)
    return good


def band_codes(band: pd.Series, bands: tuple[str, ...]) -> np.ndarray:
    """Integer index of each epoch's band in ``bands`` (-1 if absent or missing), without
    materializing Python strings for Arrow-backed columns."""
    if isinstance(band.dtype, pd.ArrowDtype):
        import pyarrow as pa
        import pyarrow.compute as pc

        idx = pc.index_in(pa.array(band.array), value_set=pa.array(list(bands)))
        return np.asarray(idx.to_pandas(), dtype=float)  # nulls -> nan
    codes = pd.Categorical(band, categories=list(bands)).codes.astype(float)
    codes[codes < 0] = np.nan
    return codes


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

    Epochs are grouped by the integer key ``obj * n_bands + band`` with ``np.bincount``
    (no pandas groupby, no string columns), keeping memory to a few float arrays per epoch.
    """
    nb = len(bands)
    code = band_codes(flat[cols.band], bands)
    good = good_epoch_mask(flat, cols, flags) & np.isfinite(code)
    code = code[good].astype(np.int64)
    key = np.asarray(obj)[good].astype(np.int64) * nb + code
    f = flat[cols.flux].to_numpy(dtype=float, na_value=np.nan)[good]
    e = cuts.err_scale * flat[cols.flux_err].to_numpy(dtype=float, na_value=np.nan)[good]
    s = flat[cols.sci_flux].to_numpy(dtype=float, na_value=np.nan)[good]
    if isinstance(cuts.sys_frac, dict):
        floor = np.array([cuts.sys_frac.get(b, 0.0) for b in bands])[code]
    else:
        floor = np.full(len(code), float(cuts.sys_frac))
    if np.any(floor > 0):
        e = np.hypot(e, floor * np.nan_to_num(np.abs(s)))
    e2 = e**2
    w = 1.0 / e2
    K = n_obj * nb

    def gsum(x):
        return np.bincount(key, weights=x, minlength=K)

    n = np.bincount(key, minlength=K).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sw = gsum(w)
        mu = gsum(w * f) / sw
        d = f - mu[key]
        r2 = w * d * d
        chi2 = gsum(r2)  # chi^2 about the weighted mean (two-pass: no cancellation)
        mu_u = gsum(f) / n
        var_f = gsum((f - mu_u[key]) ** 2) / (n - 1)
        var_f[n < 2] = np.nan
        mean_e2 = gsum(e2) / n

    # median direct-image flux per group: sort by (key, s); NaNs sort last within a group
    order = np.lexsort((s, key))
    ks, ss = key[order], s[order]
    uniq, start = np.unique(ks, return_index=True)
    n_valid = np.bincount(key, weights=np.isfinite(s).astype(float), minlength=K)[uniq].astype(np.int64)
    med = np.full(K, np.nan)
    if len(ss):
        lo = start + np.maximum(n_valid - 1, 0) // 2
        hi = start + n_valid // 2
        med[uniq] = np.where(n_valid > 0, 0.5 * (ss[lo] + ss[hi]), np.nan)

    # Leave-one-out chi^2 without each light curve's most deviant epoch, so a single
    # unflagged glitch cannot make an object "variable". Exact closed form: dropping epoch k
    # lowers chi^2 by w_k d_k^2 * sw / (sw - w_k).
    order = np.lexsort((-r2, key))
    ks = key[order]
    first = order[np.r_[True, ks[1:] != ks[:-1]]] if len(ks) else order
    wk = np.zeros(K)
    r2k = np.zeros(K)
    wk[key[first]] = w[first]
    r2k[key[first]] = r2[first]
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_c = np.maximum(chi2 - r2k * sw / (sw - wk), 0.0)
        dof, dof_c = n - 1.0, n - 2.0
        xs = (var_f - mean_e2) / med**2
        stats = {
            "nEpoch": n,
            "medFlux": med,  # direct-image flux [nJy], includes host
            "rmsErr": np.sqrt(mean_e2),  # after err_scale / sys_frac
            "chi2dof": np.where(dof > 0, chi2 / dof, np.nan),
            "varSig": chi2_to_sigma(chi2, dof),
            "chi2dofClip": np.where(dof_c > 0, chi2_c / dof_c, np.nan),
            "varSigClip": clip_chi2_to_sigma(chi2_c, n),
            "excessVar": xs,
            "fvar": np.sqrt(np.clip(xs, 0.0, None)),
        }
    empty = n == 0
    out = {}
    combo = combined_stats(cuts, bands, n, chi2, sw, key, code, r2, w, n_obj)
    for j, b in enumerate(bands):
        for name in STATS:
            v = stats[name].reshape(n_obj, nb)[:, j] if n_obj else np.empty(0)
            if name == "nEpoch":
                out[f"{b}_{name}"] = v.astype("int32")
            else:
                out[f"{b}_{name}"] = np.where(empty.reshape(n_obj, nb)[:, j], np.nan, v) if n_obj else v
    out.update(combo)
    return pd.DataFrame(out, index=np.arange(n_obj))


COMBO_STATS = ("all_nEpoch", "all_dof", "all_chi2dof", "all_varSig", "all_chi2dofClip", "all_varSigClip")


def combined_stats(cuts, bands, n, chi2, sw, key, code, r2, w, n_obj) -> dict:
    """One chi^2 over all epochs in ``cuts.combo_bands``, each band about its own mean."""
    nb = len(bands)
    cb = [bands.index(b) for b in cuts.combo_bands if b in bands]
    if not cb or n_obj == 0:
        return {c: np.full(n_obj, np.nan) for c in COMBO_STATS}
    counts = n.reshape(n_obj, nb)[:, cb]
    chi2_all = chi2.reshape(n_obj, nb)[:, cb].sum(1)
    n_all = counts.sum(1)
    dof_all = np.clip(counts - 1, 0, None).sum(1)
    # leave-one-out: the epoch with the largest normalized residual over the combo bands
    inb = np.isin(code, cb)
    oid = key[inb] // nb
    rr = r2[inb]
    order = np.lexsort((-rr, oid))
    os_ = oid[order]
    first = order[np.r_[True, os_[1:] != os_[:-1]]] if len(os_) else order
    g = key[inb][first]
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = np.where(n[g] > 1, rr[first] * sw[g] / (sw[g] - w[inb][first]), 0.0)
    drop = np.zeros(n_obj)
    drop[oid[first]] = eff
    chi2_c = np.maximum(chi2_all - drop, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "all_nEpoch": n_all.astype("int32"),
            "all_dof": dof_all.astype("int32"),
            "all_chi2dof": np.where(dof_all > 0, chi2_all / dof_all, np.nan),
            "all_varSig": chi2_to_sigma(chi2_all, dof_all),
            "all_chi2dofClip": np.where(dof_all > 1, chi2_c / (dof_all - 1), np.nan),
            "all_varSigClip": combo_clip_sigma(chi2_c, counts),
        }


def summarize(stats: pd.DataFrame, cuts: Cuts) -> pd.DataFrame:
    """Cross-band summary over ``cuts.select_bands``: variable band count and best significance."""
    bands = cuts.select_bands
    key = "varSigClip" if cuts.robust else "varSig"
    sig = np.column_stack([stats[f"{b}_{key}"].to_numpy(dtype=float) for b in bands])
    nep = np.column_stack([stats[f"{b}_nEpoch"].to_numpy() for b in bands])
    usable = nep >= cuts.min_epochs
    sig = np.where(usable, sig, np.nan)
    with np.errstate(invalid="ignore"):
        n_var = np.sum(sig >= cuts.min_sig, axis=1)
    max_sig = np.full(len(stats), np.nan)
    any_usable = usable.any(axis=1)
    max_sig[any_usable] = np.nanmax(sig[any_usable], axis=1)
    per_band = n_var >= cuts.min_bands
    combo_key = "all_varSigClip" if cuts.robust else "all_varSig"
    if combo_key in stats:
        csig = stats[combo_key].to_numpy(dtype=float)
        cdof = stats["all_dof"].to_numpy()
        with np.errstate(invalid="ignore"):
            var_combo = (cdof >= cuts.min_combo_dof) & (csig >= cuts.min_sig)
    else:
        var_combo = np.zeros(len(stats), bool)
    if cuts.combined == "only":
        is_var = var_combo
    elif cuts.combined == "either":
        is_var = per_band | var_combo
    else:
        is_var = per_band
    return pd.DataFrame(
        {"nVarBands": n_var.astype("int32"), "maxVarSig": max_sig, "varCombo": var_combo, "isVariable": is_var},
        index=stats.index,
    )
