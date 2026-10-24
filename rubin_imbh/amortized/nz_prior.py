"""Empirical magnitude-redshift prior p(z | m) and resampling of training sets to match it.

The prior is a smoothed 2D histogram of a deep photo-z catalogue (COSMOS2020 by default:
LePhare ``lp_zbest`` vs HSC r ``auto`` magnitude). A training set is resampled with weights
w = p_emp(z | m) / p_train(z | m), matching the conditional only: the training magnitude
distribution is kept, so bright objects stay covered, while the redshift distribution at each
magnitude becomes realistic (the BPZ-style prior that removes many catastrophic photo-z
failures).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EmpiricalNzm:
    m_edges: np.ndarray
    z_edges: np.ndarray
    pz_given_m: np.ndarray  # (n_m, n_z), each row normalized to unit sum

    @classmethod
    def from_catalog(cls, z: np.ndarray, m: np.ndarray, *, m_range=(16.0, 26.0), dm: float = 0.25,
                     z_range=(0.0, 3.0), dz: float = 0.02, z_smooth: float = 0.04, min_per_bin: int = 200):
        """Histogram p(z | m); magnitude bins with few objects are merged with their neighbours."""
        ok = np.isfinite(z) & np.isfinite(m) & (z >= z_range[0]) & (z < z_range[1]) & (m >= m_range[0]) & (m < m_range[1])
        z, m = z[ok], m[ok]
        m_edges = np.arange(m_range[0], m_range[1] + 1e-9, dm)
        z_edges = np.arange(z_range[0], z_range[1] + 1e-9, dz)
        h, _, _ = np.histogram2d(m, z, [m_edges, z_edges])
        if z_smooth > 0:  # Gaussian smoothing along z
            zc = 0.5 * (z_edges[1:] + z_edges[:-1])
            kern = np.exp(-0.5 * ((zc[:, None] - zc[None, :]) / z_smooth) ** 2)
            h = h @ kern
        # pool sparse magnitude bins with neighbours until each has min_per_bin objects
        counts = np.histogram(m, m_edges)[0]
        pz = np.empty_like(h)
        for i in range(len(h)):
            lo = hi = i
            while counts[lo: hi + 1].sum() < min_per_bin and (lo > 0 or hi < len(h) - 1):
                lo, hi = max(lo - 1, 0), min(hi + 1, len(h) - 1)
            row = h[lo: hi + 1].sum(0)
            pz[i] = row / row.sum() if row.sum() > 0 else np.full(row.size, 1.0 / row.size)
        return cls(m_edges, z_edges, pz)

    @classmethod
    def from_cosmos2020(cls, path, **kw):
        d = pd.read_parquet(Path(path).expanduser())
        with np.errstate(divide="ignore", invalid="ignore"):
            m = 23.9 - 2.5 * np.log10(d["hsc_r_flux_auto"].to_numpy(float))  # uJy -> AB
        return cls.from_catalog(d["lp_zbest"].to_numpy(float), m, **kw)

    def density(self, z: np.ndarray, m: np.ndarray) -> np.ndarray:
        """p(z | m) per unit z at the given points (0 outside the grid)."""
        i = np.clip(np.digitize(m, self.m_edges) - 1, 0, len(self.m_edges) - 2)
        j = np.digitize(z, self.z_edges) - 1
        inside = (j >= 0) & (j < len(self.z_edges) - 1)
        dz = np.diff(self.z_edges)[np.clip(j, 0, len(self.z_edges) - 2)]
        return np.where(inside, self.pz_given_m[i, np.clip(j, 0, len(self.z_edges) - 2)] / dz, 0.0)


def observed_r_mag(df: pd.DataFrame) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 31.4 - 2.5 * np.log10(np.clip(df["coadd_r"].to_numpy(float), 1e-3, None))


def nzm_weights(df: pd.DataFrame, target: EmpiricalNzm, max_weight: float = 1.0e4) -> np.ndarray:
    """w = p_target(z | m) / p_train(z | m), with p_train estimated on the same grid.

    ``max_weight`` (in units of the median positive weight) only guards against extreme
    ratios; a low cap biases the result toward the training distribution where the target is
    narrow (bright magnitudes). Check the effective sample size instead.
    """
    z = df["redshift"].to_numpy(float)
    m = observed_r_mag(df)
    train = EmpiricalNzm.from_catalog(z, m, m_range=(target.m_edges[0], target.m_edges[-1]),
                                      dm=float(np.diff(target.m_edges)[0]), z_range=(target.z_edges[0], target.z_edges[-1]),
                                      dz=float(np.diff(target.z_edges)[0]), z_smooth=0.04, min_per_bin=100)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = target.density(z, m) / train.density(z, m)
    w = np.where(np.isfinite(w), w, 0.0)
    return np.clip(w, 0.0, max_weight * np.median(w[w > 0]) if (w > 0).any() else max_weight)


def resample_to_nzm(df: pd.DataFrame, target: EmpiricalNzm, n_out: int, rng: np.random.Generator) -> tuple[pd.DataFrame, float]:
    """Resample (with replacement) so the conditional p(z | m) matches ``target``.

    Returns the resampled frame and the effective sample size of the weights.
    """
    w = nzm_weights(df, target)
    ess = float(w.sum() ** 2 / (w**2).sum()) if w.sum() > 0 else 0.0
    idx = rng.choice(len(df), size=n_out, replace=True, p=w / w.sum())
    return df.iloc[idx].reset_index(drop=True), ess
