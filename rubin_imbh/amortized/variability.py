"""AGN variability model for the simulator: damped random walk sampled at survey cadences.

A single unit-variance DRW drives all bands (AGN variability is strongly correlated across
optical bands); each band's light curve is that process scaled by its rms amplitude.
"""

from __future__ import annotations

import numpy as np

REST_REF_ANGSTROM = 5100.0


def fvar_by_band(f0: np.ndarray, alpha: np.ndarray, rest_wave_angstrom: np.ndarray) -> np.ndarray:
    """Fractional rms of the variable AGN flux per band.

    ``f0`` is the fractional rms at rest-frame 5100 Å, rising toward the blue as
    ``(lambda_rest / 5100)^-alpha``. Shapes broadcast: (n,), (n,), (n, n_bands) -> (n, n_bands).
    """
    ratio = np.asarray(rest_wave_angstrom) / REST_REF_ANGSTROM
    return np.asarray(f0)[:, None] * ratio ** (-np.asarray(alpha)[:, None])


def drw_unit(times: np.ndarray, tau: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance damped random walk (Ornstein–Uhlenbeck) at sorted or unsorted ``times``.

    Exact discretization: x_{k+1} = x_k e^{-dt/tau} + sqrt(1 - e^{-2 dt/tau}) N(0, 1),
    started from the stationary distribution.
    """
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return t.copy()
    order = np.argsort(t)
    ts = t[order]
    x = np.empty_like(ts)
    x[0] = rng.standard_normal()
    if ts.size > 1:
        a = np.exp(-np.diff(ts) / tau)
        eps = rng.standard_normal(ts.size - 1) * np.sqrt(1.0 - a * a)
        for k in range(1, ts.size):
            x[k] = a[k - 1] * x[k - 1] + eps[k - 1]
    out = np.empty_like(x)
    out[order] = x
    return out


def drw_unit_batch(times: np.ndarray, tau: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance DRW for many objects at once.

    ``times`` is (n, L), NaN-padded, each row sorted ascending (NaNs last); ``tau`` is (n,).
    Returns (n, L) with NaN where ``times`` is NaN. The recursion runs over the L columns,
    vectorized across objects.
    """
    t = np.asarray(times, dtype=float)
    n, L = t.shape
    x = np.full((n, L), np.nan)
    if L == 0:
        return x
    x[:, 0] = np.where(np.isfinite(t[:, 0]), rng.standard_normal(n), np.nan)
    tau = np.asarray(tau, dtype=float)
    for k in range(1, L):
        ok = np.isfinite(t[:, k])
        dt = np.where(ok, t[:, k] - t[:, k - 1], 0.0)
        a = np.exp(-dt / tau)
        x[:, k] = np.where(ok, a * x[:, k - 1] + np.sqrt(1.0 - a * a) * rng.standard_normal(n), np.nan)
    return x
