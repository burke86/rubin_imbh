"""Regenerate the null-distribution table for the leave-one-out statistic in jaxpz.lightcurve.

For n Gaussian epochs, drop the one with the largest residual about the weighted mean,
recompute chi^2, and record the mean and sd of (chi^2 / (n - 2))^(1/3).
"""

import numpy as np

rng = np.random.default_rng(0)
ns = np.unique(np.round(np.geomspace(4, 400, 25)).astype(int))
mean, sd = [], []
for n in ns:
    m = 40000 if n < 100 else 15000
    z = rng.standard_normal((m, n))
    k = ((z - z.mean(1, keepdims=True)) ** 2).argmax(1)
    keep = np.ones_like(z, bool)
    keep[np.arange(m), k] = False
    zc = z[keep].reshape(m, n - 1)
    y = np.cbrt(((zc - zc.mean(1, keepdims=True)) ** 2).sum(1) / (n - 2))
    mean.append(round(float(y.mean()), 4))
    sd.append(round(float(y.std()), 4))
print("_CLIP_N =", ns.tolist())
print("_CLIP_MEAN =", mean)
print("_CLIP_SD =", sd)
