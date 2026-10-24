"""Population AGN prior applied after training, by importance-reweighting posterior samples.

The flow is trained under ``Priors``: half the objects host an AGN with log lambda L_5100
uniform in [40, 46], the other half a negligible luminosity uniform in [36, 40]. That prior is
far more AGN-rich than real galaxies, and for objects whose data barely constrain L_AGN
(non-variable, host-dominated) the posterior inherits it. Reweighting each object's posterior
samples by p_pop(theta) / p_train(theta) gives the posterior under a population prior without
retraining (Bayes' theorem with the same likelihood).

The population prior here is a specific-accretion-rate (Eddington-ratio) distribution in the
spirit of Aird et al. (2012): a fraction f(z) of galaxies host an AGN with
p(log lambda_Edd) ~ lambda_Edd^gamma over [lam_min, lam_max], where
lambda_Edd = k_bol lambda L_5100 / (1.26e38 erg/s x M_BH / M_sun) and M_BH = mbh_frac x M*; the
rest keep the training prior's no-AGN tail. The defaults are approximate and meant to be tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generator import Priors


def train_log_l_density(log_l: np.ndarray, priors: Priors = Priors()) -> np.ndarray:
    """Training prior density of log10 lambda L_5100 (per dex)."""
    lo, hi = priors.log_L5100_erg_range
    nlo, nhi = priors.no_agn_log_L5100_erg_range
    f = priors.agn_fraction
    return (np.where((log_l >= lo) & (log_l <= hi), f / (hi - lo), 0.0)
            + np.where((log_l >= nlo) & (log_l < nhi), (1 - f) / (nhi - nlo), 0.0))


@dataclass
class EddingtonRatioPrior:
    gamma: float = -0.65               # slope of p(log lambda_Edd)
    log_lam_range: tuple[float, float] = (-4.0, 0.5)
    f_agn_z0: float = 0.1              # fraction of galaxies with an AGN in log_lam_range at z = 0
    z_evolution: float = 3.5           # f(z) = f_agn_z0 (1 + z)^z_evolution, capped at f_max
    f_max: float = 0.9
    k_bol: float = 10.0                # L_bol / lambda L_5100
    mbh_frac: float = 0.002            # M_BH / M*

    def log_lam(self, log_l: np.ndarray, log_mstar: np.ndarray) -> np.ndarray:
        return log_l + np.log10(self.k_bol) - np.log10(1.26e38) - np.log10(self.mbh_frac) - log_mstar

    def density(self, log_l: np.ndarray, log_mstar: np.ndarray, z: np.ndarray,
                priors: Priors = Priors()) -> np.ndarray:
        """Population prior density of log10 lambda L_5100 given M* and z (per dex)."""
        lo, hi = self.log_lam_range
        g = self.gamma * np.log(10.0)
        norm = (np.exp(g * hi) - np.exp(g * lo)) / g          # integral of 10^(gamma x) dx
        ll = self.log_lam(log_l, log_mstar)
        erdf = np.where((ll >= lo) & (ll <= hi), np.exp(g * ll) / norm, 0.0)
        f = np.minimum(self.f_agn_z0 * (1.0 + z) ** self.z_evolution, self.f_max)
        nlo, nhi = priors.no_agn_log_L5100_erg_range
        tail = np.where((log_l >= nlo) & (log_l < nhi), 1.0 / (nhi - nlo), 0.0)
        return f * erdf + (1.0 - f) * tail


def reweight(samples: np.ndarray, theta: list[str], prior: EddingtonRatioPrior = EddingtonRatioPrior(),
             priors: Priors = Priors()) -> tuple[np.ndarray, np.ndarray]:
    """Normalized importance weights (n_obj, n_samp) and their effective sample size per object.

    ``samples`` is ``NPE.sample`` output, (n_obj, n_samp, n_theta).
    """
    get = {name: samples[:, :, j] for j, name in enumerate(theta)}
    log_l, log_m, z = get["log_L5100_erg"], get["log_stellar_mass"], get["redshift"]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = prior.density(log_l, log_m, z, priors) / train_log_l_density(log_l, priors)
    w = np.where(np.isfinite(w), w, 0.0)
    tot = w.sum(axis=1, keepdims=True)
    w = np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), 1.0 / w.shape[1])
    ess = 1.0 / (w**2).sum(axis=1)
    return w, ess


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs) -> np.ndarray:
    """Per-row weighted quantiles: ``values``, ``weights`` (n_obj, n_samp) -> (len(qs), n_obj)."""
    order = np.argsort(values, axis=1)
    v = np.take_along_axis(values, order, axis=1)
    c = np.cumsum(np.take_along_axis(weights, order, axis=1), axis=1)
    c = c / c[:, -1:]
    return np.stack([v[np.arange(len(v)), np.minimum((c < q).sum(axis=1), v.shape[1] - 1)] for q in qs])
