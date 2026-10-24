"""Neural posterior estimation (NPE) with a conditional normalizing flow (flowjax).

The flow models p(theta | x) directly, trained by maximum likelihood on simulated
(theta, x) pairs from the training-set generator; the training priors are the priors of
the resulting posteriors (reweight posterior samples by p_new(theta) / p_train(theta) to
change them without retraining).

x: coadd photometry as asinh magnitudes softened at each band's 1-sigma error, log errors,
per-band variability summaries (excess-variance signal-to-noise, clipped significance,
nights) and the combined-band statistic; missing statistics are zero-filled with a mask.
theta: redshift, log stellar mass, log lambda L_lambda(5100 A), dust, SFH age.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = tuple("ugrizy")
VAR_BANDS = tuple("griz")
THETA = ("redshift", "log_stellar_mass", "log_L5100_erg", "log_ebv_gal", "log_sfh_age_gyr")
LN10 = np.log(10.0)


def luptitude(flux_njy: np.ndarray, err_njy: np.ndarray) -> np.ndarray:
    """asinh magnitude (Lupton et al. 1999) with softening b = 1-sigma error; finite for f <= 0."""
    b = np.clip(err_njy, 1e-3, None)
    return 31.4 - (2.5 / LN10) * (np.arcsinh(flux_njy / (2.0 * b)) + np.log(b))


def build_features(df: pd.DataFrame, use_variability: bool = True) -> tuple[np.ndarray, list[str]]:
    """Feature matrix (float32) and names from a training-set or real-object table.

    ``use_variability=False`` keeps only the coadd photometry (for ablations).
    """
    cols, names = [], []
    for b in BANDS:
        f = df[f"coadd_{b}"].to_numpy(float)
        e = df[f"coadd_err_{b}"].to_numpy(float)
        ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
        cols += [np.where(ok, luptitude(np.where(ok, f, 0), np.where(ok, e, 1)), 0.0),
                 np.where(ok, np.log10(np.where(ok, e, 1)), 0.0), ok.astype(float)]
        names += [f"lupt_{b}", f"logerr_{b}", f"has_{b}"]
    if not use_variability:
        return np.column_stack(cols).astype(np.float32), names
    for b in VAR_BANDS:
        n = df[f"{b}_nEpoch"].to_numpy(float)
        xs = df[f"{b}_xsAbs"].to_numpy(float)
        rms = df[f"{b}_rmsErr"].to_numpy(float)
        sig = df[f"{b}_varSigClip"].to_numpy(float)
        ok = np.isfinite(xs) & np.isfinite(rms) & (rms > 0)
        cols += [np.where(ok, np.arcsinh(np.where(ok, xs, 0) / np.where(ok, rms, 1) ** 2), 0.0),
                 np.where(np.isfinite(sig), np.arcsinh(np.nan_to_num(sig)), 0.0),
                 np.log1p(np.nan_to_num(n)), ok.astype(float)]
        names += [f"xsnr_{b}", f"sig_{b}", f"lognight_{b}", f"hasvar_{b}"]
    s = df["all_varSigClip"].to_numpy(float)
    cols += [np.where(np.isfinite(s), np.arcsinh(np.nan_to_num(s)), 0.0), np.log1p(np.nan_to_num(df["all_dof"].to_numpy(float)))]
    names += ["sig_all", "logdof_all"]
    return np.column_stack(cols).astype(np.float32), names


@dataclass
class FlowSpec:
    flow_layers: int = 8
    nn_width: int = 128
    nn_depth: int = 2
    knots: int = 8
    interval: float = 5.0
    theta: tuple[str, ...] = THETA
    use_variability: bool = True


@dataclass
class NPE:
    """Trained flow plus the standardization of theta and x."""

    spec: FlowSpec
    feature_names: list[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    t_mean: np.ndarray
    t_std: np.ndarray
    flow: object = field(repr=False, default=None)

    # ------------------------------------------------------------------ build / train
    @staticmethod
    def _make_flow(key, spec: FlowSpec, cond_dim: int):
        import jax.numpy as jnp
        from flowjax.bijections import RationalQuadraticSpline
        from flowjax.distributions import Normal
        from flowjax.flows import masked_autoregressive_flow

        d = len(spec.theta)
        return masked_autoregressive_flow(
            key, base_dist=Normal(jnp.zeros(d)), transformer=RationalQuadraticSpline(knots=spec.knots, interval=spec.interval),
            cond_dim=cond_dim, flow_layers=spec.flow_layers, nn_width=spec.nn_width, nn_depth=spec.nn_depth,
        )

    @classmethod
    def train(cls, df: pd.DataFrame, spec: FlowSpec = FlowSpec(), *, seed: int = 0, learning_rate: float = 1e-3,
              max_epochs: int = 200, max_patience: int = 10, batch_size: int = 512, show_progress: bool = True) -> "NPE":
        import jax
        from flowjax.train import fit_to_data

        x, names = build_features(df, spec.use_variability)
        t = df[list(spec.theta)].to_numpy(np.float32)
        x_mean, x_std = x.mean(0), x.std(0) + 1e-6
        t_mean, t_std = t.mean(0), t.std(0) + 1e-6
        key, k_init, k_fit = jax.random.split(jax.random.key(seed), 3)
        flow = cls._make_flow(k_init, spec, x.shape[1])
        flow, losses = fit_to_data(k_fit, flow, data=((t - t_mean) / t_std, (x - x_mean) / x_std),
                                   learning_rate=learning_rate, max_epochs=max_epochs, max_patience=max_patience,
                                   batch_size=batch_size, show_progress=show_progress)
        npe = cls(spec, names, x_mean, x_std, t_mean, t_std, flow)
        npe.losses = {k: [float(v) for v in vals] for k, vals in losses.items()}
        return npe

    # ------------------------------------------------------------------ inference
    def sample(self, df: pd.DataFrame, n_samples: int = 500, seed: int = 0, chunk: int = 2048) -> np.ndarray:
        """Posterior samples, shape (n_objects, n_samples, n_theta), in physical units."""
        import jax

        x, names = build_features(df, self.spec.use_variability)
        if names != self.feature_names:
            raise ValueError("feature layout differs from training")
        xs = (x - self.x_mean) / self.x_std
        fn = jax.jit(jax.vmap(lambda k, c: self.flow.sample(k, (n_samples,), condition=c)))
        out = []
        for i, start in enumerate(range(0, len(xs), chunk)):
            c = xs[start: start + chunk]
            keys = jax.random.split(jax.random.key(seed + i), len(c))  # flowjax needs typed keys
            out.append(np.asarray(fn(keys, c)))
        z = np.concatenate(out) if out else np.zeros((0, n_samples, len(self.spec.theta)))
        return z * self.t_std + self.t_mean

    # ------------------------------------------------------------------ persistence
    def save(self, directory) -> None:
        import equinox as eqx

        d = Path(directory).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(d / "flow.eqx", self.flow)
        meta = {"spec": asdict(self.spec), "feature_names": self.feature_names,
                "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
                "t_mean": self.t_mean.tolist(), "t_std": self.t_std.tolist(),
                "losses": getattr(self, "losses", None)}
        (d / "npe.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, directory) -> "NPE":
        import equinox as eqx
        import jax

        d = Path(directory).expanduser()
        meta = json.loads((d / "npe.json").read_text())
        spec = FlowSpec(**{**meta["spec"], "theta": tuple(meta["spec"]["theta"])})
        skeleton = cls._make_flow(jax.random.key(0), spec, len(meta["feature_names"]))
        flow = eqx.tree_deserialise_leaves(d / "flow.eqx", skeleton)
        return cls(spec, meta["feature_names"], np.array(meta["x_mean"], np.float32), np.array(meta["x_std"], np.float32),
                   np.array(meta["t_mean"], np.float32), np.array(meta["t_std"], np.float32), flow)


def evaluate(npe: NPE, df: pd.DataFrame, n_samples: int = 300, seed: int = 1) -> pd.DataFrame:
    """Accuracy and calibration on held-out simulations, per parameter.

    bias / scatter use the posterior median (scatter = 1.4826 MAD); ``cov68`` / ``cov95`` are
    the fractions of true values inside the central 68% / 95% credible intervals (a
    calibrated posterior gives 0.68 / 0.95). For redshift, ``nmad`` and ``outlier`` use
    dz / (1 + z) with |dz| / (1 + z) > 0.15 as an outlier.
    """
    s = npe.sample(df, n_samples, seed)
    rows = {}
    for j, name in enumerate(npe.spec.theta):
        true = df[name].to_numpy(float)
        post = s[:, :, j]
        med = np.median(post, axis=1)
        lo68, hi68 = np.percentile(post, [16, 84], axis=1)
        lo95, hi95 = np.percentile(post, [2.5, 97.5], axis=1)
        resid = med - true
        r = {"bias": float(np.median(resid)), "scatter": float(1.4826 * np.median(np.abs(resid - np.median(resid)))),
             "cov68": float(np.mean((true >= lo68) & (true <= hi68))), "cov95": float(np.mean((true >= lo95) & (true <= hi95))),
             "post_width68": float(np.median(hi68 - lo68))}
        if name == "redshift":
            dz = resid / (1 + true)
            r["nmad"] = float(1.4826 * np.median(np.abs(dz - np.median(dz))))
            r["outlier"] = float(np.mean(np.abs(dz) > 0.15))
        rows[name] = r
    return pd.DataFrame(rows).T


def summarize_posteriors(npe: "NPE", obs: pd.DataFrame, n_samples: int = 100, chunk: int = 20000,
                         seed: int = 0) -> pd.DataFrame:
    """Posterior summaries for every row of ``obs`` (``real_observables`` output), in chunks.

    Adds ``post_<theta>_{med,lo,hi}`` (median, 16th, 84th percentile) under the training prior;
    with ``log_L5100_erg`` among the parameters also ``pop_<theta>_{med,lo,hi}`` under the
    population AGN prior (``agn_prior``, importance reweighting), ``p_agn_train`` / ``p_agn_pop``
    = P(lambda_Edd > 0.01) and ``pop_ess`` (effective number of reweighted samples).
    """
    from .agn_prior import EddingtonRatioPrior, reweight, weighted_quantiles

    th = list(npe.spec.theta)
    parts = []
    for a in range(0, len(obs), chunk):
        o = obs.iloc[a : a + chunk].copy()
        s = npe.sample(o, n_samples, seed=seed + a)
        for j, name in enumerate(th):
            o[f"post_{name}_med"] = np.median(s[:, :, j], axis=1)
            o[f"post_{name}_lo"], o[f"post_{name}_hi"] = np.percentile(s[:, :, j], [16, 84], axis=1)
        if "log_L5100_erg" in th:
            w, ess = reweight(s, th)
            for j, name in enumerate(th):
                o[f"pop_{name}_lo"], o[f"pop_{name}_med"], o[f"pop_{name}_hi"] = weighted_quantiles(s[:, :, j], w, (0.16, 0.5, 0.84))
            agn = EddingtonRatioPrior().log_lam(s[:, :, th.index("log_L5100_erg")], s[:, :, th.index("log_stellar_mass")]) > -2.0
            o["p_agn_train"], o["p_agn_pop"], o["pop_ess"] = agn.mean(axis=1), (w * agn).sum(axis=1), ess
        parts.append(o)
    return pd.concat(parts) if parts else obs
