"""Tests for the amortized-inference training-set pieces that do not need jaxsedfit."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.amortized.generator import ObserveConfig, Priors, observe  # noqa: E402
from rubin_imbh.amortized.library import BANDS, match_library, mock_library  # noqa: E402
from rubin_imbh.amortized.variability import drw_unit_batch, fvar_by_band  # noqa: E402


def test_drw_batch_unit_variance_and_autocorrelation():
    rng = np.random.default_rng(0)
    n, L, tau = 20000, 6, 50.0
    t = np.tile(np.array([0.0, 10.0, 30.0, 60.0, 100.0, 300.0]), (n, 1))
    x = drw_unit_batch(t, np.full(n, tau), rng)
    assert abs(x.var(axis=0) - 1).max() < 0.05
    for k, dt in ((1, 10.0), (3, 60.0)):
        r = np.corrcoef(x[:, 0], x[:, k])[0, 1]
        assert abs(r - np.exp(-dt / tau)) < 0.03
    # NaN padding is preserved and does not disturb the finite part
    t2 = t.copy()
    t2[:, 4:] = np.nan
    x2 = drw_unit_batch(t2, np.full(n, tau), rng)
    assert np.isnan(x2[:, 4:]).all() and np.isfinite(x2[:, :4]).all()


def test_fvar_rises_to_the_blue():
    f = fvar_by_band(np.array([0.1]), np.array([0.5]), np.array([[2550.0, 5100.0, 10200.0]]))
    np.testing.assert_allclose(f[0], [0.1 * 2**0.5, 0.1, 0.1 * 2**-0.5])


def test_match_library_picks_similar_brightness():
    rng = np.random.default_rng(1)
    lib = pd.concat([mock_library(4000, rng, "wfd"), mock_library(1000, rng, "ddf")], ignore_index=True)
    target = 10 ** ((31.4 - np.array([19.0, 22.0, 25.0])) / 2.5)
    idx = match_library(lib, np.repeat(target, 200), rng, layer="wfd", k=25)
    assert (lib.layer.iloc[idx] == "wfd").all()
    got = np.log10(lib.r_coadd_flux.to_numpy()[idx]).reshape(3, 200)
    assert np.all(np.abs(np.median(got, axis=1) - np.log10(target)) < 0.1)


def _sim(n, rng, varagn_njy):
    rmag = rng.uniform(19.0, 23.0, n)
    sim = pd.DataFrame({"redshift": rng.uniform(0.1, 1.0, n)})
    for b in BANDS:
        sim[f"true_{b}"] = 10 ** ((31.4 - rmag) / 2.5)
        sim[f"varagn_{b}"] = varagn_njy
    return sim


def test_observe_null_without_agn_and_recovers_injected_rms():
    rng = np.random.default_rng(2)
    lib = mock_library(3000, rng, "ddf")
    pri = Priors()
    cfg = ObserveConfig(layer="ddf", sys_frac={}, zp_sigma=0.0)  # pure measurement noise
    null = observe(_sim(1500, rng, 0.0), lib, pri, rng, cfg)
    z = null["r_varSigClip"].dropna()
    assert len(z) > 1000 and abs(z.mean()) < 0.15 and abs(z.std() - 1) < 0.15
    assert (null["all_varSigClip"] > 5).mean() < 0.01
    # coadd noise matches the borrowed errors
    pull = (null["coadd_r"] - null["true_r"]) / null["coadd_err_r"]
    assert abs(pull.std() - 1) < 0.08

    strong = observe(_sim(4000, rng, 3.0e4), lib, pri, rng, cfg)  # large variable AGN flux
    # A DRW observed over a baseline shorter than tau shows less than its asymptotic variance:
    # E[S^2] = sigma^2 (1 - sum_{i != j} rho_ij / (n (n - 1))), rho_ij = exp(-|t_i - t_j| / tau_obs)
    tau_obs = strong["tau_rest_days"].to_numpy() * (1 + strong["redshift"].to_numpy())
    expected = np.empty(len(strong))
    for i, (li, tau) in enumerate(zip(strong["lib_index"], tau_obs)):
        t = np.asarray(lib["r_night_mjd"].iat[li])
        n = len(t)
        rho = np.exp(-np.abs(t[:, None] - t[None, :]) / tau)
        expected[i] = strong["true_rms_r"].iat[i] ** 2 * (1 - (rho.sum() - n) / (n * (n - 1)))
    rec = strong["r_xsAbs"].to_numpy()
    ok = np.isfinite(rec) & (strong["r_nEpoch"] >= 20) & (expected > 0)
    # E[S^2] is about the mean (DRW sample variances are right-skewed, so medians sit low); the
    # mean of per-object ratios is stable (0.99-1.01 over seeds), unlike the ratio of sums,
    # which the largest-amplitude objects dominate
    ratio = float(np.mean(rec[ok] / expected[ok]))
    assert 0.95 < ratio < 1.05, ratio  # excess variance tracks the sampling-expected variance
    assert (strong["r_varSigClip"] > 5).mean() > 0.8
    for c in ("fvar0", "fvar_alpha", "tau_rest_days", "lib_index", "all_dof"):
        assert c in strong


def test_npe_flow_is_calibrated_on_a_toy_problem():
    """The flow + standardization + sampling machinery recovers a known Gaussian posterior."""
    try:
        import flowjax  # noqa: F401
    except ImportError:
        print("  (skipped: needs flowjax)")
        return
    from rubin_imbh.amortized.npe import NPE, FlowSpec

    rng = np.random.default_rng(3)
    n = 4000
    theta = rng.normal(0, 1, (n, 2))
    df = pd.DataFrame({"redshift": theta[:, 0], "log_stellar_mass": theta[:, 1]})
    for j, b in enumerate("ugrizy"):  # x = noisy linear functions of theta, as coadd fluxes
        df[f"coadd_{b}"] = 1e4 * (1 + 0.3 * theta[:, j % 2] + 0.05 * rng.normal(size=n))
        df[f"coadd_err_{b}"] = 500.0
    for b in "griz":
        for c in ("nEpoch", "xsAbs", "rmsErr", "varSigClip"):
            df[f"{b}_{c}"] = np.nan
    df["all_varSigClip"], df["all_dof"] = np.nan, 0
    spec = FlowSpec(flow_layers=2, nn_width=32, nn_depth=1, theta=("redshift", "log_stellar_mass"))
    npe = NPE.train(df.iloc[:3500], spec, max_epochs=40, max_patience=5, show_progress=False)
    s = npe.sample(df.iloc[3500:], n_samples=200)
    true = theta[3500:]
    lo, hi = np.percentile(s, [16, 84], axis=1)
    cov = np.mean((true >= lo) & (true <= hi), axis=0)
    assert np.all(np.abs(cov - 0.68) < 0.1), cov
    assert np.all(np.abs(np.median(s, axis=1) - true).mean(axis=0) < 0.3)


def test_empirical_nzm_resampling_matches_target():
    """Resampling a training set to an empirical p(z | m) reproduces its median z per magnitude."""
    from rubin_imbh.amortized.nz_prior import EmpiricalNzm, observed_r_mag, resample_to_nzm

    rng = np.random.default_rng(5)
    # target: median redshift rises with magnitude
    m_t = rng.uniform(18, 25.5, 200000)
    z_t = np.clip(rng.gamma(4.0, (0.05 + 0.08 * (m_t - 18)) / 4.0), 0, 2.99)
    target = EmpiricalNzm.from_catalog(z_t, m_t, m_range=(18.0, 25.5), z_range=(0.0, 3.0))
    # training set: uniform redshift, independent of magnitude
    n = 60000
    m = rng.uniform(18, 25.5, n)
    df = pd.DataFrame({"redshift": rng.uniform(0.0, 3.0, n), "coadd_r": 10 ** ((31.4 - m) / 2.5)})
    out, ess = resample_to_nzm(df, target, 30000, rng)
    assert 1000 < ess < n
    mo = observed_r_mag(out)
    for lo, hi in ((19, 20), (22, 23), (24, 25)):
        want = np.median(z_t[(m_t > lo) & (m_t < hi)])
        got = np.median(out.redshift[(mo > lo) & (mo < hi)])
        assert abs(got - want) < 0.05 + 0.05 * want, (lo, got, want)


def test_observe_zero_point_offsets_add_fractional_scatter():
    rng = np.random.default_rng(8)
    lib = mock_library(3000, rng, "ddf")
    sim = _sim(4000, rng, 0.0)
    out = observe(sim, lib, Priors(), rng, ObserveConfig(layer="ddf", sys_frac={}, zp_sigma=0.03))
    frac = (out["coadd_r"] - out["true_r"]) / out["true_r"]
    noise = out["coadd_err_r"] / out["true_r"]
    bright = (31.4 - 2.5 * np.log10(out["true_r"])) < 20.5  # measurement noise << 3%
    assert abs(np.std(frac[bright]) - np.sqrt(0.03**2 + np.median(noise[bright]) ** 2)) < 0.005


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
