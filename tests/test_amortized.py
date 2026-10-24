"""Tests for the amortized-inference training-set pieces that do not need jaxsedfit."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.amortized.generator import ObserveConfig, Priors, observe, redraw_coadds  # noqa: E402
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
    idx = match_library(lib, np.repeat(target, 200), rng, layer="wfd", k=25, band="r")
    assert (lib.layer.iloc[idx] == "wfd").all()
    got = np.log10(lib.r_coadd_flux.to_numpy()[idx]).reshape(3, 200)
    assert np.all(np.abs(np.median(got, axis=1) - np.log10(target)) < 0.1)


def test_match_library_keeps_missing_band_patterns():
    rng = np.random.default_rng(2)
    lib = mock_library(5000, rng, "wfd")
    lib.loc[rng.random(len(lib)) < 0.5, ["r_coadd_flux", "r_coadd_err"]] = np.nan  # like DP2 WFD
    target = 10 ** ((31.4 - rng.uniform(19, 25, 4000)) / 2.5)
    idx = match_library(lib, target, rng, layer="wfd")
    assert abs(lib.r_coadd_flux.iloc[idx].isna().mean() - 0.5) < 0.05


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


def test_bright_null_is_standard_with_injected_floor_scatter():
    rng = np.random.default_rng(10)
    lib = mock_library(3000, rng, "ddf")
    sim = _sim(3000, rng, 0.0)
    for b in BANDS:  # r ~ 19: the 0.75% floor dominates the photon noise
        sim[f"true_{b}"] = 10 ** ((31.4 - 19.0) / 2.5)
    sig = observe(sim, lib, Priors(), rng, ObserveConfig(layer="ddf"))["r_varSigClip"]
    assert abs(np.median(sig)) < 0.25 and 0.8 < 1.4826 * np.median(np.abs(sig - np.median(sig))) < 1.25
    flat = observe(sim, lib, Priors(), rng, ObserveConfig(layer="ddf", inject_sys=False))["r_varSigClip"]
    assert np.median(flat) < -1.0  # without the scatter the floor makes bright objects look too constant


def test_band_dropout_hides_one_band_per_object():
    from rubin_imbh.amortized.npe import build_features

    rng = np.random.default_rng(9)
    lib = mock_library(2000, rng, "ddf")
    out = observe(_sim(6000, rng, 0.0), lib, Priors(), rng, ObserveConfig(layer="ddf", band_dropout=0.3))
    missing = np.column_stack([out[f"coadd_{b}"].isna() for b in BANDS])
    assert abs(missing.any(axis=1).mean() - 0.3) < 0.03 and missing.sum(axis=1).max() == 1
    assert np.allclose(missing.mean(axis=0), 0.05, atol=0.015)
    x, names = build_features(out)
    assert np.isfinite(x).all() and np.allclose(x[:, names.index("has_r")], ~missing[:, 2])


def test_redraw_coadds_changes_calibration_error_and_keeps_gaps():
    rng = np.random.default_rng(11)
    lib = mock_library(3000, rng, "ddf")
    out = observe(_sim(4000, rng, 0.0), lib, Priors(), rng, ObserveConfig(layer="ddf", sys_frac={}, band_dropout=0.3))
    gaps = out["coadd_g"].isna().to_numpy()
    redrawn = redraw_coadds(out.copy(), 0.05, rng)
    assert np.array_equal(redrawn["coadd_g"].isna().to_numpy(), gaps)
    bright = ((31.4 - 2.5 * np.log10(out["true_r"])) < 20.5).to_numpy() & ~redrawn["coadd_r"].isna().to_numpy()
    frac = ((redrawn["coadd_r"] - redrawn["true_r"]) / redrawn["true_r"])[bright]
    assert 0.045 < np.std(frac) < 0.06


def test_real_observables_match_training_features_and_reference_matching():
    from rubin_imbh.amortized.npe import build_features
    from rubin_imbh.amortized.real import real_observables
    from rubin_imbh.amortized.references import match_reference

    rng = np.random.default_rng(9)
    lib = mock_library(500, rng, "ddf")
    sim = observe(_sim(300, rng, 1e4), lib, Priors(), rng, ObserveConfig(layer="ddf"))
    # spread over 2 x 2 deg so unmatched objects have a negligible chance of a 1" coincidence
    st = pd.DataFrame({"objectId": np.arange(300), "coord_ra": rng.uniform(149, 151, 300),
                       "coord_dec": rng.uniform(1, 3, 300), "envClean": True})
    for b in BANDS:
        st[f"{b}_cModelFlux"], st[f"{b}_cModelFluxErr"] = sim[f"coadd_{b}"], sim[f"coadd_err_{b}"]
    for b in "griz":
        for c in ("nEpoch", "varSigClip", "rmsErr"):
            st[f"{b}_{c}"] = sim[f"{b}_{c}"]
        st[f"{b}_medFlux"] = sim[f"true_{b}"]
        st[f"{b}_excessVar"] = sim[f"{b}_xsAbs"] / sim[f"true_{b}"] ** 2
    st["all_varSigClip"], st["all_dof"] = sim["all_varSigClip"], sim["all_dof"]
    obs = real_observables(st, "{b}_cModelFlux", "{b}_cModelFluxErr")
    x_real, n_real = build_features(obs)
    x_sim, n_sim = build_features(sim)
    assert n_real == n_sim
    np.testing.assert_allclose(x_real, x_sim, rtol=1e-5, atol=1e-5)
    ref = pd.DataFrame({"ra": st.coord_ra[:100] + 0.2 / 3600, "dec": st.coord_dec[:100], "z_ref": np.arange(100) / 10.0,
                        "z_type": "spec"})
    m = match_reference(obs, ref, 1.0)
    assert np.isfinite(m.ref_z_ref[:100]).all() and np.allclose(m.ref_z_ref[:100], np.arange(100) / 10.0)
    assert m.ref_z_ref[100:].isna().all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")


def test_agn_prior_reweighting_reproduces_population_prior():
    from rubin_imbh.amortized.agn_prior import EddingtonRatioPrior, reweight, weighted_quantiles

    rng = np.random.default_rng(12)
    pri, pop = Priors(), EddingtonRatioPrior()
    n = 200_000  # one "object" whose posterior equals the training prior, at fixed M* and z
    has = rng.random(n) < pri.agn_fraction
    log_l = np.where(has, rng.uniform(*pri.log_L5100_erg_range, n), rng.uniform(*pri.no_agn_log_L5100_erg_range, n))
    s = np.stack([np.full(n, 0.5), np.full(n, 10.5), log_l], axis=-1)[None]
    w, ess = reweight(s, ["redshift", "log_stellar_mass", "log_L5100_erg"], pop, pri)
    p_agn = (w[0] * (log_l > 40)).sum()
    f = min(pop.f_agn_z0 * 1.5 ** pop.z_evolution, pop.f_max)
    # target: f of the mass in the AGN part, minus the part of the ERDF below L5100 = 1e40
    lam40 = pop.log_lam(np.array([40.0]), np.array([10.5]))[0]
    g = pop.gamma * np.log(10)
    lo, hi = pop.log_lam_range
    below = (np.exp(g * max(lam40, lo)) - np.exp(g * lo)) / (np.exp(g * hi) - np.exp(g * lo))
    assert abs(p_agn - f * (1 - below)) < 0.01 and ess[0] > 1000
    med = weighted_quantiles(log_l[None], w, (0.5,))[0, 0]
    assert 36 < med < 46
