"""Tests on mock DP2-like data. Run with ``pytest`` or ``python tests/test_pipeline.py``."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.varstats import Columns, Cuts, chi2_to_sigma, lightcurve_stats, summarize  # noqa: E402

COLS = Columns()


def mock_dia(n_const=2000, n_var=50, n_epoch=60, seed=0):
    """Constant objects with Gaussian noise plus a few 20%-rms red-noise-ish variables."""
    rng = np.random.default_rng(seed)
    n = n_const + n_var
    base = pd.DataFrame(
        {
            "diaObjectId": np.arange(n),
            "ra": rng.uniform(28.0, 29.0, n),
            "dec": rng.uniform(0.0, 1.0, n),
            "nDiaSources": np.r_[rng.integers(0, 6, n_const), rng.integers(5, 20, n_var)],
            "truth_var": np.r_[np.zeros(n_const, bool), np.ones(n_var, bool)],
            "tract": rng.integers(9000, 9100, n),
        }
    )
    rows = []
    for i in range(n):
        band = rng.choice(list("gri"), n_epoch)
        sci = rng.uniform(2e3, 2e4)  # nJy, ~21-23.5 mag
        err = 0.03 * sci * np.ones(n_epoch)
        signal = 0.2 * sci * np.sin(np.linspace(0, 3, n_epoch) + i) if base.truth_var[i] else 0.0
        diff = signal + rng.normal(0, err)
        flag = rng.random(n_epoch) < 0.05
        diff[flag] += 50 * sci  # flagged epochs are garbage and must be ignored
        rows.append(
            pd.DataFrame(
                {
                    "band": band,
                    "psfDiffFlux": diff,
                    "psfDiffFluxErr": err,
                    "psfFlux": sci + diff,
                    "psfDiffFlux_flag": flag,
                    "psfFluxErr": err,
                    "midpointMjdTai": 60800.0 + np.sort(rng.uniform(0, 270, n_epoch)),
                    "visit": rng.integers(2025_04_01_00000, 2026_01_31_99999, n_epoch),
                    "diff_PixelFlags_nodataCenter": np.zeros(n_epoch, bool),
                    "pixelFlags_edge": np.zeros(n_epoch, bool),
                },
                index=np.full(n_epoch, i),
            )
        )
    return base, pd.concat(rows)


def mock_dia_sources(lc):
    """Every 5th epoch as a 'detection', standing in for the unused nested diaSource column."""
    ds = lc.iloc[::5]
    return pd.DataFrame({"band": ds["band"], "snr": ds["psfDiffFlux"] / ds["psfDiffFluxErr"]}, index=ds.index)


def write_hats(nf, path, rows):
    """Write a mock HATS catalog with at most ~``rows`` objects per partition."""
    import lsdb

    import inspect

    # renamed in newer lsdb (RSP); older versions swallow unknown kwargs, so check the signature
    key = "partition_rows" if "partition_rows" in inspect.signature(lsdb.from_dataframe).parameters else "threshold"
    cat = lsdb.from_dataframe(nf, lowest_order=2, highest_order=9, **{key: rows})
    cat.to_hats(path, catalog_name=Path(path).name)


def test_clip_null_is_standard_normal():
    from rubin_imbh.varstats import clip_chi2_to_sigma

    rng = np.random.default_rng(5)
    for n in (8, 40, 120, 800):  # 800 exercises the large-n extrapolation
        e = rng.uniform(0.5, 2.0, (20000, n))
        f = rng.standard_normal(e.shape) * e
        w = 1 / e**2
        mu = (w * f).sum(1, keepdims=True) / w.sum(1, keepdims=True)
        k = (w * (f - mu) ** 2).argmax(1)
        keep = np.ones_like(f, bool)
        keep[np.arange(len(f)), k] = False
        fc, wc = f[keep].reshape(len(f), n - 1), w[keep].reshape(len(f), n - 1)
        muc = (wc * fc).sum(1, keepdims=True) / wc.sum(1, keepdims=True)
        z = clip_chi2_to_sigma((wc * (fc - muc) ** 2).sum(1), np.full(len(f), n))
        assert abs(z.mean()) < 0.06 and abs(z.std() - 1) < 0.06, (n, z.mean(), z.std())


def test_chi2_to_sigma_null():
    rng = np.random.default_rng(1)
    dof = 29
    z = chi2_to_sigma(rng.chisquare(dof, 200_000), np.full(200_000, dof))
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1) < 0.02
    assert chi2_to_sigma(np.array([1e6]), np.array([10]))[0] > 100  # no saturation


def test_stats_recover_variables():
    base, lc = mock_dia()
    obj = lc.index.to_numpy()
    cuts = Cuts()
    st = lightcurve_stats(obj, lc, len(base), COLS, cuts, flags=("psfDiffFlux_flag",))
    summ = summarize(st, cuts)
    const = ~base.truth_var.to_numpy()
    # null population is well calibrated once flagged epochs are dropped
    assert abs(np.nanmedian(st.loc[const, "r_chi2dof"]) - 1) < 0.1
    # ...and the leave-one-out significance is standardized despite its low bias
    zc = st.loc[const, "r_varSigClip"]
    assert abs(zc.mean()) < 0.1 and abs(zc.std() - 1) < 0.1
    assert (st.loc[const, "r_chi2dofClip"].median() < 0.95)  # raw clipped chi2 is biased low
    assert summ.isVariable[const].mean() < 0.005
    assert summ.isVariable[~const].all()
    fvar = st.loc[~const, "r_fvar"]
    assert 0.05 < np.nanmedian(fvar) < 0.3
    # without the flag cut the outliers make everything "variable"
    st_bad = lightcurve_stats(obj, lc, len(base), COLS, cuts, flags=())
    assert summarize(st_bad, Cuts(robust=False)).isVariable[const].mean() > 0.5


def test_lsdb_end_to_end():
    import lsdb
    import nested_pandas as npd

    from rubin_imbh.pipeline import select_variables

    base, lc = mock_dia(n_const=500, n_var=20)
    nf = npd.NestedFrame(base).add_nested(lc, "diaObjectForcedSource")
    with tempfile.TemporaryDirectory() as tmp:
        write_hats(nf, Path(tmp) / "mock_dia", rows=100)
        cat = lsdb.open_catalog(Path(tmp) / "mock_dia")
        assert cat.npartitions > 1
        res = select_variables(cat, COLS, Cuts(min_dia_sources=3)).compute()
        assert set(res.diaObjectId) == set(base.diaObjectId[base.truth_var])
        assert "diaObjectForcedSource" in res.columns and "r_varSig" in res.columns
        allstats = select_variables(cat, COLS, Cuts(min_dia_sources=0), select=False).compute()
        assert len(allstats) == len(base)


def test_tiles_partition_the_catalog():
    """Tiles finer than the partitions must neither drop nor duplicate objects."""
    import lsdb
    import nested_pandas as npd

    from rubin_imbh.pipeline import select_variables, tile_pixels

    base, lc = mock_dia(n_const=500, n_var=20)
    nf = npd.NestedFrame(base).add_nested(lc, "diaObjectForcedSource")
    with tempfile.TemporaryDirectory() as tmp:
        # threshold high enough that partitions are coarser than the order-6 tiles
        write_hats(nf, Path(tmp) / "mock_dia", rows=600)
        cat = lsdb.open_catalog(Path(tmp) / "mock_dia")
        tiles = tile_pixels(cat, 6)
        assert len(tiles) > cat.npartitions
        cuts = Cuts(min_dia_sources=0)
        ids = []
        for pix in tiles:
            ids += select_variables(cat, COLS, cuts, select=False, tile=(6, pix)).compute().diaObjectId.tolist()
        assert sorted(ids) == sorted(base.diaObjectId)


def test_needed_columns_prunes_unused_nested_data():
    import lsdb
    import nested_pandas as npd

    from rubin_imbh.pipeline import needed_columns, select_variables

    base, lc = mock_dia(n_const=300, n_var=15)
    nf = npd.NestedFrame(base).add_nested(lc, "diaObjectForcedSource").add_nested(mock_dia_sources(lc), "diaSource")
    with tempfile.TemporaryDirectory() as tmp:
        write_hats(nf, Path(tmp) / "mock_dia", rows=100)
        full = lsdb.open_catalog(Path(tmp) / "mock_dia")
        columns = needed_columns(full, COLS)
        assert "diaSource" not in columns and "truth_var" not in columns
        assert "diaObjectForcedSource.midpointMjdTai" in columns
        cat = lsdb.open_catalog(Path(tmp) / "mock_dia", columns=columns)
        assert cat.nested_columns == ["diaObjectForcedSource"]
        res = select_variables(cat, COLS, Cuts()).compute()
        assert set(res.diaObjectId) == set(base.diaObjectId[base.truth_var])
        lc_fields = set(res["diaObjectForcedSource"].nest.fields)
        assert {"midpointMjdTai", "visit", "psfDiffFlux"} <= lc_fields


def test_clipped_significance_ignores_single_glitch():
    """One unflagged outlier epoch inflates varSig but not varSigClip; real variables survive."""
    base, lc = mock_dia(n_const=1000, n_var=30, seed=3)
    lc = lc[~lc.psfDiffFlux_flag].copy()  # start clean, then plant one glitch per constant object
    first_r = ~lc.index.duplicated() & (lc.band == "r").to_numpy()
    glitch = lc.index.isin(base.diaObjectId[~base.truth_var]) & first_r
    lc.loc[glitch, "psfDiffFlux"] += 30 * lc.loc[glitch, "psfDiffFluxErr"]
    obj = lc.index.to_numpy()
    st = lightcurve_stats(obj, lc, len(base), COLS, Cuts())
    const, var = ~base.truth_var.to_numpy(), base.truth_var.to_numpy()
    hit = np.zeros(len(base), bool)
    hit[lc.index[glitch]] = True
    assert (st.loc[hit, "r_varSig"] > 5).mean() > 0.9  # naive statistic is fooled
    assert summarize(st, Cuts()).isVariable[const].mean() < 0.01  # clipped one is not
    assert summarize(st, Cuts()).isVariable[var].all()
    assert summarize(st, Cuts(robust=False)).isVariable[hit].mean() > 0.9


def test_per_band_sys_frac_suppresses_bright_floor():
    """A brightness-proportional extra scatter is absorbed by a matching per-band floor."""
    base, lc = mock_dia(n_const=600, n_var=0, seed=7)
    lc = lc[~lc.psfDiffFlux_flag].copy()
    rng = np.random.default_rng(8)
    gband = (lc.band == "g").to_numpy()
    lc.loc[gband, "psfDiffFlux"] += rng.normal(0, 0.10 * lc.loc[gband, "psfFlux"].abs())  # >> 3% mock errors
    obj = lc.index.to_numpy()
    st0 = lightcurve_stats(obj, lc, len(base), COLS, Cuts())
    st1 = lightcurve_stats(obj, lc, len(base), COLS, Cuts(sys_frac={"g": 0.10}))
    assert (st0.g_varSigClip > 5).mean() > 0.5  # floor unmodelled: mostly "variable"
    assert (st1.g_varSigClip > 5).mean() < 0.02  # modelled per band
    assert np.allclose(st0.r_varSigClip, st1.r_varSigClip, equal_nan=True)  # other bands untouched


def test_visit_offsets_are_recovered_and_removed():
    """Coherent per-visit zeropoint offsets are measured from bright reference objects."""
    from rubin_imbh.visits import add_residuals, apply_visit_offsets, frac5_by_mag, mag_to_njy, visit_table

    rng = np.random.default_rng(21)
    n_obj, n_vis = 800, 40
    off = rng.normal(0, 0.02, n_vis)
    flux = 10 ** ((31.4 - rng.uniform(17, 24, n_obj)) / 2.5)
    obj = np.repeat(np.arange(n_obj), n_vis)
    visit = np.tile(np.arange(n_vis), n_obj)
    e = np.full(len(obj), 120.0)
    f = rng.normal(0, e) + off[visit] * flux[obj]
    df = pd.DataFrame({"obj": obj, "nDia": 1, "band": "r", "visit": visit, "mjd": 61000.0 + visit,
                       "night": visit, "f": f, "e": e, "s": flux[obj] + f})
    df = add_residuals(df)
    ref = np.ones(len(df), bool)
    bright = (df.F.abs() > mag_to_njy(21)).to_numpy()
    vt = visit_table(df, ref, bright)
    # offsets are relative to each object's mean, i.e. recovered up to the mean offset
    assert np.allclose(vt.off_frac.to_numpy(), off - off.mean(), atol=0.002)
    df["f_off"] = apply_visit_offsets(df, vt)
    groups = {"all": np.ones(n_obj, bool)}
    raw, _ = frac5_by_mag(df, n_obj, groups, [-np.inf, 21, 25], bands="r")
    cor, _ = frac5_by_mag(df, n_obj, groups, [-np.inf, 21, 25], flux="f_off", bands="r")
    assert raw[("r", "all", "-inf-21")] > 0.9 and cor[("r", "all", "-inf-21")] < 0.02


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
