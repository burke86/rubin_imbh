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
                },
                index=np.full(n_epoch, i),
            )
        )
    return base, pd.concat(rows)


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
    assert summ.isVariable[const].mean() < 0.005
    assert summ.isVariable[~const].all()
    fvar = st.loc[~const, "r_fvar"]
    assert 0.05 < np.nanmedian(fvar) < 0.3
    # without the flag cut the outliers make everything "variable"
    st_bad = lightcurve_stats(obj, lc, len(base), COLS, cuts, flags=())
    assert summarize(st_bad, cuts).isVariable[const].mean() > 0.5


def test_lsdb_end_to_end():
    import lsdb
    import nested_pandas as npd

    from rubin_imbh.pipeline import select_variables

    base, lc = mock_dia(n_const=500, n_var=20)
    nf = npd.NestedFrame(base).add_nested(lc, "diaObjectForcedSource")
    with tempfile.TemporaryDirectory() as tmp:
        lsdb.from_dataframe(nf, lowest_order=2, highest_order=9, threshold=100).to_hats(
            Path(tmp) / "mock_dia", catalog_name="mock_dia"
        )
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
        lsdb.from_dataframe(nf, lowest_order=2, highest_order=9, threshold=600).to_hats(
            Path(tmp) / "mock_dia", catalog_name="mock_dia"
        )
        cat = lsdb.open_catalog(Path(tmp) / "mock_dia")
        tiles = tile_pixels(cat, 6)
        assert len(tiles) > cat.npartitions
        cuts = Cuts(min_dia_sources=0)
        ids = []
        for pix in tiles:
            ids += select_variables(cat, COLS, cuts, select=False, tile=(6, pix)).compute().diaObjectId.tolist()
        assert sorted(ids) == sorted(base.diaObjectId)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
