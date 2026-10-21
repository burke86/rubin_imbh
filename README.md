# rubin_imbh

Variability selection of AGN / IMBH candidates in Rubin DP2, using LSDB on the DP2
`dia_object_collection` (DiaObjects with nested difference-image forced photometry).

The DP2 HATS catalogs live only at data-rights sites (RSP `/rubin/lsdb_data`, NERSC,
CC-IN2P3), so the pipeline runs there; it is tested locally on mock catalogs.

## Layout

- `rubin_imbh/varstats.py` — per-object, per-band statistics (pure pandas, vectorized)
- `rubin_imbh/pipeline.py` — LSDB wrapper: schema check, coarse cut, `map_partitions`
- `scripts/select_variables.py` — CLI (`inspect`, `tiles`, `run`)
- `tests/test_pipeline.py` — mock-data tests (`python tests/test_pipeline.py`)

## Statistics (per band, on good forced-photometry epochs)

| column | meaning |
|---|---|
| `{b}_nEpoch` | epochs passing flag cuts |
| `{b}_chi2dof` | χ²/dof of `psfDiffFlux` about its weighted mean |
| `{b}_varSig` | Gaussian-equivalent significance of that χ² (Wilson–Hilferty) |
| `{b}_chi2dofClip`, `{b}_varSigClip` | same with the single most deviant epoch removed (exact leave-one-out) |
| `{b}_excessVar` | (var − ⟨σ²⟩) / median(`psfFlux`)², normalized by direct-image flux |
| `{b}_fvar` | √excessVar (fractional rms) |
| `nVarBands`, `maxVarSig`, `isVariable` | over `--select-bands` (default `griz`): bands with `varSigClip ≥ min_sig` and `nEpoch ≥ min_epochs` |

Selection uses the clipped significance so a single unflagged bad epoch cannot make an
object variable (`--no-robust` reverts to `varSig`). u and y are computed but do not count
by default: in DP2 they have ~10 epochs and showed no variable/reference separation.

### Calibration (ECDFS, 0.5° cone, Oct 2026)

23.5k DiaObjects; median epochs g/r/i/z = 49/45/86/54, u/y ≈ 9–10. Objects with 1–2
DIA detections (reference) have median χ²/dof 1.10–1.15 in griz against ~0.985 expected,
so errors are underestimated by ~7%: use `--err-scale 1.07`. Objects with ≥3 detections
have median χ²/dof 3–7 in g/i. The reference group's heavy upper tail (mean ≫ median)
is what motivated the clipped statistic. ECDFS is a deep-drilling field; wide-survey areas
have far fewer epochs, so revisit `--min-epochs` there.

`excessVar` is diluted by host light because `psfFlux` includes the galaxy, which
matters for dwarf hosts.

## What it does

1. **Coarse cut** on the DiaObject summary row: `nDiaSources >= 3` (drops one-off
   detections, most of the catalog) before any light curve is read.
2. **Per-band statistics** on the nested forced-photometry light curve, after dropping
   flagged epochs: χ²/dof about a constant, its significance, and the excess variance.
3. **Select** objects with `varSig >= 5` and `nEpoch >= 10` in at least one band, keeping
   their statistics and light curves.

LSDB runs this lazily and in parallel over HEALPix partitions. DP2's HATS catalogs can
only be read where they are hosted, so full runs happen on the RSP; outputs are small
enough to download.

## Running on the RSP

**1. Log in and open a terminal.** At <https://data.lsst.cloud> go to Notebooks, pick the
largest server size and the recommended image, then File → New → Terminal.

**2. Get the code** (private repo: use a GitHub personal access token as the password):

```bash
git clone https://github.com/burke86/rubin_imbh ~/rubin_imbh
cd ~/rubin_imbh
```

**3. Point at the data.** `/rubin/lsdb_data/dia_object_collection` is a symlink to the
current release; use the explicit DP2 path so a later release doesn't change it. Set
`CAT` again in each new terminal.

```bash
python -c "import lsdb; print(lsdb.__version__)"     # 0.10.4 as of Oct 2026; tested on 0.6 and 0.10
CAT=/rubin/lsdb_data/dp2/dia_object_collection
```

**4. Inspect the schema** (confirmed against DP2 in Oct 2026; rerun after a new release):

```bash
python scripts/select_variables.py --catalog $CAT inspect > inspect.txt
cat inspect.txt
```

**5. Calibrate on a small field** (0.5° radius on ECDFS, a deep-drilling field; DP2 barely covers Stripe 82): statistics for every object,
no selection, no light curves:

```bash
python scripts/select_variables.py --catalog $CAT run --cone 53.13 -28.10 1800 \
    --no-select --stats-only --min-dia-sources 0 --n-workers 4 --memory-limit 6GB \
    --out ~/test_cone_stats.parquet
```

Look at the `r_chi2dof` distribution of the bulk (non-variable) population. DIA errors are
usually underestimated; if its median is ~1.3, use `--err-scale ~1.15` (√median) so the
null sits at 1. Keep `--n-workers × --memory-limit` within the server size.

**6. Full run in HEALPix tiles**, only after steps 4–5 look right:

```bash
python scripts/select_variables.py --catalog $CAT tiles --order 3     # preview
python scripts/select_variables.py --catalog $CAT run --err-scale 1.07 \
    --tile-order 3 --n-workers 8 --out ~/rubin_imbh/dp2_variables
```

### Tiling

`--tile-order K` loops over every order-K HEALPix pixel overlapping the catalog
(order 3 ≈ 54 deg², order 4 ≈ 13 deg²) and writes each to `<out>/oK_p<pixel>/`.
A `_DONE` marker holding the row count is written last, so rerunning the same command
after an interruption skips finished tiles and redoes partial ones. Rows are cut on
their `_healpix_29` index, so partitions coarser than a tile are never double counted.
Run one tile with `--tile K PIXEL`, e.g. to split work across sessions.
Each tile is its own HATS catalog; with `--stats-only`/`--no-select` it is a single
`stats.parquet` instead.

A `CommClosedError` traceback at the very end of a run is dask shutting down its
workers after the output is written; it is harmless (exit status 0).

### Columns

Only the needed columns are loaded: `diaObjectId`, `ra`, `dec`, `tract`, `nDiaSources`, and
from the forced-photometry light curve the band, fluxes, errors, flags, `midpointMjdTai`
and `visit`. The nested `diaSource` column (detections only, ~50 fields) is skipped
unless `--keep-dia-source` is given; the collection is ~1.1 TB, mostly nested data.

Column names can be overridden with `--lc-column`, `--flux-column`, `--sci-flux-column`.
Missing flag columns are dropped with a warning; missing required columns raise.
