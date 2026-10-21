# rubin_imbh

Variability selection of AGN / IMBH candidates in Rubin DP2, using LSDB on the DP2
`dia_object_collection` (DiaObjects with nested difference-image forced photometry).

The DP2 HATS catalogs live only at data-rights sites (RSP `/rubin/lsdb_data`, NERSC,
CC-IN2P3), so the pipeline runs there; it is tested locally on mock catalogs.

## Layout

- `rubin_imbh/varstats.py` — per-object, per-band statistics (pure pandas, vectorized)
- `rubin_imbh/pipeline.py` — LSDB wrapper: schema check, coarse cut, `map_partitions`
- `scripts/select_variables.py` — CLI (`inspect`, `run`)
- `tests/test_pipeline.py` — mock-data tests (`python tests/test_pipeline.py`)

## Statistics (per band, on good forced-photometry epochs)

| column | meaning |
|---|---|
| `{b}_nEpoch` | epochs passing flag cuts |
| `{b}_chi2dof` | χ²/dof of `psfDiffFlux` about its weighted mean |
| `{b}_varSig` | Gaussian-equivalent significance of that χ² (Wilson–Hilferty) |
| `{b}_excessVar` | (var − ⟨σ²⟩) / median(`psfFlux`)², normalized by direct-image flux |
| `{b}_fvar` | √excessVar (fractional rms) |
| `nVarBands`, `maxVarSig`, `isVariable` | bands with `varSig ≥ min_sig` and `nEpoch ≥ min_epochs` |

`excessVar` is diluted by host light because `psfFlux` includes the galaxy, which
matters for dwarf hosts.

## Running on the RSP

```bash
# copy this folder to ~/rubin_imbh (JupyterLab upload, or git clone once it has a remote)
cd ~/rubin_imbh

# 1. confirm column names (nested field names and flags are DP1-based guesses)
python scripts/select_variables.py inspect

# 2. calibrate on one field: stats for every object, no selection
python scripts/select_variables.py run --cone 28.5 0.4 1800 --no-select \
    --out ~/rubin_imbh/test_cone_stats.parquet
#    -> look at the r_chi2dof distribution of the bulk population; if its median is
#       ~1.3, rerun with --err-scale ~1.15 (sqrt of the median) so the null is ~1

# 3. full selection in HEALPix tiles, HATS output with nested light curves
python scripts/select_variables.py tiles --order 3          # preview tile count
python scripts/select_variables.py run --err-scale 1.15 --tile-order 3 --n-workers 8 \
    --out ~/rubin_imbh/dp2_variables
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

Column names can be overridden with `--lc-column`, `--flux-column`, `--sci-flux-column`.
Missing flag columns are dropped with a warning; missing required columns raise.
