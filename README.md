# rubin_imbh

Variability selection of AGN / IMBH candidates in Rubin DP2, using LSDB on the DP2
`dia_object_collection` (DiaObjects with nested difference-image forced photometry).

The DP2 HATS catalogs live only at data-rights sites (RSP `/rubin/lsdb_data`, NERSC,
CC-IN2P3), so the pipeline runs there; it is tested locally on mock catalogs.

## Layout

- `rubin_imbh/varstats.py` — per-object, per-band statistics (pure pandas, vectorized)
- `rubin_imbh/pipeline.py` — LSDB wrapper: schema check, coarse cut, `map_partitions`
- `scripts/select_variables.py` — CLI (`inspect`, `tiles`, `run`)
- `scripts/diagnose_cone.py` — false-positive diagnostics for a calibration cone
- `scripts/fit_floor.py` — per-band `--sys-frac` from a calibration cone, without rerunning
- `scripts/diagnose_visits.py` — per-visit offsets, bad visits and same-night correlation on one cone (`rubin_imbh/visits.py`)
- `scripts/diagnose_objects.py` — outlier tails, airmass correlation, morphology and bright-star proximity of false positives (`rubin_imbh/objects.py`)
- `scripts/simulate_clip_null.py` — regenerates the null table for `varSigClip`
- `tests/test_pipeline.py` — mock-data tests (`python tests/test_pipeline.py`)

## Statistics (per band, on good forced-photometry epochs)

| column | meaning |
|---|---|
| `{b}_nEpoch` | epochs passing flag cuts |
| `{b}_medFlux`, `{b}_rmsErr` | median direct-image flux (nJy, includes host) and rms per-epoch error after rescaling |
| `{b}_chi2dof` | χ²/dof of `psfDiffFlux` about its weighted mean |
| `{b}_varSig` | Gaussian-equivalent significance of that χ² (Wilson–Hilferty) |
| `{b}_chi2dofClip`, `{b}_varSigClip` | same with the single most deviant epoch removed (exact leave-one-out); `varSigClip` is standardized against the simulated null of this statistic (`scripts/simulate_clip_null.py`), since dropping the largest residual biases χ² low |
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
is what motivated the clipped statistic. With `--err-scale 1.07` the reference median is
0.96–1.00, but 7–15% of reference objects still exceed 5σ after clipping (≈10⁴ variables
per deg² in total, far above expectations), i.e. a multi-epoch systematic remains;
`scripts/diagnose_cone.py` splits this by brightness to set `--sys-frac`.

v3 (`diagnose_cone.py`): the reference 5σ fraction climbs from ~10% at 22–25 mag to 100%
brighter than ~19 mag, and the flagged bright objects have `fvar` ≈ 3% (g) and 1.5–2.5%
(riz): a brightness-proportional floor from subtraction residuals, larger in g (DCR). The
>5-detection group shows the same `fvar` ≲ 1% at the bright end, so many bright
DiaObjects exist because of these residuals. `scripts/fit_floor.py` picks per-band floors
from the same parquet. Separately, i band at 21–23 mag has reference median χ²/dof ≈ 1.6,
which a flux-proportional floor cannot fix; possibly correlated same-night visits in the
deep-drilling sequences (to check by nightly binning).

`fit_floor.py` on v3: a 1.5–2% floor (2.5% in g) removes nearly all bright >5-detection
"variables" (they are residuals), but bright 1–2-detection objects stay flagged even at a
5% floor, and the 21–25 mag excess (g ~27%, i ~31%) does not respond at all. So the
excess is not a single per-object floor. Leading hypothesis: visit-level problems (bad
subtractions, per-visit zeropoint offsets shared by all objects in an exposure), which
`scripts/diagnose_visits.py` measures from the reference ensemble. Per-visit offsets can
be *removed* (ensemble correction) rather than absorbed into inflated errors, which keeps
sensitivity to low-amplitude variability.

`diagnose_visits.py` on ECDFS rejects the visit hypothesis: coherent per-visit offsets are
only 0.1–0.2%, no visit has excess core scatter (robust sd 1.01–1.18), and removing offsets,
dropping visits or nightly binning barely changes the false-positive rate. Instead every
visit has heavy tails: 1.6–4.6% of reference epochs beyond 4σ (Gaussian: 0.006%), worst
in g then i, matching the false-positive ordering. So the excess is per-object, per-epoch
and not visit-coherent; `scripts/diagnose_objects.py` tests airmass correlation (DCR),
morphology (galaxy residuals), bright-star proximity, and whether flags are driven by a
few outlier epochs.

A floor of a few percent of the *total* direct-image flux limits sensitivity to low-amplitude,
host-diluted variability, which is exactly the dwarf-AGN regime, so keep it as low as the
data allow. ECDFS is a deep-drilling field; wide-survey areas
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
