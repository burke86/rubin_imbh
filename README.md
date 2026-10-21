# rubin_imbh

Variability selection of nuclear AGN, TDE and IMBH candidates in Rubin DP2 with LSDB.
Two sources of difference-image forced light curves (`--source`):

- **`object`** (default): `object_collection`, forced photometry on every coadd Object. No
  DIA detection is required, so low-amplitude nuclear variability (host-diluted dwarf AGN,
  early TDEs) is kept; hosts can be preselected with `--prefilter` / `--extended-only`.
- **`dia`**: `dia_object_collection`, DIA-detected objects only, for off-nuclear or
  hostless sources. In CDF-S only 93 of 1088 X-ray AGN with a coadd Object have a
  DiaObject at all, which is why the object source is the default.

The DP2 HATS catalogs live only at data-rights sites (RSP `/rubin/lsdb_data`, NERSC,
CC-IN2P3), so the pipeline runs there; it is tested locally on mock catalogs.

## Layout

- `rubin_imbh/varstats.py` — per-object, per-band statistics (pure pandas, vectorized)
- `rubin_imbh/pipeline.py` — LSDB wrapper: schema check, coarse cut, `map_partitions`
- `scripts/select_variables.py` — CLI (`inspect`, `tiles`, `run`; `--source object|dia`)
- `rubin_imbh/environment.py` — Object matching (dia source) and bright-star veto
- `rubin_imbh/calib.py` — calibration groups for either source
- `scripts/diagnose_cone.py` — false-positive diagnostics for a calibration cone
- `scripts/fit_floor.py` — per-band `--sys-frac` from a calibration cone, without rerunning
- `scripts/diagnose_visits.py` — per-visit offsets, bad visits and same-night correlation on one cone (`rubin_imbh/visits.py`)
- `scripts/diagnose_objects.py` — outlier tails, airmass correlation, morphology and bright-star proximity of false positives (`rubin_imbh/objects.py`)
- `scripts/validate_xray.py` — completeness and X-ray purity against CDF-S 7 Ms / E-CDF-S 250 ks AGN (VizieR)
- `scripts/depth_map.py` — coadd 5σ depth per HEALPix pixel from object_collection; finds DP2's depth layers
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
| `nVarBands`, `maxVarSig` | over `--select-bands` (default `riz`): bands with `varSigClip ≥ min_sig` and `nEpoch ≥ min_epochs` |
| `all_nEpoch`, `all_dof`, `all_chi2dof`, `all_varSig`, `all_chi2dofClip`, `all_varSigClip` | combined-band statistic over `--combo-bands` (default `riz`): one χ² over all epochs, each band about its own mean, dof = Σ(n_b − 1); the leave-one-out drops the most deviant epoch over all bands and is standardized against a null simulated per band split (cached; single-band table above 60 epochs) |
| `varCombo`, `isVariable` | combined criterion (`all_varSigClip ≥ min_sig`, `all_dof ≥ --min-combo-dof` = 4); `isVariable` is per-band OR combined by default (`--combined either|only|off`) |

Selection uses the clipped significance so a single unflagged bad epoch cannot make an
object variable (`--no-robust` reverts to `varSig`). u and y are computed but do not count
by default: in DP2 they have ~10 epochs and showed no variable/reference separation.

## Object-source calibration and validation (Oct 2026)

ECDFS 30′ cone (255k Objects): median χ²/dof of faint Objects is 0.96–1.0 in every band, so
`--err-scale 1.0`. Faint (>21 mag) extended Objects are flagged at 0.2–1.5% (r, i, z); the
excess is a brightness-proportional floor brighter than ~20 mag, with flagged `fvar`
0.5–1%. Objects without `refExtendedness` (~9%) have degenerate photometry (median
χ²/dof 0 at 19–22 mag): use `--prefilter "refExtendedness >= 0"`.

CDF-S validation (20′ cone, 103k Objects): 1045 of 1095 X-ray AGN with an Object are now
tested (93 with the DIA source). `validate_xray.py --floors` on a floor-free run:

| floor | riz ≥1 compl. r≤22 / r≤24 | X-ray purity (ext) | r compl. r≤22 | r purity |
|---|---|---|---|---|
| 0 | 0.35 / 0.11 | 0.066 (0.085) | 0.28 | 0.11 |
| 0.5% | 0.28 / 0.10 | 0.088 (0.084) | 0.28 | 0.15 |
| 1% | 0.26 / 0.09 | 0.114 (0.081) | 0.26 | 0.20 |
| 2.5% (`fit_floor`) | 0.22 / 0.08 | 0.114 (0.070) | 0.22 | 0.21 |

The floor mainly removes bright point sources; the extended-host X-ray fraction stays at
~8–16% for any floor. Completeness is about the same as the DIA source (11% at r ≤ 24;
45% at r < 21, 0% at 23–24), so the AGN without DiaObjects are mostly not significantly
variable over DP2's ~9 months at these depths. `fit_floor.py`'s rule (flatten the
brightest bins) over-picks floors; choose on the X-ray metrics instead (~0.5–1%).
Non-X-ray extended "variables" plausibly include supernovae and TDEs, which needs
light-curve shape and nuclear-offset information to separate.

### WFD pilot (tile 753, RA 321, Dec −30; Oct 2026)

1.39M Objects in 54 deg², ~15 s compute. DP2 WFD light curves are sparse: median epochs per
Object u/g/r/i/z/y = 0/1/2/4/3/0, so the per-band criterion (≥10 epochs) applies to <1%
of Objects. Where it does (i band), median χ²/dof is 0.89–0.97 and faint false positives
are 0–0.5%. The combined-band statistic exists for this regime: on WFD-like mocks it is
calibrated (mean +0.05σ, sd 1.01) and recovers 37/66/76/83% of objects with 10/20/30/50%
rms variability with no false positives, so in the WFD it targets large-amplitude
variability (TDEs, strongly variable AGN), not low-amplitude dwarf AGN.

## DIA-source calibration history (Oct 2026)

Everything below up to "What it does" was measured with `--source dia`; the object source
still needs its own calibration (error scale, floors, environment), see step 5.

### Calibration (ECDFS, 0.5° cone)

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

`diagnose_objects.py` on ECDFS (Oct 2026):
- **Bright stars**: reference DiaObjects within 2″ of an r<18 point source are flagged
  91–100%, at 2–10″ 38–62%, beyond 10″ 2–12%. The 0–10″ zone holds ~8% of reference
  objects but 40–60% of their false positives.
- **No coadd counterpart**: only 24% of DiaObjects have an Object within 1″.
- **Morphology**: extended reference objects are the cleanest (flagged 0.9% z, 2.2% r,
  4.3% i, 7.7% g), below point-like ones; galaxy subtraction residuals are not the issue.
- **Airmass**: 52% of flagged g-band reference objects have |corr(z, airmass)| > 0.3 vs 10%
  of unflagged (weaker in i, r, z); per-visit offsets are tiny, so it is object-dependent,
  plausibly colour-dependent (DCR or second-order extinction).
- **Tails**: the top 10% of objects carry ~89% of 4σ outliers; removing all |z|>4 epochs
  rescues only 20–42% of flags.

Cut sequence on ECDFS (reference false positives per deg², single band, 5σ):

| band | all | + bright-star veto (10″) | + Object within 1″ | extended only |
|---|---|---|---|---|
| g | 3855 | 2427 | 215 | 126 (7.5%) |
| r | 2203 | 998 | 74 | 36 (2.1%) |
| i | 3691 | 2298 | 237 | 70 (4.1%) |
| z | 1867 | 775 | 65 | 13 (0.8%) |

The Object match is the biggest lever and removes most of the faint g/i excess. In g the
sign of corr(z, airmass) follows g−i colour (Spearman 0.33 for flagged objects; blue
objects flagged 12% vs red 3%), so g is excluded from selection by default. In i, bright
(19–21 mag) point sources are still flagged 20% vs 7% for extended objects, with no colour
trend; possibly a PSF/brightness residual.

### X-ray validation (CDF-S, 20′ cone at 53.12, −27.81; Oct 2026)

Of 1390 X-ray AGN in coverage, 1088 have a coadd Object within 1.5″ but only **93 have a
DiaObject at all**, so DIA detection, not the variability statistic, caps completeness.
`riz ≥1` (envClean) selects 11% of X-ray AGN with r ≤ 24 (42% at r < 21, 21% at 21–22,
8% at 22–23, 0% at 23–24). Of selected objects in the footprint, 8% match an X-ray AGN
(chance 0.3%) against a 5.8% base rate among all envClean DiaObjects: extended hosts 15–25%,
point-like 3–8% (below the base rate, i.e. dominated by stars and residual systematics).
`r` alone gives the highest X-ray fraction (12%; 25% for extended hosts).

Open issues: (1) a DiaObject-based search misses most AGN; measuring variability on every
coadd Object (object_collection's nested forced photometry) is needed, especially for
low-amplitude dwarf AGN; (2) point-like selections need a star veto (e.g. Gaia parallax and
proper motion) and the bright point-source i-band residual understood; (3) DP2's ~9-month
baseline limits amplitudes.

A floor of a few percent of the *total* direct-image flux limits sensitivity to low-amplitude,
host-diluted variability, which is exactly the dwarf-AGN regime, so keep it as low as the
data allow. ECDFS is a deep-drilling field; wide-survey areas
have far fewer epochs, so revisit `--min-epochs` there.

`excessVar` is diluted by host light because `psfFlux` includes the galaxy, which
matters for dwarf hosts.

## What it does

1. **Cheap row cuts** before any light curve is read: `--prefilter` (object source, e.g.
   host selection) or `nDiaSources >= 3` (dia source).
2. **Per-band statistics** on the nested forced-photometry light curve, after dropping
   flagged epochs: χ²/dof about a constant, its outlier-robust significance, and the
   excess variance.
3. **Select** objects with `varSigClip >= 5` and `nEpoch >= 10` in at least one of r, i, z,
   keeping their statistics and light curves.
4. **Environment**: bright-star veto (r < 18 point sources within 10″) for both sources;
   the dia source can also require a coadd Object within 1″ (`--object-catalog`).

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

**3. Check the environment.** The default catalogs are the explicit DP2 paths
(`/rubin/lsdb_data/dp2/object_collection`, `.../dia_object_collection`); the top-level
symlinks follow the current release.

```bash
python -c "import lsdb; print(lsdb.__version__)"     # 0.10.4 as of Oct 2026; tested on 0.6 and 0.10
```

**4. Inspect the schema.** The object source needs difference-image fluxes
(`psfDiffFlux`) in its nested light curve; the light-curve column is auto-detected as the
nested column holding them, and the run stops if there is none.

```bash
python scripts/select_variables.py inspect > obj_inspect.txt        # object_collection
python scripts/select_variables.py --source dia inspect > dia_inspect.txt
```

**5. Calibrate on a small field** (ECDFS, a deep-drilling field; DP2 barely covers Stripe
82): statistics for every Object, no selection, no light curves, then the diagnostics:

```bash
python scripts/select_variables.py run --cone 53.13 -28.10 1800 \
    --no-select --stats-only --n-workers 4 --memory-limit 6GB --out ~/obj_cone_stats.parquet
python scripts/diagnose_cone.py ~/obj_cone_stats.parquet
python scripts/fit_floor.py ~/obj_cone_stats.parquet
```

Nearly every Object is constant, so the bulk sets the error scale: if the median
`r_chi2dof` is ~1.3, use `--err-scale ~1.15` (√median). The DIA-source value was 1.07.
Keep `--n-workers × --memory-limit` within the server size.

**6. Validate** against Chandra AGN (cone on CDF-S):

```bash
python scripts/select_variables.py run --cone 53.12 -27.81 1200 --no-select --stats-only \
    --err-scale <from step 5> --out ~/obj_cdfs_stats.parquet
python scripts/validate_xray.py --stats ~/obj_cdfs_stats.parquet --out ~/obj_cdfs_selected.csv
```

**7. Full run in HEALPix tiles**, only after steps 4–6 look right:

```bash
python scripts/select_variables.py tiles --order 3     # preview
python scripts/select_variables.py run --err-scale <from step 5> --tile-order 3 \
    --n-workers 8 --out ~/rubin_imbh/dp2_variables
```

For the dia source add `--source dia` (and `--min-dia-sources`, `--object-catalog`).

### Environment cuts (lsdb >= 0.10)

Object source: the bright-star veto runs against the input catalog itself (`--no-veto` to
skip), adding `brightStarDist` and `envClean`; a bright star's own row is flagged.

Dia source, with `--object-catalog /rubin/lsdb_data/dp2/object_collection`: selected objects are
crossmatched to the nearest coadd Object within `--match-arcsec` (default 1″; unmatched
objects are dropped unless `--keep-unmatched`), and to r < `--bright-star-mag` (18) point
sources within `--veto-arcsec` (10″). Added columns: `objSep`, the Object's id,
extendedness and `{band}_psfFlux`, `brightStarDist` (NaN if no bright star within the veto
radius) and `envClean`. Bright stars are pulled once into a small in-memory catalog with a
margin wider than the veto radius (the veto exceeds object_collection's margin) and cached
as `bright_stars_r18*.parquet` next to `--out`.

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
