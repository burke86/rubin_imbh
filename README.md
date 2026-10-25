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

**Data policy:** DP2 tables and anything that can recreate them (stats, selections, the survey
library, photo-z exports) are proprietary: they never go in this repo or to non-data-rights
holders. See [DATA_POLICY.md](DATA_POLICY.md) for the relevant Rubin rules and how they map onto
this project's files.

## Layout

The light-curve statistics and the amortized inference (training sets, posterior estimator,
priors) live in [jaxpz](https://github.com/burke86/jaxpz) (`jaxpz.lightcurve`, `jaxpz.npe`,
`jaxpz simulate|train|predict|redraw|fetch-nzm`); install it alongside this repo
(`pip install -e ../jaxpz`, or on the RSP `pip install --user git+https://github.com/burke86/jaxpz`).

- `rubin_imbh/pipeline.py` — LSDB wrapper: schema check, coarse cut, `map_partitions`
- `scripts/select_variables.py` — CLI (`inspect`, `tiles`, `run`; `--source object|dia`)
- `rubin_imbh/environment.py` — Object matching (dia source) and bright-star veto
- `rubin_imbh/calib.py` — calibration groups for either source
- `scripts/diagnose_cone.py` — false-positive diagnostics for a calibration cone
- `scripts/fit_floor.py` — per-band `--sys-frac` from a calibration cone, without rerunning
- `scripts/diagnose_visits.py` — per-visit offsets, bad visits and same-night correlation on one cone (`rubin_imbh/visits.py`)
- `scripts/diagnose_objects.py` — outlier tails, airmass correlation, morphology and bright-star proximity of false positives (`rubin_imbh/objects.py`)
- `scripts/validate_xray.py` — completeness and X-ray purity against CDF-S 7 Ms / E-CDF-S 250 ks AGN (VizieR)
- `scripts/summarize_selection.py` — combine a tiled run; counts by tile, latitude, morphology, magnitude, route; sky map
- `scripts/inspect_lightcurves.py` — light curves of sampled selected objects per tile, and which nights their excursions share
- `scripts/gaia_flag.py` — Gaia DR3 crossmatch (remote HATS): flags stars (parallax/PM > 5σ), quasar candidates, Gaia variables
- `amortized/simulator_proto.py` — jaxsedfit as a batched host+AGN LSST photometry simulator (feasibility)
- `scripts/export_survey_library.py` — DP2 noise/cadence library for the simulator (RSP)
- `scripts/attach_object_columns.py` — attach cModel (or any Object) columns to a selection by objectId (RSP)
- `scripts/validate_real.py` — jaxpz posteriors for real DP2 objects vs COSMOS2020 / CDF-S X-ray / Milliquas references (`rubin_imbh/references.py`) and DP2 photo-z; `--condition-on-specz`
- `scripts/depth_map.py` — coadd 5σ depth per HEALPix pixel from object_collection; finds DP2's depth layers
- `scripts/simulate_clip_null.py` — regenerates the null table for `varSigClip`
- `tests/` — mock-data tests (`python -m pytest`; needs lsdb and jaxpz)

## Statistics (per band, on good forced-photometry epochs)

| column | meaning |
|---|---|
| `{b}_nEpoch` | epochs passing flag cuts |
| `{b}_medFlux`, `{b}_rmsErr` | median direct-image flux (nJy, includes host) and rms per-epoch error after rescaling |
| `{b}_chi2dof` | χ²/dof of `psfDiffFlux` about its weighted mean |
| `{b}_varSig` | Gaussian-equivalent significance of that χ² (Wilson–Hilferty) |
| `{b}_worstZ` | signed significance of the epoch (or night) the leave-one-out removes; added-light artefacts such as asteroids are positive |
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

### Ecliptic contamination and nightly binning (Oct 2026)

The first full-DP2 pass (per-band criterion, epoch-level statistics; 42 of 147 tiles before
the server stopped) selected 434k objects, dominated by high-latitude tiles on the ecliptic:
tile 290 (ecliptic latitude +1.8°) 78k, falling to ~1k by |β| ≈ 11° (pilot tile 753 at
−13.9°: 179). In tile 290, 84% are flagged in a single band and epochs are higher (median
17/15/6 in r/i/z), consistent with asteroids crossing forced-photometry positions on one
night (visit pairs defeat a leave-one-epoch-out). Statistics now use **nightly means** by
default (`Cuts.nightly`, `--no-nightly` to disable): each band's epochs are averaged within a
night (dayObs boundary 12:00 UTC) before any statistic, so the leave-one-out removes the
worst night and variability must persist over ≥2 nights. `nEpoch` then counts nights.

### Full DP2 run v2 (Oct 2026)

99 tiles with |b| ≥ 25° plus 14 lower-latitude tiles finished before the cut. 3.19M rows
selected *before* the bright-star flag (`envClean`) is applied; the run adds `envClean` but
does not filter on it (767k selected point sources brighter than r = 18 are mostly vetoed
bright stars). 75% enter only through the combined-band statistic and 86% lie within 10° of
the ecliptic, including high-Galactic-latitude tiles (761, 764, 759, 758: b = −31 to −45°,
|β| < 3°; tile 753 at β = −14° has ~2k). Under investigation with `inspect_lightcurves.py`:
until understood, treat combined-route selections near the ecliptic as unreliable. No Gaia
star veto had been applied: `gaia_flag.py` adds one (stars, including high-proper-motion stars
that fake variability in difference images; Gaia quasar candidates are kept).

### Clean variable sample (Oct 2026)

`summarize_selection.py ~/dp2_variables_v2_gaia.parquet --out ~/dp2_clean`:
3,192,268 selected → 1,725,492 `envClean` → **731,886 after dropping Gaia stars**
(`gaia_flag.py`: 60% of all selections are Gaia stars with parallax/PM > 5σ, 83% of the
point-like ones; 36k Gaia quasar candidates kept). The clean sample is 66% extended; routes:
per-band only 28k, combined only 515k, both 188k. The DDF tiles now rank high (ELAIS-S1 552:
23.6k, EDFS 536: 17.6k). For science use also drop |b| < 25° (tiles finished before the cut,
e.g. 671, 670, 458, 677: 100% combined-route): `d[d.gal_b.abs() >= 25]`.

**Open issue: ecliptic concentration.** 82% of the clean sample lies within |β| < 10°, the
same fraction as before the Gaia cut, and it includes galaxies (tiles 761, 764, 759, 758 at
b = −31 to −45°: 26–38k each, ~72% extended). Leading explanation: DP2 sampled the ecliptic
band much more densely (tile 290: median 17/15/6 nights in r/i/z vs 2/4/3 in tile 753), so
more real variables are detectable, as in the DDFs; not yet excluded: an ecliptic-specific
systematic. Pending test: selected fraction of faint galaxies vs `all_dof` in an ecliptic tile
(761) and an off-ecliptic one (724), from `--no-select` runs (command in the conversation log;
equal fractions at equal dof → sampling, not a systematic). `inspect_lightcurves.py` is the
follow-up if a systematic is indicated.

## Status and next steps (Oct 2026)

Done: variability selection over DP2 (object source, nightly means, per-band + combined
statistics, environment and Gaia cuts), calibrated on ECDFS/CDF-S and a WFD pilot.
Outputs on the RSP: `~/dp2_variables_v2/` (tiles), `~/dp2_variables_v2_gaia.parquet`,
`~/dp2_clean.parquet`.

Next:
1. **Amortized joint inference** (AGN luminosity + photo-z + stellar mass, using the
   host-free variability amplitude): a neural posterior estimator trained on jaxsedfit
   simulations, with DP2 noise/cadence drawn from an empirical survey library and the DDFs
   (CDF-S/COSMOS spec-z, X-ray AGN, deep masses) as the test set. SN/TDE contaminants
   ignored for now.
2. Ecliptic test above; WFD validation against known quasars (e.g. Milliquas).
3. Later: classification (stochastic vs flare, DIA nuclear offset), coadd total-flux columns
   for host photometry.

## Amortized joint inference (started Oct 2026)

(The code below now lives in [jaxpz](https://github.com/burke86/jaxpz): `rubin_imbh/amortized/*`
→ `jaxpz.*`, `make_training_set.py` → `jaxpz simulate`, `train_flow.py` → `jaxpz train`,
`redraw_coadds.py` → `jaxpz redraw`, `infer.py` → `jaxpz predict`, `fetch_cosmos2020_nzm.py`
→ `jaxpz fetch-nzm`.)

Goal: posteriors for redshift, stellar mass, SFH, dust and AGN luminosity per object from
coadd photometry plus the host-free variability amplitude, via a neural posterior estimator
trained on jaxsedfit simulations (DDFs as the test set).

Feasibility (`amortized/simulator_proto.py`, jaxcpu env, laptop CPU): jaxsedfit's NumPyro
model wrapped with `seed` + `substitute` + `trace` (likelihood off, `include_components=True`)
and `jit(vmap(...))` simulates LSST ugrizy total, host and AGN fluxes (mJy) at **0.47 ms per
object** after a 3–4 s compile (context build 4 s, wide redshift grid 0.02–3): ~8 min per
10⁶ simulations, no GPU needed for simulating. Sampled sites include redshift,
log_stellar_mass, delayed-SFH age/tau, log_ebv_gal/agn, log_agn_amp (ln λL_λ(5100 Å) in W),
disk slopes, torus, line strengths/widths, dust_alpha. jaxsedfit's fitting priors are not
population priors: log-uniform E(B−V) up to ~10 mag gives near-zero fluxes, SFH age is not
bounded by the age of the universe, and the default AGN-luminosity prior is seeded from the
observed photometry, so the simulator needs its own population priors.

Code split: survey-agnostic SED pieces go to jaxsedfit as PRs; LSST/DP2 specifics stay
here. jaxsedfit PR #121 (draft, branch `agent/simulate-api`, worktree
`../jaxsedfit-simulate-pr`): public `jaxsedfit.simulate` / `simulation_config` /
`SimulationResult` / `make_simulator` (batched simulation with fixed and prior-drawn
parameters, component fluxes incl. `variable_agn_fluxes`), plus a fix for a tracer leak in
the cached jax_cosmo distance table that broke repeated jit traces. Planned later PR:
per-filter variability amplitude as an optional observable (≈ f_var × `variable_agn_fluxes`).

Training-set generator (`rubin_imbh/amortized/`, `scripts/make_training_set.py`):
population priors (z 0.02–3, log M* 6.5–12, half-normal E(B−V), 50% AGN with λL5100
10^40–10^46 erg/s; SFH age bounded by the age of the universe through jaxsedfit's redshift-
dependent prior) → `jaxsedfit.simulate` (LSST ugrizy total and `variable_agn_fluxes`) →
r < 25.5 → `observe()`: coadd noise, cadence and nightly errors borrowed from a brightness-
matched library object; one DRW shared by all bands with rms = f_var(λ_rest) × variable AGN
flux (f_var is the asymptotic SF∞/√2 amplitude, median 0.15 at 5100 Å, 0.4 dex; τ_rest median
200 d, 0.3 dex), sampled at the real nights; reduced with the pipeline's `lightcurve_stats`.
The DRW step is what maps L_AGN to the variability DP2 can record: over a ~9-month baseline a
long-τ AGN shows only part of its asymptotic variance (tested against
E[S²] = σ²(1 − Σρ_ij/n(n−1))), so without it the network would underestimate L_AGN.
Missing data: every band's features carry a presence flag (`has_<b>`, `hasvar_<b>`), and the
gaps of the borrowed library object are inherited (the library is matched on i band, present
for ~99% of Objects, so e.g. the 54% of WFD Objects without an r coadd are represented); on
top of that `--band-dropout` (default 0.05) removes one random band from that fraction of
objects, so patterns absent from the library are still seen. Extra inputs (WISE, X-ray, spec-z) need retraining.
Smoke test (mock DDF library, 4000 objects, 17 s incl. compile): variable fraction of AGN rises
1% → 13% → 25% → 51% → 75% from λL5100 = 10^40–42 to 10^45–46 erg/s; 0 false positives
without AGN. Survey library: `scripts/export_survey_library.py` (run on the RSP; DDF cones plus
WFD cones at several ecliptic latitudes); add `--strip-ids` (drops objectId and coordinates,
shuffles rows) for a copy that leaves the RSP ([DATA_POLICY.md](DATA_POLICY.md)).

Neural posterior estimator (`rubin_imbh/amortized/npe.py`, `scripts/train_flow.py`; flowjax
19 + equinox 0.13 installed in the jaxcpu env with jax pinned at 0.6.2): masked autoregressive
flow with rational-quadratic splines conditioned on x (coadd luptitudes + log errors; per-band
excess-variance S/N, clipped significance, nights; combined statistic), θ = (z, log M*,
log λL5100, log E(B−V)_gal, log SFH age), standardized. The no-AGN half of the prior is
continuous at λL5100 = 10^36–40 erg/s (flows handle point masses badly).

First result (mock DDF library, 36k training / 4k held-out, 60 epochs, ~90 s CPU training):
posteriors calibrated (68/95% coverage 0.66–0.70 / 0.94); photo-z NMAD 0.089 with 28%
outliers (weak: small training set, uniform 0–3 redshift prior, crude mock noise); log M*
scatter 0.5 dex. For variable objects, variability inputs vs photometry-only (ablation,
`--no-variability`): L_AGN scatter 0.51 vs 1.7 dex (bias −0.04 vs −1.06, posterior width 1.2 vs
5.6 dex); log M* bias +0.08 vs +0.82 dex (AGN light mistaken for stars without variability);
coverage 0.67/0.94 vs 0.50/0.89. The variability model was still improving at epoch 60, so
all-object comparisons are not yet fair.

Empirical magnitude–redshift prior (`rubin_imbh/amortized/nz_prior.py`,
`scripts/fetch_cosmos2020_nzm.py`, `train_flow.py --nzm`): p(z | r) from COSMOS2020 CLASSIC
(LePhare `lp_zbest` vs HSC r auto; 711,918 clean-area galaxies, 237,751 at r < 25.5, 5.1% of
those at z > 3, beyond the current 0–3 training range), applied by resampling the training set
with w = p_COSMOS(z|m) / p_train(z|m) (conditional only; training magnitudes kept). Mock DDF
pool, 50k training each, same realistic held-out set (6k, COSMOS-like p(z|m)), trained to
convergence:

| | broad prior | COSMOS p(z\|m) |
|---|---|---|
| photo-z NMAD | 0.076 | 0.020 |
| outliers (\|Δz\|/(1+z) > 0.15) | 25% | 9.9% |
| log M* scatter | 0.47 dex | 0.24 dex |
| z coverage 68/95% | 0.66/0.94 | 0.66/0.92 |

Calibration slips slightly with resampling (duplicated objects; some parameters at 0.88–0.92
for 95%); more simulations should fix it. That run capped weights at 50× the median, which
under-corrects bright magnitudes; the cap is now 10⁴.

Simulator now spans z = 0.02–4 and adds per-band 1% zero-point offsets to the coadd fluxes.
Real-data validation (`scripts/validate_real.py`, `rubin_imbh/amortized/real.py`,
`references.py`): converts pipeline stats into the training observables (`--keep-columns` on
`select_variables.py` carries coadd total-flux columns; run it with the simulator's floor),
samples posteriors, matches COSMOS2020 photo-z or CDF-S 7 Ms X-ray sources (spec-z flagged,
absorption-corrected L_X), and reports photo-z NMAD / outliers / bias by r magnitude for ours
and for DP2's photo-z on the same objects, 68% coverage, and L5100 vs L_X for X-ray AGN.
On mock "real" data it reproduces evaluate() (NMAD 0.021, outliers 10.6%); by magnitude
NMAD 0.013 / outliers 1% at r < 21 rising to 0.12 / 37% at r = 24–25.5.

### Real survey library (Oct 24–25, 2026)

Library: `export_survey_library.py --depth-map … --strip-ids` (cModel coadds; 167,685 rows,
60k DDF + 108k WFD). DDF reaches r(S/N=5) ≈ 25.7 with ≥ 5 riz nights for every object; WFD
reaches ≈ 24.6 and is patchy: no r coadd for 54% of Objects, no z for 53%, no u/y for ~73%
(g and i nearly complete); median WFD nights per band 0–5.

Simulator fixes found against real COSMOS statistics:
- library matched on i, not r (matching on r never drew the missing-r half of WFD);
- the 0.75% floor's scatter is now injected (`inject_sys`): before, bright simulated galaxies
  had median all_varSigClip −2 vs +0.2 real; after, the simulated null is N(0, 1) at all
  magnitudes (real COSMOS galaxies: median +0.2, width 1.07; r = 18–20: −0.5);
- p(z|m) weights use the intrinsic `true_r` (observed r is NaN after band dropout, and NaN
  fell in the faintest bin: ESS 26k → 9k);
- …and specifically the host magnitude `true_r − varagn_r`: COSMOS2020's p(z|m) describes
  galaxies, and applied to host + AGN it left only 0.2% of bright (r < 21) AGN-dominated
  training objects at z > 0.8 (now 15%), so real bright quasars at z = 1–3 were all placed at
  z ≈ 0.1–0.2.

Held-out photo-z, DDF, 80k candidates resampled to COSMOS p(z|m) (ESS ≈ 26k; single runs vary
by ~±0.02 in NMAD at this size):

| simulation settings | NMAD | outliers |
|---|---|---|
| mock library, no calibration offsets, z < 3 (earlier) | 0.020 | 9.9% |
| real library, no offsets, no dropout, z < 4 | 0.044 | 20% |
| + 1% calibration offsets | 0.048 | 21% |
| + one-band dropout for 10% of objects | 0.068 | 24% |
| + independent 10% per-band dropout (abandoned) | 0.116 | 29% |

Defaults now: zp_sigma 0.01, one-band dropout for 5%.

DP2's own photo-z against COSMOS2020 (103k extended objects, NMAD / outliers, all r < 25.5):
BPZ 0.105/35%, DNF 0.091/33%, FlexZBoost 0.048/16%, GPz 0.124/35%, TPZ 0.065/22%, kNN
0.005/13%. kNN (and likely the other trained estimators) must have COSMOS photo-z and CDF-S
spec-z in its training set, so only BPZ (templates) is a fair comparison on these fields.

First real-data run (a 25k-unique-object test model) on CDF-S X-ray sources (744 matched,
583 spec-z): NMAD 0.140 / 38% outliers vs BPZ 0.102 / 38%; better than BPZ at r = 24–25.5,
worse and overconfident at r < 21 (68% coverage 0.39). The bright outliers are luminous
variable quasars (L_X 10^44–45, z = 0.8–2.8, placed at z ≈ 0.15: the galaxy p(z|m) applied to
total magnitude, fixed above) and stars (z = 0; no star class); bright extended galaxies alone
give NMAD 0.08 / 8% outliers. log L5100 vs L_X: Spearman 0.14.

Production model `npe_v1` (Oct 24, 2026; saved to `models/npe_v1`, gitignored): 10^6
simulations per layer (27 min each on the laptop CPU), host-magnitude p(z|m) resampling (ESS
657k), 1.3M training rows, 123 epochs in 1.9 h. Held-out simulations (DDF + WFD): photo-z
NMAD 0.046 / 24% outliers, calibrated (68/95% coverage 0.68/0.95); variable objects: NMAD
0.021 / 11%, log L5100 scatter 0.23 dex; log M* scatter 0.38 dex.

Real data, photo-z NMAD / outliers by r (cModel), ours vs DP2 BPZ (the fair, template-based
comparison; the trained DP2 estimators evidently saw these fields' reference redshifts):

| | r < 21 | 21–22 | 22–23 | 23–24 | 24–25.5 | all |
|---|---|---|---|---|---|---|
| COSMOS vs COSMOS2020 (103k extended): ours | 0.034/1% | 0.036/2% | 0.037/5% | 0.056/17% | 0.134/37% | 0.093/30% |
| BPZ | 0.026/1% | 0.027/2% | 0.030/7% | 0.060/26% | 0.176/44% | 0.111/36% |
| FlexZBoost | 0.013/1% | 0.015/1% | 0.019/2% | 0.030/7% | 0.072/22% | 0.050/18% |
| CDF-S X-ray sources (744): ours | 0.042/12% | 0.044/6% | 0.046/9% | 0.085/36% | 0.210/56% | 0.094/32% |
| BPZ | 0.035/15% | 0.025/9% | 0.034/13% | 0.098/42% | 0.255/59% | 0.102/38% |

CDF-S spec-z subset (583): ours 0.073 / 25% outliers, BPZ 0.063 / 30%, FlexZBoost 0.039 / 23%.
So: better than BPZ at r > 23 and in outlier rate, slightly worse NMAD at r < 23, behind the
trained estimators. Real-data 68% coverage is 0.43–0.63 (overconfident, worst when bright),
so a sim-to-real gap remains in the bright-galaxy colours. Bright z > 0.7 quasars now get
broad, often bimodal redshift posteriors that cover the truth for some, with medians still low;
their L5100 posteriors (10^43–45) track L_X. Variable X-ray AGN (22): median log(L5100/L_X)
= +0.14, scatter 0.49 dex; non-variable X-ray AGN get low, broad L5100 (median −2.9 dex
relative to L_X, 68% width 5.9 dex): without variability the photometry hardly constrains
L_AGN, and the 50%-AGN, flat-in-log-L prior dominates.

Bright-end diagnosis: matched at fixed redshift, real bright COSMOS galaxies' nearest
simulated colours differ by |median| ≤ 0.012 mag in every colour, and real colours lie within
the simulated distribution (median match distance 0.075 mag), so there are no zero-point
offsets. The median colour offsets (real redder by 0.1–0.3 in u−g and g−r) come from the
simulation prior's galaxy mix. But real posteriors were ~2× too narrow at r < 23 (half-width
0.015–0.020 vs NMAD 0.034–0.037), so ~3% of model error was missing.

`npe_v2` (`models/npe_v2`): the same 2×10^6 simulations with coadds redrawn at
zp_sigma = 0.03 (`scripts/redraw_coadds.py`, seconds, no new SEDs), trained with batch 2048,
lr 2e-3, patience 5 and 1× ESS resampling: 13 min instead of 2 h (stopped at 48 epochs on a
noisy validation loss, so possibly under-trained). Stars are now removed (Gaia, `gaia_flag.py`;
reference stars), and COSMOS includes point sources.

| | COSMOS all (114k) | COSMOS r < 22 | COSMOS 68% coverage | CDF-S spec-z (579) |
|---|---|---|---|---|
| npe_v1 (zp 1%) | 0.100 / 31% | 0.035 / 2% | 0.42–0.63 | 0.074 / 24% |
| npe_v2 (zp 3%) | 0.114 / 32% | 0.038 / 2% | **0.65–0.70** | 0.080 / 26% |
| DP2 BPZ | 0.121 / 38% | 0.027 / 1% | | 0.062 / 29% |

v2's posteriors are calibrated on real data at the cost of a little point-estimate accuracy.
COSMOS extended galaxies: 0.108 / 31%; point sources not flagged by Gaia (7.5k; faint stars
and quasars): 0.205 / 46%.

Population AGN prior (`amortized/agn_prior.py`, reported by `validate_real.py`): an
Eddington-ratio distribution p(log λ) ∝ λ^−0.65 over 10^−4–10^0.5 (k_bol = 10,
M_BH = 0.002 M*), a fraction 0.1 (1+z)^3.5 (≤ 0.9) of galaxies, which gives P(λ > 0.01) ≈ 0.5% at
z = 0 and ≈ 5% at z = 1, applied by importance reweighting of the posterior samples. CDF-S,
mean P(λ_Edd > 0.01), training → population prior: galaxies that are not X-ray AGN
0.18 → 0.04, X-ray AGN 0.35 → 0.10, variable X-ray AGN 0.99 → 0.91.

Clean variable sample with cModel (`attach_object_columns.py` on the RSP: 731,886 of
731,886 matched by objectId; coadd coverage u 68%, g 87%, r 82%, i 85%, z 85%, y 83% —
patchier than the 13-region library, where i is missing for only ~2%).

Quasar check (`validate_real.py --reference milliquas`, npe_v2): 7,786 variable objects with
Milliquas spectroscopic redshifts: NMAD 0.29, 65% outliers, bias −0.28; by redshift, outliers
27% at z < 0.5, 64% at z = 1–1.5, ≥ 94% at z > 1.5, with predicted medians stuck at
z ≈ 0.7. Cause: simulated AGN-dominated objects look nothing like real quasars. Median
colours, real point-like Milliquas quasars vs simulated AGN-dominated (r < 22.5):
u−g 0.1–0.7 vs 0.7–1.8, g−r 0.1–0.3 vs 0.4–1.3. The AGN dust prior (half-normal σ = 0.5,
median E(B−V) = 0.32) reddens most simulated AGN, but even simulated AGN with E(B−V) < 0.05 are
too red in u−g (by 0.2–0.9) and have far too much colour structure with redshift
(g−r = +0.93 at z = 1–1.5, r−i = +0.81 at z = 1.5–2, i−z = +0.70 at z = 2–2.5; real quasars vary by
0.1–0.3), which points to over-strong emission lines / FeII and a too-red continuum in the
AGN SED prior. Fix before running the full variable sample: AGN SED priors tuned to match
the SDSS quasar colour–redshift relation, then re-simulate.

AGN priors retuned to real quasar colours (in jaxpz): half of AGN unobscured (both dust
screens thin, since jaxsedfit attenuates the AGN by E(B−V)_gal + E(B−V)_agn), broad lines
log-normal around 0.5× the template (jaxsedfit's default allows 20×), pl_slope ~ N(−2.1, 0.3).
Unobscured simulated quasars then match the median colours of point-like Milliquas quasars to
0.10 mag rms (was 0.60). jaxsedfit already applies IGM absorption (CIGALE v2025.1 port) to all
components.

`npe_v3` (10^6 new simulations per layer, zp 3%, patience 10, lr 1e-3; 163 epochs, 38 min):
COSMOS 0.102 / 30% (68% coverage 0.64–0.71), CDF-S 0.093 / 31% (BPZ 0.100 / 37%). Milliquas
variable quasars: NMAD 0.37, 64% outliers, but the bias is gone (−0.03, was −0.28) and 68%
coverage is 0.63–0.83. The posteriors are broad and multi-peaked (median 26% of the mass
within 0.15(1+z) of the truth; the mode is no better than the median), and the errors are
systematic in z: low-z Seyferts pushed up (z < 0.5: predicted 0.51), z = 1.5–2.5 quasars
pulled down to ~1. Likely cause: an AGN's redshift prior comes from the galaxy p(z | host m),
and a z ~ 2 quasar's faint host (r ~ 25) sits where galaxies are at z ~ 1; real quasars follow
the quasar luminosity function, peaking at z ~ 1.5–2.5.

`npe_v3b` (same simulations, `jaxpz train --specz-input --agn-fraction`; stopped at 54
epochs): with spec-z given, z is recovered (NMAD 0.001) and on CDF-S the L5100 posterior tracks
L_X better (Spearman 0.41 vs 0.28 in photo-z mode; variable X-ray AGN median
log(L5100/L_X) = +0.62). Its photo-z mode is worse than v3 (COSMOS 0.174 / 42%), so the extra
input costs accuracy at this training length; the AGN fraction is poorly constrained (logit
scatter 2.6, 68% coverage 0.47 with z known). Population prior, CDF-S, mean
P(λ_Edd > 0.01): not X-ray AGN 0.03, X-ray AGN 0.10, variable X-ray AGN 0.97–0.98.

Physical population prior (jaxpz `PopulationPriors`): mass function × volume, sSFR and dust
vs mass and redshift, Eddington-ratio AGN with luminosity evolution and no 10^46 cap. Reweighting
the v3 simulations to it showed (i) v3's p(z | host m) prior made 33% of r < 21.5 objects
AGN-dominated, half of them at z < 0.5 (realistic: ~0.7%), explaining its low-z Seyfert and
high-z quasar errors; (ii) jaxsedfit's mass-independent default SFH makes galaxies 0.5–3 mag too
bright at fixed (z, M*). `npe_v4` (2×10^6 drawn directly from the hand-set population) is worse
than v3: COSMOS 0.121 / 34% (v3 0.102 / 30%), bright-end bias +0.04, 68% coverage 0.44–0.51
(v3 0.64–0.71); CDF-S 0.110 / 35%; Milliquas 0.36 / 72% (v3 0.37 / 64%); L5100–L_X Spearman 0.35
(v3 0.23). The hand-set population still puts bright galaxies at too high z, so the prior biases
the posteriors: the motivation for inferring the population hyperparameters from the data
(jaxpz `docs/design/hierarchical_population.rst`; Phase 1 code in jaxpz.hierarchical).

Next: hierarchical fit of the population on COSMOS (broad-prior training set and model, then
`jaxpz fit-population`); retrain under the fitted population; then `jaxpz predict` on the full
variable sample.

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
pip install --user git+https://github.com/burke86/jaxpz   # light-curve statistics (numpy/pandas only)
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

**7. Full run in HEALPix tiles**, only after steps 4–6 look right. Current preferred
settings (Oct 2026, 125 GB server, 50 GB home quota):

```bash
cd ~/rubin_imbh                       # git pull only when no run is in progress
nohup python scripts/select_variables.py run --tile-order 3 --stats-only --min-abs-b 25 \
    --err-scale 1.0 --sys-frac r=0.0075,i=0.0075,z=0.0075 --prefilter "refExtendedness >= 0" \
    --bright-stars ~/wfd_pilot/bright_stars_r18.parquet --chunk-objects 5000 \
    --n-workers 3 --memory-limit 24GB --out ~/dp2_variables_v2 >> ~/dp2_variables_v2.log 2>&1 &
```

| setting | why |
|---|---|
| `--stats-only` | selected objects' statistics only (<1 GB); all-Object stats or light curves would exceed the 50 GB quota. Fetch light curves for candidates later by `objectId`. |
| `--min-abs-b 25` | skip tiles centred within 25° of the Galactic plane: slowest (crowded) and dominated by extinction and variable stars, of little use for nuclear AGN/TDEs. 15° was not enough: tile 688 (RA 225, Dec −36, b = +20°, towards the inner Galaxy) has 284 partitions and twice took the whole RSP server down mid-tile (no error in the log, "server not running"). |
| `--err-scale 1.0` | object-source errors are already calibrated (ECDFS, WFD) |
| `--sys-frac r,i,z = 0.75%` | chosen from the CDF-S X-ray floor scan (0.5–1% optimum); `fit_floor.py`'s suggestion over-picks |
| `--prefilter "refExtendedness >= 0"` | drops Objects with degenerate photometry (no extendedness) |
| `--bright-stars …parquet` | reuse the cached r < 18 point-source list (10.9M stars; 8 min to rebuild) |
| `--chunk-objects 5000`, 3 × 24 GB | memory headroom for dense partitions (tile 290: 112 partitions, 10.8M Objects) |
| defaults | object source, nightly means, per-band (r, i, z; ≥10 nights) **or** combined-band (dof ≥ 4) at 5σ, clipped |

Runtime: ~1–4 min per tile (sparse WFD < 1 min; DDF and ecliptic high-cadence tiles 5–15
min); ~3–5 h for the high-latitude sky. Run one job at a time: a second job competes for
memory and cores.

Keep the server alive: the RSP shuts the server down (killing the job) when the browser
disconnects, e.g. when the laptop sleeps, and a tile that exhausts memory can take the server
down too (see `--min-abs-b`). After any "server not running", restart the server (largest
size), check `_DONE` count and the last `took` line, and rerun the same command. Keep the RSP tab open and run `caffeinate -dims`
in a terminal on the Mac (plugged in, lid open) until the run finishes.

Monitor:

```bash
pgrep -af select_variables.py || echo "not running"
grep -E "INFO skipping|INFO .*tiles," ~/dp2_variables_v2.log | tail -2   # skipped tiles, number to run
grep took ~/dp2_variables_v2.log | tail -1                                # last tile, time remaining
ls ~/dp2_variables_v2/o3_p*/_DONE 2>/dev/null | wc -l                     # tiles finished (all runs)
grep -E "Traceback|Error" ~/dp2_variables_v2.log | tail -3
```

Stop with `pkill -f select_variables.py`. If it stops early, rerun the same command: tiles
with a `_DONE` marker are skipped and a half-written tile is redone (`>>` appends to the log).
When finished, combine and summarize the tiles (`python scripts/summarize_selection.py
~/dp2_variables_v2 --out ~/dp2_variables_v2_all`), or just combine them:

```bash
python - <<'PY'
import glob, pandas as pd
parts = [pd.read_parquet(f) for f in sorted(glob.glob("/home/colinjburke/dp2_variables_v2/o3_p*/stats.parquet"))]
d = pd.concat(parts, ignore_index=True)
d.to_parquet("/home/colinjburke/dp2_variables_v2_all.parquet")
print(len(d), "selected;", d.objectId.nunique(), "unique;", int((d.nVarBands >= 1).sum()), "per-band,",
      int(d.varCombo.sum()), "combined")
PY
```

Low-latitude tiles finished before the latitude cut was raised (448–452 at |b| < 15°, and
others between 15° and 25°) remain in the output; drop them with a |b| cut or keep them as a
stellar-variability control. CPU was ~77% idle with 3 workers, so on sparse sky more workers
(e.g. 6 × 14 GB) should be faster; keep 3 × 24 GB when dense tiles remain. To profile
a tile, add `--profile tile.html` (dask performance report). For the dia source add
`--source dia` (and `--min-dia-sources`, `--object-catalog`).

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
