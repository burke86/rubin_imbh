# Design: replace Milliquas with DESI as the AGN (and galaxy) validation reference

*Status: planned (October 2026). Not implemented. Milliquas stays available as
`validate_real.py --reference milliquas` until the DESI reference replaces it.*

## Goal

Validate photo-z, stellar mass and AGN parameters against a large, homogeneous spectroscopic
sample with clean classifications, for both AGN and galaxies, instead of Milliquas.

## Why replace Milliquas

- **Heterogeneous.** Milliquas (Flesch 2023, VizieR VII/294) merges SDSS, 2dF, DESI and many
  smaller programmes. It also includes "A"-type AGN and sources classified through radio or
  X-ray associations, so redshift quality and classification vary across the sample.
- **AGN only.** It has no galaxies, so the Milliquas column of the photo-z figure has an
  empty galaxy panel.
- **DESI DR1 is homogeneous.** About 18M spectra reduced by one pipeline, with spectral types
  (`GALAXY` / `QSO` / `STAR`) and a redshift-warning bitmask (`ZWARN`). That gives thousands of
  clean QSOs plus very large BGS, LRG and ELG galaxy samples.

## Reference definition

- **Source:** the DESI DR1 healpix-coadded redshift catalogue (`zpix`, main survey), queried
  through Astro Data Lab's TAP service or downloaded from the DESI data release.
- **Selection:**
  - `ZWARN == 0`;
  - the primary redshift per target (`ZCAT_PRIMARY`);
  - `SPECTYPE` in {`QSO`, `GALAXY`};
  - for galaxies, `DELTACHI2 > 25`;
  - stars (`SPECTYPE == STAR` or z < 0.003) dropped.
- **Columns** (same layout as the other builders in `rubin_imbh/references.py`):
  - `ra`, `dec`, `z_ref`;
  - `z_type = "spec"`;
  - `is_qso = SPECTYPE == "QSO"`, used as the AGN label in the figures (DESI QSOs are
    broad-line by construction).
- **Complement:** SDSS DR16Q (VizieR VII/289) where DESI does not cover a DP2 field. The
  footprints overlap only partly.

## Footprint

DP2 is mostly southern. By declination, the DP2 variables catalogue is:

| Declination | Fraction |
|---|---|
| below −30° | 15% |
| −30 to −20° | 31% |
| −20 to −10° | 30% |
| −10 to 0° | 17% |
| above 0° | 6% |

DESI covers roughly dec ≳ −20° and SDSS mostly dec ≳ −10°. So overlap exists but is partial
and depends on right ascension. Step 1 below measures it before committing.

## Plan

1. **Count matches.** Cross-match the DP2 variables catalogue
   (`data/dp2_clean_variables_cmodel.parquet`, all of DP2) to DESI DR1 QSOs and to SDSS DR16Q
   within 1″. Report counts, the redshift distribution and the overlap area.
2. **AGN reference (local, no new DP2 data).** Add `desi_reference()` (QSOs; optionally
   DR16Q) to `references.py` as `--reference desi-qso`. Run `validate_real.py` on the DP2
   variables, and swap the Milliquas column in `scripts/plot_photoz.py` for DESI.
3. **Galaxy reference (needs an RSP export).** Galaxies need DP2 photometry of *non-variable*
   objects. Today that exists only for COSMOS and CDF-S. Export DP2 Objects with cModel fluxes
   and variability statistics (the `cosmos_stats` format) in the DP2 tiles that overlap DESI,
   then validate the full DESI reference with galaxies and QSOs in separate panels.
4. **Retire Milliquas from the default figures** once 2–3 work. Keep it as an option for fields
   without DESI or SDSS coverage.

## Data policy

DESI and SDSS are public. Anything derived from DP2 (exports, matched tables, validation
outputs) stays private and out of git, per `DATA_POLICY.md`.
