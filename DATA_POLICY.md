# Rubin data policy: what this repo may and may not hold

Working notes on how the Rubin data rights rules apply to this project. This is our reading
of the policy, not legal advice; the governing text is
[RDO-013, Rubin Data Policy v1.2.5](https://docushare.lsst.org/docushare/dsweb/Get/RDO-013)
(revised 2025-11-05), plus the user agreement each data-rights holder signs (DPOL-307) and the
Acceptable Use Policy (LPM-123), which have not been checked here and may be stricter. If in doubt,
ask the Data Policy Committee ([data policy page](https://rubinobservatory.org/for-scientists/data-products/data-policy)).

[RDO-121](https://docushare.lsst.org/docushare/dsweb/Get/RDO-121) (Pre-Release Data Policy)
does **not** apply: it covers Data Facilities and Data Access Centers that receive data before
release, not data-rights holders using released DP2 data.

## Rules that matter here

| Clause | What it says |
|---|---|
| DPOL-302, -401 | Only data-rights holders ("LSST Users") may access proprietary data; US users access it through the Rubin DACs (RSP). |
| DPOL-307 | Users sign an agreement restricting **redistribution** of proprietary data; each user is personally accountable. |
| DPOL-901 | Sanctions (up to loss of data rights) for violating the policy or **facilitating non-users' access** to proprietary data. |
| DPOL-512 | Bulk downloads of significant fractions of the data go through a future, possibly restricted, process. |
| DPOL-601 | A derived data product (DDP) is one that **cannot be used to recreate** proprietary data products, including "equivalent" catalogs of positions and magnitudes. Catalog entries are not DDPs, except lists of up to 1000 sources (DPOL-603). DDPs may be shared with anyone. |
| DPOL-701, -702 | No restriction on publishing science results and DDPs; non-users may co-author but get no access to the data. |
| DPOL-703 | Reproducibility is achieved by publishing the **queries**, not copies of the proprietary data. |
| DPOL-520 | The fact that observations exist (observing epochs, dataset metadata) is not proprietary. |
| §8.6 | DP2 photo-z are proprietary; producing an equivalent set of photo-z from proprietary data is constrained the same way. |

There is no clause forbidding a data-rights holder from keeping a copy on their own computer.
The restriction is on who can see it.

## How this applies to the project's files

| File | Status | Laptop | Git repo / non-users |
|---|---|---|---|
| Code, simulations, mock catalogs | Not Rubin data | yes | yes |
| `data/cosmos2020_*.parquet` (public COSMOS2020) | Not Rubin data | yes | yes (gitignored only for size) |
| Survey library (`export_survey_library.py`: objectId, coordinates, fluxes, nightly errors) | Proprietary | yes, private storage | **no** |
| Variability stats, selections, `object_photoz` exports, `validate_real.py` outputs | Proprietary | yes, private storage | **no** |
| Our photo-z / M* / L_AGN catalog for DP2 objects | Proprietary (§8.6) | yes, private storage | **no**; publish only as plots, statistics, or lists of ≤ 1000 targets |
| Trained flow weights | Probably a DDP (cannot recreate catalog entries) | yes | ask the DPC before publishing |
| Figures and summary statistics | DDP | yes | yes |

## Practice

1. **Never commit DP2-derived tables.** `.gitignore` excludes `*.parquet` and `dp2_variables/`;
   keep other formats (CSV, FITS, HDF5) of DP2 data out of the repo too. Don't put them in
   shared cloud folders that non-users can open.
2. **Copy only what is needed.** Prefer small derived tables over raw catalog pulls; large-scale
   inference on all of DP2 runs on the RSP (`pip install --user jax==0.6.2 flowjax`).
3. **Strip identifiers before download** where they are not needed. Training uses only fluxes,
   errors and cadences, so the survey library needs neither objectId nor coordinates.
4. **Papers** acknowledge the policy as in DPOL-306 and cite the queries (DPOL-703), not data copies.
5. **Collaborators without data rights** may write code and see plots (RDO-013 §8.1, Scenario 2) but must not
   receive DP2 tables or RSP access.
