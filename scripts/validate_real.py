"""Validate the neural posterior estimator on real DP2 objects with reference redshifts.

    python scripts/validate_real.py --stats ~/cosmos_stats.parquet --npe ~/npe_v1 \
        --flux-template "{b}_cModelFlux" --err-template "{b}_cModelFluxErr" \
        --reference cosmos2020 --cosmos2020 data/cosmos2020_classic_zr.parquet \
        --photoz ~/cosmos_photoz.parquet --out ~/val_cosmos

``--stats``: a ``select_variables.py run --no-select --stats-only`` output for the field, run
with the simulator's floor (``--sys-frac r=0.0075,i=0.0075,z=0.0075``) and keeping the coadd
flux and error columns (``--keep-columns``). ``--reference``: ``cosmos2020`` (COSMOS2020 photo-z)
``cdfs-xray`` (CDF-S 7 Ms sources, spec-z where available, with L_X) or ``milliquas``
(spectroscopic quasars/AGN anywhere, e.g. for the WFD variable sample). ``--photoz``: optional
table with ``objectId`` and DP2 photo-z columns (export_photoz.py); every
``*_z_median`` estimator is compared on the same objects.

Reports photo-z NMAD / outlier fraction / bias by r magnitude for our posterior medians and
for the DP2 photo-z, coverage of our 68% intervals, and for X-ray AGN the relation between
the posterior lambda L_5100 and L_X.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.amortized.real import real_observables  # noqa: E402
from rubin_imbh.amortized.references import (  # noqa: E402
    cdfs_xray_reference, cosmos2020_reference, match_reference, milliquas_reference)

MAG_BINS = [-np.inf, 21, 22, 23, 24, 25.5]


def photoz_metrics(z: np.ndarray, zref: np.ndarray) -> dict:
    ok = np.isfinite(z) & np.isfinite(zref)
    if ok.sum() < 5:
        return {"n": int(ok.sum()), "nmad": np.nan, "outlier": np.nan, "bias": np.nan}
    dz = (z[ok] - zref[ok]) / (1 + zref[ok])
    return {"n": int(ok.sum()), "nmad": float(1.4826 * np.median(np.abs(dz - np.median(dz)))),
            "outlier": float(np.mean(np.abs(dz) > 0.15)), "bias": float(np.median(dz))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", required=True)
    p.add_argument("--npe", required=True, help="trained model directory (train_flow.py --out)")
    p.add_argument("--flux-template", default="{b}_psfFlux")
    p.add_argument("--err-template", default="{b}_psfFluxErr")
    p.add_argument("--reference", required=True,
                   help="cosmos2020, cdfs-xray, milliquas (spectroscopic quasars/AGN), or a parquet file with ra, dec, z_ref, z_type (and optionally logLX, is_xray_agn)")
    p.add_argument("--cosmos2020", default="data/cosmos2020_classic_zr.parquet")
    p.add_argument("--photoz", help="table with objectId and DP2 photo-z column(s)")
    p.add_argument("--photoz-column", help="photo-z column(s) in --photoz, comma-separated "
                   "(default: every *_z_median column, e.g. bpz, dnf, fzboost, gpz, knn, tpz)")
    p.add_argument("--match-arcsec", type=float, default=1.0)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--extended-only", action="store_true")
    p.add_argument("--keep-stars", action="store_true",
                   help="keep Gaia stars (gaia_star, from scripts/gaia_flag.py) and reference stars; dropped by default")
    p.add_argument("--out", required=True, help="output prefix (.parquet with posteriors, .csv with metrics)")
    args = p.parse_args(argv)
    pd.set_option("display.width", 200)

    from rubin_imbh.amortized.npe import NPE

    stats = pd.read_parquet(Path(args.stats).expanduser())
    if "envClean" in stats:
        stats = stats[stats["envClean"].astype(bool)]
    if not args.keep_stars and "gaia_star" in stats:
        n0 = len(stats)
        stats = stats[~stats["gaia_star"].fillna(False).astype(bool)]
        print(f"dropped {n0 - len(stats)} Gaia stars")
    elif not args.keep_stars:
        print("no gaia_star column (run scripts/gaia_flag.py on --stats): stars are kept")
    if args.extended_only and "refExtendedness" in stats:
        stats = stats[stats["refExtendedness"] >= 0.5]
    obs = real_observables(stats, args.flux_template, args.err_template)
    if args.reference == "cosmos2020":
        ref = cosmos2020_reference(args.cosmos2020)
    elif args.reference == "cdfs-xray":
        ref = cdfs_xray_reference()
    elif args.reference == "milliquas":
        ref = milliquas_reference()
    else:
        ref = pd.read_parquet(Path(args.reference).expanduser())
    obs = match_reference(obs, ref, args.match_arcsec)
    obs = obs[np.isfinite(obs["ref_z_ref"].to_numpy(float))].reset_index(drop=True)
    if not args.keep_stars and "ref_is_star" in obs:
        obs = obs[~obs["ref_is_star"].astype(bool)].reset_index(drop=True)
    print(f"{len(obs)} objects with a {args.reference} reference within {args.match_arcsec}\"")
    if len(obs) == 0:
        raise SystemExit("no matches")

    npe = NPE.load(args.npe)
    s = npe.sample(obs, args.n_samples)
    th = list(npe.spec.theta)
    for j, name in enumerate(th):
        obs[f"post_{name}_med"] = np.median(s[:, :, j], axis=1)
        obs[f"post_{name}_lo"], obs[f"post_{name}_hi"] = np.percentile(s[:, :, j], [16, 84], axis=1)
    if "log_L5100_erg" in th:
        # the same posteriors under a population AGN prior (Eddington-ratio distribution), by
        # importance reweighting: columns pop_<theta>_{med,lo,hi}, p_agn_{train,pop} (L5100 > 1e40)
        from rubin_imbh.amortized.agn_prior import reweight, weighted_quantiles

        w, ess = reweight(s, th)
        for j, name in enumerate(th):
            obs[f"pop_{name}_lo"], obs[f"pop_{name}_med"], obs[f"pop_{name}_hi"] = weighted_quantiles(s[:, :, j], w, (0.16, 0.5, 0.84))
        agn = s[:, :, th.index("log_L5100_erg")] > 40.0
        obs["p_agn_train"], obs["p_agn_pop"], obs["pop_ess"] = agn.mean(axis=1), (w * agn).sum(axis=1), ess
    if args.photoz:
        pz = pd.read_parquet(Path(args.photoz).expanduser())
        pz_cols = (args.photoz_column.split(",") if args.photoz_column
                   else [c for c in pz.columns if c.endswith("_z_median")] or [next(c for c in pz.columns if c != "objectId" and "z" in c.lower())])
        obs = obs.merge(pz[["objectId", *pz_cols]].rename(columns={c: f"dp2_{c}" for c in pz_cols}), on="objectId", how="left")
        print(f"DP2 photo-z columns: {pz_cols}")
    dp2 = [c for c in obs.columns if c.startswith("dp2_")]

    with np.errstate(divide="ignore", invalid="ignore"):
        obs["rmag"] = 31.4 - 2.5 * np.log10(obs["coadd_r"].to_numpy(float))
    zref = obs["ref_z_ref"].to_numpy(float)
    rows = {}
    for lo, hi in zip(MAG_BINS[:-1], MAG_BINS[1:]):
        m = (obs["rmag"] > lo) & (obs["rmag"] <= hi)
        r = {f"ours_{k}": v for k, v in photoz_metrics(obs["post_redshift_med"].to_numpy()[m], zref[m]).items()}
        cov = (zref[m] >= obs["post_redshift_lo"][m]) & (zref[m] <= obs["post_redshift_hi"][m])
        r["ours_cov68"] = float(np.mean(cov)) if m.sum() else np.nan
        for c in dp2:
            met = photoz_metrics(obs[c].to_numpy(float)[m], zref[m])
            r.update({f"{c[4:].split('_z_')[0]}_{k}": met[k] for k in ("nmad", "outlier")})
        rows[f"r {lo}-{hi}"] = r
    allm = {f"ours_{k}": v for k, v in photoz_metrics(obs["post_redshift_med"].to_numpy(), zref).items()}
    for c in dp2:
        met = photoz_metrics(obs[c].to_numpy(float), zref)
        allm.update({f"{c[4:].split('_z_')[0]}_{k}": met[k] for k in ("nmad", "outlier")})
    rows["all"] = allm
    table = pd.DataFrame(rows).T
    print("\n=== photo-z vs reference (outlier: |dz|/(1+z) > 0.15)")
    print(table[[c for c in table.columns if not any(c.endswith(s) for s in ("_nmad", "_outlier")) or c.startswith("ours")]].round(3).to_string())
    for metric in ("nmad", "outlier"):
        print(f"\n{metric} by estimator:")
        print(table[[c for c in table.columns if c.endswith(f"_{metric}")]].rename(columns=lambda c: c.rsplit("_", 1)[0]).round(3).to_string())
    if "pointlike" in obs:
        for name, m in (("extended", ~obs["pointlike"].to_numpy(bool) & (obs["refExtendedness"].to_numpy(float) >= 0.5)),
                        ("point-like, not a Gaia star", obs["pointlike"].to_numpy(bool))):
            print(f"{name}: ", {k: round(v, 3) for k, v in photoz_metrics(obs["post_redshift_med"].to_numpy()[m], zref[m]).items()})
    if "ref_z_type" in obs:
        spec = obs["ref_z_type"].astype(str).eq("spec").to_numpy()
        if spec.sum() >= 5:
            print("spec-z subset (ours):", {k: round(v, 3) for k, v in photoz_metrics(obs["post_redshift_med"].to_numpy()[spec], zref[spec]).items()})
            for c in dp2:
                print(f"spec-z subset ({c[4:]}):", {k: round(v, 3) for k, v in photoz_metrics(obs[c].to_numpy(float)[spec], zref[spec]).items()})

    if "ref_logLX" in obs and "log_L5100_erg" in th:
        agn = obs["ref_is_xray_agn"].astype(bool).to_numpy() & np.isfinite(obs["ref_logLX"].to_numpy(float))
        if agn.sum() >= 10:
            lx = obs["ref_logLX"].to_numpy(float)[agn]
            lo = obs["post_log_L5100_erg_med"].to_numpy()[agn]
            var = (obs["all_varSigClip"].to_numpy(float)[agn] >= 5)
            rho = pd.Series(lo).corr(pd.Series(lx), method="spearman")
            print(f"\n=== X-ray AGN (n={agn.sum()}, variable {var.sum()}): Spearman(log L5100 posterior, log LX) = {rho:.2f}; "
                  f"median log(L5100/LX) = {np.median(lo - lx):.2f} (variable: {np.median((lo - lx)[var]) if var.any() else np.nan:.2f})")
            if "p_agn_pop" in obs:
                lp = obs["pop_log_L5100_erg_med"].to_numpy()[agn]
                print(f"    population AGN prior: Spearman {pd.Series(lp).corr(pd.Series(lx), method='spearman'):.2f}; "
                      f"median log(L5100/LX) = {np.median(lp - lx):.2f} (variable: {np.median((lp - lx)[var]) if var.any() else np.nan:.2f})")
                notx = ~obs["ref_is_xray_agn"].fillna(False).astype(bool).to_numpy()
                for name, m in (("X-ray AGN", agn), ("X-ray AGN, variable", agn & (obs["all_varSigClip"].to_numpy(float) >= 5)),
                                ("not X-ray AGN", notx)):
                    print(f"    mean P(AGN): {name:22s} training prior {obs['p_agn_train'][m].mean():.2f}, population prior {obs['p_agn_pop'][m].mean():.2f}")

    out = Path(args.out).expanduser()
    obs.to_parquet(out.with_suffix(".parquet"))
    table.to_csv(out.with_suffix(".csv"))
    print(f"\nwrote {out.with_suffix('.parquet')} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
