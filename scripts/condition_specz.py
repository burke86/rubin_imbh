"""Host mass and AGN luminosity conditioned on the reference (spectroscopic) redshift.

Draws posterior samples for the variable objects of validate_real.py outputs and weights them by a
Gaussian kernel in redshift around z_spec (sigma = --kernel (1 + z_spec)), which approximates the
posterior given the data and z = z_spec. Writes the input table (variable objects only) with
``cond_<theta>_{lo,med,hi}`` (16/50/84%) and ``cond_ess`` columns, for plot_mass_redshift.py.
Objects with fewer than --min-ess effective draws near z_spec get NaN.

usage: python scripts/condition_specz.py --npe models/npe_v12 data/sims/val_milliquas_v12.parquet \
           data/sims/val_cosmos-specz_v12.parquet data/sims/val_cdfs-xray_v12.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from jaxpz.agn_prior import weighted_quantiles
from jaxpz.npe import NPE


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tables", nargs="+", help="validate_real.py outputs (with ref_z_ref)")
    p.add_argument("--npe", required=True)
    p.add_argument("--min-sig", type=float, default=5.0, help="variability selection: all_varSigClip >= this")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--kernel", type=float, default=0.03, help="redshift kernel sigma in units of (1 + z_spec)")
    p.add_argument("--min-ess", type=float, default=10.0)
    p.add_argument("--suffix", default="_cond", help="output: <table stem><suffix>.parquet next to the input")
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args(argv)
    npe = NPE.load(args.npe)
    th = list(npe.spec.theta)
    for path in map(Path, args.tables):
        d = pd.read_parquet(path)
        d = d[d["all_varSigClip"] >= args.min_sig].reset_index(drop=True)
        s = npe.sample(d, args.n_samples, seed=args.seed)
        z = s[:, :, th.index("redshift")]
        zs = d["ref_z_ref"].to_numpy(float)[:, None]
        k = np.nan_to_num(np.exp(-0.5 * ((z - zs) / (args.kernel * (1 + zs))) ** 2))
        ksum = k.sum(axis=1, keepdims=True)
        w = np.where(ksum > 0, k / np.where(ksum > 0, ksum, 1.0), 1.0 / z.shape[1])
        ess = np.where(ksum[:, 0] > 0, 1.0 / np.maximum((w**2).sum(axis=1), 1e-30), 0.0)
        for name in ("log_stellar_mass", "log_L5100_erg"):
            q = weighted_quantiles(s[:, :, th.index(name)], w, (0.16, 0.5, 0.84))
            for j, tag in enumerate(("lo", "med", "hi")):
                d[f"cond_{name}_{tag}"] = np.where(ess >= args.min_ess, q[j], np.nan)
        d["cond_ess"] = ess
        out = path.with_name(f"{path.stem}{args.suffix}.parquet")
        d.to_parquet(out)
        print(f"{out}: {len(d)} objects (all_varSigClip >= {args.min_sig:g}), median ESS near z_spec {np.median(ess):.0f}, "
              f"{100 * np.mean(ess < args.min_ess):.1f}% without an estimate")


if __name__ == "__main__":
    main()
