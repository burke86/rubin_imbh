"""Generate an amortized-inference training set: priors -> jaxsedfit -> DP2-like observables.

    python scripts/make_training_set.py --library ~/survey_library.parquet --n 100000 \
        --layer wfd --dsps-ssp /path/to/tempdata.h5 --out ~/train_wfd.parquet

Needs jaxsedfit with the ``simulate`` API. ``--mock-library`` uses a synthetic library (for
testing without DP2 data). Output: one row per simulated object with the parameters (theta:
redshift, log_stellar_mass, log_L5100_erg, has_agn, dust, SFH, f_var, tau, ...) and the
observables (x: coadd_<b>, coadd_err_<b>, <b>_xsAbs, <b>_varSigClip, <b>_nEpoch, <b>_rmsErr,
all_varSigClip, all_dof).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.amortized.generator import ObserveConfig, Priors, lsst_simulation_config, observe, simulate_population  # noqa: E402
from rubin_imbh.amortized.library import load_library, mock_library  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", help="survey library parquet (export_survey_library.py)")
    p.add_argument("--mock-library", action="store_true", help="use a synthetic library instead")
    p.add_argument("--layer", default="wfd", help="library layer to borrow noise/cadence from")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--dsps-ssp", required=True, help="DSPS SSP template file")
    p.add_argument("--chunk", type=int, default=20000, help="objects simulated per chunk (bounds memory)")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--band-dropout", type=float, default=0.05,
                   help="fraction of objects that lose one random band's coadd flux, so the model handles missing bands")
    p.add_argument("--zp-sigma", type=float, default=ObserveConfig.zp_sigma,
                   help="per-object, per-band calibration offset rms applied to the coadd fluxes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rng = np.random.default_rng(args.seed)
    if args.mock_library:
        lib = mock_library(20000, rng, args.layer)
    elif args.library:
        lib = load_library(Path(args.library).expanduser())
    else:
        raise SystemExit("give --library or --mock-library")
    priors = Priors()
    cfg = lsst_simulation_config(args.dsps_ssp, z_range=priors.z_range)
    parts, t0 = [], time.time()
    for k, start in enumerate(range(0, args.n, args.chunk)):
        m = min(args.chunk, args.n - start)
        sim = simulate_population(m, priors, rng, cfg, seed=args.seed * 1000 + k, batch_size=args.batch_size)
        parts.append(observe(sim, lib, priors, rng, ObserveConfig(layer=args.layer, band_dropout=args.band_dropout, zp_sigma=args.zp_sigma)))
        logging.info("%d / %d objects (%.0f s)", start + m, args.n, time.time() - t0)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(Path(args.out).expanduser())
    agn = out.has_agn.astype(bool)
    print(f"wrote {len(out)} objects to {args.out} in {time.time() - t0:.0f} s; AGN fraction {agn.mean():.2f}; "
          f"variable (all_varSigClip >= 5): AGN {np.mean(out.all_varSigClip[agn] >= 5):.3f}, "
          f"no AGN {np.mean(out.all_varSigClip[~agn] >= 5):.4f}")


if __name__ == "__main__":
    main()
