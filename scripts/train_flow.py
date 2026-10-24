"""Train and evaluate the neural posterior estimator on a training set.

    python scripts/train_flow.py ~/train_wfd.parquet [~/train_ddf.parquet ...] --out ~/npe_v1

Holds out --test-frac of the simulations, trains the flow (rubin_imbh.amortized.npe), saves
it to --out, and prints/saves accuracy and calibration on the held-out set (overall and with
and without detected variability).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.amortized.npe import NPE, FlowSpec, evaluate  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("train", nargs="+", help="training-set parquet file(s) from make_training_set.py")
    p.add_argument("--out", required=True)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--layers", type=int, default=FlowSpec.flow_layers)
    p.add_argument("--width", type=int, default=FlowSpec.nn_width)
    p.add_argument("--no-variability", action="store_true", help="photometry-only inputs (ablation)")
    p.add_argument("--nzm", help="empirical p(z|m) catalogue (fetch_cosmos2020_nzm.py output): resample the "
                   "training and held-out sets to it before training")
    p.add_argument("--nzm-oversample", type=float, default=2.0,
                   help="resampled set size in units of the effective sample size (capped at the input size)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    pd.set_option("display.width", 200)

    d = pd.concat([pd.read_parquet(Path(f).expanduser()) for f in args.train], ignore_index=True)
    rng = np.random.default_rng(args.seed)
    test = rng.random(len(d)) < args.test_frac
    train_df, test_df = d[~test].reset_index(drop=True), d[test].reset_index(drop=True)
    if args.nzm:
        from rubin_imbh.amortized.nz_prior import EmpiricalNzm, resample_to_nzm

        nzm = EmpiricalNzm.from_cosmos2020(args.nzm)
        # drawing many more objects than the effective sample size only repeats the same
        # simulations, which the flow then overfits
        _, ess = resample_to_nzm(train_df, nzm, 1, rng)
        _, ess_t = resample_to_nzm(test_df, nzm, 1, rng)
        train_df, _ = resample_to_nzm(train_df, nzm, int(min(len(train_df), args.nzm_oversample * ess)), rng)
        test_df, _ = resample_to_nzm(test_df, nzm, int(min(len(test_df), args.nzm_oversample * ess_t)), rng)
        print(f"resampled to empirical p(z|m): effective sample size {ess:.0f} (training), {ess_t:.0f} (held-out)")
    print(f"{len(train_df)} training, {len(test_df)} held-out simulations")

    t0 = time.time()
    spec = FlowSpec(flow_layers=args.layers, nn_width=args.width, use_variability=not args.no_variability)
    npe = NPE.train(train_df, spec, seed=args.seed,
                    learning_rate=args.learning_rate, max_epochs=args.max_epochs, max_patience=args.patience,
                    batch_size=args.batch_size)
    print(f"trained in {time.time() - t0:.0f} s; final val loss {npe.losses['val'][-1]:.3f} "
          f"after {len(npe.losses['val'])} epochs")
    npe.save(args.out)

    out = Path(args.out).expanduser()
    report = {"all": evaluate(npe, test_df)}
    if "all_varSigClip" in test_df:
        var = (test_df["all_varSigClip"] >= 5).to_numpy()
        if var.sum() >= 50:
            report["variable"] = evaluate(npe, test_df[var].reset_index(drop=True))
            report["not variable"] = evaluate(npe, test_df[~var].reset_index(drop=True))
    for k, r in report.items():
        print(f"\n=== held-out: {k}")
        print(r.round(3).to_string())
        r.to_csv(out / f"eval_{k.replace(' ', '_')}.csv")
    print(f"\nsaved flow and evaluation to {out}")


if __name__ == "__main__":
    main()
