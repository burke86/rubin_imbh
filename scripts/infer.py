"""Run the trained posterior estimator on every object of a statistics table.

    python scripts/infer.py data/dp2_clean_variables_cmodel.parquet --npe models/npe_v2 \
        --out data/dp2_clean_variables_posteriors.parquet

The input is a ``select_variables.py`` statistics table with coadd flux columns attached
(``--keep-columns`` or ``scripts/attach_object_columns.py``). Gaia stars (``gaia_star``) are
dropped; output: one row per object with the observables, ``pointlike``, and
``post_*`` / ``pop_*`` posterior summaries and ``p_agn_train`` / ``p_agn_pop``
(see ``rubin_imbh.amortized.npe.summarize_posteriors``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.amortized.real import real_observables  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stats")
    p.add_argument("--npe", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--flux-template", default="{b}_cModelFlux")
    p.add_argument("--err-template", default="{b}_cModelFluxErr")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--keep-stars", action="store_true")
    args = p.parse_args(argv)

    from rubin_imbh.amortized.npe import NPE, summarize_posteriors

    stats = pd.read_parquet(Path(args.stats).expanduser())
    if not args.keep_stars and "gaia_star" in stats:
        stats = stats[~stats["gaia_star"].fillna(False).astype(bool)]
    obs = real_observables(stats, args.flux_template, args.err_template)
    t0 = time.time()
    out = summarize_posteriors(NPE.load(args.npe), obs, args.n_samples)
    out.to_parquet(Path(args.out).expanduser())
    z = out["post_redshift_med"]
    print(f"{len(out)} objects in {time.time() - t0:.0f} s; median photo-z {np.nanmedian(z):.2f}; "
          f"mean P(lambda_Edd > 0.01) training prior {out.get('p_agn_train', pd.Series(np.nan)).mean():.2f}, "
          f"population prior {out.get('p_agn_pop', pd.Series(np.nan)).mean():.2f}; wrote {args.out}")


if __name__ == "__main__":
    main()
