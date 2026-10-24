"""Redraw the coadd fluxes of existing training sets with a different calibration error.

    python scripts/redraw_coadds.py ~/train_ddf.parquet ~/train_wfd.parquet --zp-sigma 0.03 --suffix _zp3

The SED simulation (jaxsedfit) and the variability statistics are unchanged; only
``coadd_<b>`` is redrawn from ``true_<b>`` and ``coadd_err_<b>`` (missing bands stay missing).
Writes ``<name><suffix>.parquet`` next to each input.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.amortized.generator import redraw_coadds  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("train", nargs="+")
    p.add_argument("--zp-sigma", type=float, required=True)
    p.add_argument("--suffix", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    for f in args.train:
        path = Path(f).expanduser()
        df = redraw_coadds(pd.read_parquet(path), args.zp_sigma, rng)
        out = path.with_name(path.stem + args.suffix + ".parquet")
        df.to_parquet(out)
        print(f"wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
