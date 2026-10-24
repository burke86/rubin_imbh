"""Download the COSMOS2020 columns used for the empirical p(z | m) prior (IRSA TAP).

    python scripts/fetch_cosmos2020_nzm.py --out data/cosmos2020_classic_zr.parquet

COSMOS2020 CLASSIC (Weaver et al. 2022), clean area (flag_combined = 0), galaxies
(lp_type = 0, 2): HSC r and i auto fluxes (uJy), LePhare photo-z with 68% bounds.
"""

import argparse
from pathlib import Path

import numpy as np

QUERY = """SELECT id, ra, dec, hsc_r_flux_auto, hsc_r_fluxerr_auto, hsc_i_flux_auto, lp_zbest, lp_zpdf_l68, lp_zpdf_u68, lp_type
FROM cosmos2020c WHERE flag_combined = 0 AND lp_type IN (0, 2) AND lp_zbest > 0"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/cosmos2020_classic_zr.parquet")
    args = p.parse_args(argv)
    import pyvo

    tap = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
    job = tap.submit_job(QUERY, maxrec=3_000_000)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=1800)
    if job.phase != "COMPLETED":
        raise SystemExit(f"TAP job ended in phase {job.phase}")
    df = job.fetch_result().to_table().to_pandas()
    job.delete()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    r = 23.9 - 2.5 * np.log10(df.hsc_r_flux_auto)
    print(f"wrote {len(df)} galaxies ({int((r < 25.5).sum())} with r < 25.5) to {out}")


if __name__ == "__main__":
    main()
