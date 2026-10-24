"""Combine a tiled selection run and summarize it.

    python scripts/summarize_selection.py ~/dp2_variables_v2 --out ~/dp2_variables_v2_all

Writes ``<out>.parquet`` (all selected objects, with tile, Galactic and ecliptic latitude
columns) and ``<out>.png`` (sky density and per-tile breakdowns), and prints counts by tile,
latitude, morphology, magnitude and selection route.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="tiled output directory (contains o<order>_p<pixel>/stats.parquet)")
    p.add_argument("--out", required=True, help="output prefix (.parquet and .png are added)")
    p.add_argument("--include-vetoed", action="store_true",
                   help="also summarize objects failing envClean (bright stars and their surroundings)")
    args = p.parse_args(argv)
    pd.set_option("display.width", 200)

    run = Path(args.run_dir).expanduser()
    files = sorted(glob.glob(str(run / "o*_p*" / "stats.parquet")))
    done = sorted(glob.glob(str(run / "o*_p*" / "_DONE")))
    parts = []
    for f in files:
        d = pd.read_parquet(f)
        d["tile"] = Path(f).parent.name
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    print(f"{len(done)} finished tiles, {len(files)} with selected objects; {len(d)} rows, "
          f"{d.objectId.nunique()} unique objectId")
    if "envClean" in d and not args.include_vetoed:
        n0 = len(d)
        d = d[d["envClean"].astype(bool)].reset_index(drop=True)
        print(f"envClean only (no r<18 star within the veto radius, not a bright star itself): {len(d)} of {n0}")

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(d.coord_ra.to_numpy() * u.deg, d.coord_dec.to_numpy() * u.deg)
    d["gal_b"] = c.galactic.b.deg
    d["ecl_lat"] = c.barycentrictrueecliptic.lat.deg
    with np.errstate(divide="ignore", invalid="ignore"):
        d["rmag"] = 31.4 - 2.5 * np.log10(d["r_psfFlux"])

    per_band = d["nVarBands"] >= 1
    combo = d["varCombo"].astype(bool)
    print(f"\nselection route: per-band only {int((per_band & ~combo).sum())}, combined only "
          f"{int((combo & ~per_band).sum())}, both {int((per_band & combo).sum())}")
    print(f"extended (refExtendedness >= 0.5): {np.mean(d.refExtendedness >= 0.5):.2f}")

    print("\nby |Galactic latitude|:")
    print(d.groupby(pd.cut(d.gal_b.abs(), [0, 15, 25, 40, 60, 90]), observed=False).size().to_string())
    print("\nby |ecliptic latitude|:")
    print(d.groupby(pd.cut(d.ecl_lat.abs(), [0, 5, 10, 20, 40, 90]), observed=False).size().to_string())
    print("\nby r magnitude (coadd PSF) and morphology:")
    ext = np.where(d.refExtendedness >= 0.5, "extended", "point")
    print(pd.crosstab(pd.cut(d.rmag, [-np.inf, 18, 20, 21, 22, 23, 24, np.inf]), ext).to_string())

    t = d.groupby("tile").agg(n=("objectId", "size"), ra=("coord_ra", "median"), dec=("coord_dec", "median"),
                              gal_b=("gal_b", "median"), ecl_lat=("ecl_lat", "median"),
                              frac_combined_only=("varCombo", lambda x: float(np.mean(x & (d.loc[x.index, "nVarBands"] == 0)))),
                              frac_ext=("refExtendedness", lambda x: float(np.mean(x >= 0.5))))
    t = t.sort_values("n", ascending=False)
    print(f"\ntiles: median {int(t.n.median())} selected per tile; top 15:")
    print(t.head(15).round(2).to_string())

    out = Path(args.out).expanduser()
    d.to_parquet(out.with_suffix(".parquet"))
    print(f"\nwrote {out.with_suffix('.parquet')}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    hb = ax[0].hexbin(d.coord_ra, d.coord_dec, gridsize=120, bins="log", mincnt=1, cmap="viridis")
    ax[0].invert_xaxis()
    ax[0].set_xlabel("RA (deg)")
    ax[0].set_ylabel("Dec (deg)")
    ax[0].set_title("selected objects")
    fig.colorbar(hb, ax=ax[0], label="count per cell")
    ax[1].scatter(t.ecl_lat, t.n, s=12, c=np.abs(t.gal_b), cmap="plasma")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("tile ecliptic latitude (deg)")
    ax[1].set_ylabel("selected per tile")
    ax[2].hist(d.rmag[np.isfinite(d.rmag)], bins=np.arange(14, 27, 0.25), histtype="step", label="all")
    ax[2].hist(d.rmag[np.isfinite(d.rmag) & (d.refExtendedness >= 0.5)], bins=np.arange(14, 27, 0.25),
               histtype="step", label="extended")
    ax[2].set_xlabel("r (coadd PSF mag)")
    ax[2].set_ylabel("selected objects")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=140)
    print(f"wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
