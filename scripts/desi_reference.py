"""DESI DR1 spectroscopic reference (galaxies and AGN) over the DP2 footprint, from the DESI DR1 AGN/galaxy
value-added catalog (desi_dr1.agngal, Juneau et al.) on the NOIRLab Astro Data Lab TAP service.

Keeps primary, ZWARN = 0 GALAXY and QSO spectra in the order-3 HEALPix tiles of a DP2 table (default: the
variable selection, whose tiles trace the Object footprint). Every QSO is kept and a random fraction of the
galaxies (--galaxy-percent, via DESI's random_id), since the galaxies outnumber what validation needs.

AGN (is_agn): Redrock SPECTYPE = QSO, or a FastSpecFit broad H-alpha or H-beta component detected at
S/N >= 5 with sigma >= 500 km/s (FWHM >~ 1200 km/s), which catches low-z broad-line AGN that Redrock calls
GALAXY. Narrow-line (BPT) AGN stay in the galaxy class.

Writes the full reference (ra, dec, z_ref, z_type, is_agn, spectype, logmstar, targetid) and a light copy
(targetid, ra, dec) to upload to the RSP for scripts/match_targets.py. DESI DR1 is public.

usage: python scripts/desi_reference.py --out data/desi_dr1_reference.parquet
"""

from __future__ import annotations

import argparse
import io
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

TAP = "https://datalab.noirlab.edu/tap/sync"
COLUMNS = ("targetid, target_ra, target_dec, z, spectype, logmstar, healpix, "
           "halpha_broad_flux, halpha_broad_flux_ivar, halpha_broad_sigma, "
           "hbeta_broad_flux, hbeta_broad_flux_ivar, hbeta_broad_sigma")
BROAD_SNR, BROAD_SIGMA = 5.0, 500.0  # km/s


def tap_query(adql: str, timeout: float = 600) -> pd.DataFrame:
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}).encode()
    with urllib.request.urlopen(TAP, data=data, timeout=timeout) as r:
        text = r.read().decode()
    if text.lstrip().startswith("<"):
        raise RuntimeError(text[:500])
    return pd.read_csv(io.StringIO(text))


def broad_line(d: pd.DataFrame, line: str) -> np.ndarray:
    snr = d[f"{line}_broad_flux"] * np.sqrt(d[f"{line}_broad_flux_ivar"].clip(lower=0))
    return ((snr >= BROAD_SNR) & (d[f"{line}_broad_sigma"] >= BROAD_SIGMA)).to_numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--footprint", default="data/dp2_clean_variables_cmodel.parquet",
                   help="DP2 table with a tile column (o3_p<pixel>) defining the order-3 tiles to query")
    p.add_argument("--galaxy-percent", type=float, default=10.0, help="random percentage of DESI galaxies to keep")
    p.add_argument("--out", default="data/desi_dr1_reference.parquet")
    args = p.parse_args(argv)

    tiles = sorted(int(t.split("_p")[1]) for t in pd.read_parquet(args.footprint, columns=["tile"])["tile"].unique())
    base = "FROM desi_dr1.agngal WHERE zcat_primary = 't' AND zwarn = 0"
    parts = []
    for i, t in enumerate(tiles, 1):
        # DESI healpix is nside 64 (order 6), nested: order-3 pixel = healpix // 64
        where = f"{base} AND healpix BETWEEN {64 * t} AND {64 * t + 63}"
        q = (f"SELECT {COLUMNS} {where} AND (spectype = 'QSO' OR "
             f"(spectype = 'GALAXY' AND random_id < {args.galaxy_percent}))")
        d = tap_query(q)
        parts.append(d)
        print(f"[{i}/{len(tiles)}] tile {t}: {len(d)} spectra", flush=True)
    d = pd.concat([p_ for p_ in parts if len(p_)], ignore_index=True)
    d["spectype"] = d["spectype"].str.strip()
    is_agn = (d["spectype"] == "QSO").to_numpy() | broad_line(d, "halpha") | broad_line(d, "hbeta")
    ref = pd.DataFrame({"targetid": d["targetid"].to_numpy(), "ra": d["target_ra"].to_numpy(float),
                        "dec": d["target_dec"].to_numpy(float), "z_ref": d["z"].to_numpy(float), "z_type": "spec",
                        "is_agn": is_agn, "spectype": d["spectype"].to_numpy(),
                        "logmstar": d["logmstar"].to_numpy(float), "is_xray_agn": False})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(out)
    light = out.with_name(out.stem + "_positions.parquet")
    ref[["targetid", "ra", "dec"]].to_parquet(light)
    print(f"{len(ref)} spectra ({int(is_agn.sum())} AGN, of which {int((is_agn & (ref.spectype == 'GALAXY')).sum())} "
          f"broad-line GALAXY); wrote {out} and {light}")


if __name__ == "__main__":
    main()
