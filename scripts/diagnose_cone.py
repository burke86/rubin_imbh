"""Diagnose a ``--no-select --stats-only`` cone run: where do false positives come from?

    python scripts/diagnose_cone.py ~/ecdfs_stats_v3.parquet

Splits objects by DIA detection count (dia source: 1-2 = mostly non-variable reference,
>5 = mostly real variables) or by morphology (object source), and by direct-image
magnitude, and reports the 5-sigma fraction and the fractional rms of the flagged objects
in each bin. A false-positive rate that rises with brightness, with small fvar, points to a
systematic floor (--sys-frac).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.calib import calibration_groups  # noqa: E402

SIG = 5.0
BANDS = "griz"
MAG_BINS = [-np.inf, 18, 19, 20, 21, 22, 23, 24, 25, np.inf]

d = pd.read_parquet(sys.argv[1])
d["grp"] = calibration_groups(d)[0]
print(f"objects: {len(d)}")

pd.set_option("display.width", 200)
for b in BANDS:
    ok = d[f"{b}_nEpoch"] >= 10
    x = d[ok].copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        x["mag"] = 31.4 - 2.5 * np.log10(x[f"{b}_medFlux"])  # nJy -> AB
    x["magbin"] = pd.cut(x["mag"], MAG_BINS)
    x["var"] = x[f"{b}_varSigClip"] >= SIG
    x["fvar_var"] = x[f"{b}_fvar"].where(x["var"])
    t = x.groupby(["grp", "magbin"], observed=True).agg(
        n=("var", "size"),
        frac5=("var", "mean"),
        med_chi2=(f"{b}_chi2dof", "median"),
        med_fvar_flagged=("fvar_var", "median"),
    )
    print(f"\n=== {b}-band (nEpoch >= 10: {len(x)}; mag from median direct-image flux)")
    print(t[t.n >= 20].round(3).to_string())

if "all_varSigClip" in d:
    x = d[d["all_dof"] >= 4].copy()
    ref_b = "r" if "r_medFlux" in x else "i"
    with np.errstate(invalid="ignore", divide="ignore"):
        x["mag"] = 31.4 - 2.5 * np.log10(x[f"{ref_b}_medFlux"])
    x["magbin"] = pd.cut(x["mag"], MAG_BINS)
    x["var"] = x["all_varSigClip"] >= SIG
    t = x.groupby(["grp", "magbin"], observed=True).agg(
        n=("var", "size"), frac5=("var", "mean"), med_chi2=("all_chi2dof", "median"), med_dof=("all_dof", "median"))
    print(f"\n=== combined-band statistic (all_dof >= 4: {len(x)}; mag from {ref_b}-band median direct-image flux)")
    print(t[t.n >= 20].round(3).to_string())

print("\nisVariable by group:")
print(d.groupby("grp", observed=True).isVariable.agg(["size", "sum", "mean"]).round(3).to_string())
print("\nnVarBands among variables:")
print(d[d.isVariable].nVarBands.value_counts().sort_index().to_string())
